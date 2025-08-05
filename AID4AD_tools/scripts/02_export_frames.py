import argparse
import logging
import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from nuscenes.eval.common.utils import Quaternion
from helpers.geostuff import calc_new_latlon_w_offset
from helpers.offset_grid_helpers import load_offset_grids, get_offset_for_coordinate

Image.MAX_IMAGE_PIXELS = None


class SatImageGenerator:
    """
    Class for generating ego-aligned satellite image crops with optional basemap and overlay support.
    """

    def __init__(self, args, basemap_names, satmap_origins, offset_grids=None):
        self.args = args
        self.basemap_names = basemap_names
        self.satmap_origins = satmap_origins
        self.offset_grids = offset_grids

        if self.args.crop_size_meters is None:
            raise ValueError("--crop_size_meters is mandatory but not provided.")

        self._check_crop_args_consistency()

        self.half_w_m = self.args.crop_size_meters[0] / 2.0
        self.half_h_m = self.args.crop_size_meters[1] / 2.0
        self._valid_final_pixels = self.args.final_image_size_pixels

        self.basemap_images = {}
        self.satmap_images = {}
        self.frame_counter_per_map = {basemap: 0 for basemap in basemap_names}
        self.frame_counter_lock = threading.Lock()

        for basemap in basemap_names:
            basemap_path = Path(self.args.basemap_path) / f"{basemap}.png"
            satmap_path  = Path(self.args.satmap_path) / basemap / "stitched_new.png"

            self.basemap_images[basemap] = self._load_image(basemap_path)
            self.satmap_images[basemap]  = self._load_image(satmap_path)

    def run(self):
        for split in self.args.splits:
            path = Path(self.args.annotation_pickle_path) / f"nuscenes_map_infos_temporal_{split}_newsplit.pkl"
            data = self._load_pickle(path)
            all_infos = data["infos"]
            logging.info(f"Loaded {len(all_infos)} frames from '{path}'.")
            self._process_split(all_infos, split)

    def _process_split(self, annotations, split):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._process_frame, annotations, i) for i in range(len(annotations))]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing frames for {split}"):
                try:
                    future.result()
                except Exception as e:
                    import traceback
                    logging.error(f"Error processing frame: {e}")
                    traceback.print_exc()

        if not self.args.dry_run:
            out_dir = self.args.per_frame_output_path
            filename = f"nuscenes_map_infos_temporal_{split}_newsplit.pkl"
            out_path = Path(out_dir) / filename
            with open(out_path, "wb") as f:
                pickle.dump({"infos": annotations}, f)
            logging.info(f"Saved updated annotations for '{split}' to: {out_path}")

    def _process_frame(self, annotations, frame_idx):
        frame = annotations[frame_idx]
        map_name = frame["map_location"]
        if map_name not in self.basemap_images:
            return

        translation, rotation = self._calc_lidar_to_basemap_offset(frame, ref_frame=self.args.reference_frame)
        raw_x, raw_y = translation[0], translation[1]

        offset_x, offset_y = 0.0, 0.0
        if self.offset_grids is not None:
            try:
                offset_x, offset_y = get_offset_for_coordinate(
                    orig_x=raw_x,
                    orig_y=raw_y,
                    basemap_name=map_name,
                    offset_grids=self.offset_grids,
                    resolution=5.0
                )
            except ValueError:
                pass

        corrected_x = raw_x - offset_x
        corrected_y = raw_y + offset_y
        frame["basemap2sat_offset"] = [-offset_x, +offset_y]

        yaw = self._quaternion_to_yaw(Quaternion(rotation))
        heading_deg = np.degrees(yaw)

        with self.frame_counter_lock:
            counter = self.frame_counter_per_map[map_name]
            filename = f"{counter}_{frame['token']}.png"
            self.frame_counter_per_map[map_name] += 1

        output_path = Path(self.args.per_frame_output_path) / map_name / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cropped_final = self._crop_then_rotate(corrected_x, corrected_y, heading_deg, self.satmap_images[map_name], scaling=self.args.img_scaling)
        if self._valid_final_pixels is not None:
            cropped_final = cropped_final.resize(self._valid_final_pixels, Image.BILINEAR)

        if not self.args.dry_run:
            cropped_final.save(output_path)

        if self.args.crop_basemap:
            dbg_output = output_path.with_name(output_path.stem + "_basemap.png")
            dbg_crop = self._crop_then_rotate(raw_x, raw_y, heading_deg, self.basemap_images[map_name], scaling=0.1)
            if self._valid_final_pixels is not None:
                dbg_crop = dbg_crop.resize(self._valid_final_pixels, Image.BILINEAR)
            elif not self.args.dry_run:
                size = (round(2 * self.half_w_m / self.args.img_scaling), round(2 * self.half_h_m / self.args.img_scaling))
                dbg_crop = dbg_crop.resize(size, Image.BILINEAR)
            if not self.args.dry_run:
                dbg_crop.save(dbg_output)

        if self.args.create_map_overlay:
            overlay_path = output_path.with_name(output_path.stem + "_map_overlay.png")
            self._save_map_overlay(frame, overlay_path)

    def _crop_then_rotate(self, ego_x_m: float, ego_y_m: float, heading_deg: float, big_img: Image.Image, scaling: float = 0.1) -> Image.Image:
        pivot_x_px = ego_x_m / scaling
        pivot_y_px = big_img.size[1] - (ego_y_m / scaling)

        half_w_px = self.half_w_m / scaling
        half_h_px = self.half_h_m / scaling

        overshoot = 2
        sub_w_half = max(half_w_px, half_h_px) * overshoot
        sub_h_half = sub_w_half

        left = max(0, pivot_x_px - sub_w_half)
        top = max(0, pivot_y_px - sub_h_half)
        right = min(big_img.size[0], pivot_x_px + sub_w_half)
        bottom = min(big_img.size[1], pivot_y_px + sub_h_half)

        sub_img = big_img.crop((left, top, right, bottom))
        pivot_sub_x = pivot_x_px - left
        pivot_sub_y = pivot_y_px - top

        rot_img = sub_img.rotate(-heading_deg, center=(pivot_sub_x, pivot_sub_y), expand=False)

        final_crop = rot_img.crop((
            pivot_sub_x - half_w_px,
            pivot_sub_y - half_h_px,
            pivot_sub_x + half_w_px,
            pivot_sub_y + half_h_px
        ))
        return final_crop

    def _save_map_overlay(self, frame_annotations, save_path: Path):
        try:
            if self.args.reference_frame == "lidar":
                size = (round(self.args.crop_size_meters[0] / self.args.img_scaling),
                        round(self.args.crop_size_meters[1] / self.args.img_scaling))
            elif self.args.reference_frame == "ego":
                size = (round(self.args.crop_size_meters[1] / self.args.img_scaling),
                        round(self.args.crop_size_meters[0] / self.args.img_scaling))
            else:
                raise ValueError("Unknown reference frame.")
            
            image = Image.new('RGBA', size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

            for key, vectors in frame_annotations.get('annotation', {}).items():
                color = {
                    'divider': 'red',
                    'ped_crossing': 'green',
                    'boundary': 'blue',
                    'centerline': 'yellow'
                }.get(key, 'white')

                for vector in vectors:
                    coords = self._ego_to_image_coords(vector, size[1], size[0], scale=1.0/self.args.img_scaling)
                    draw.line(coords.flatten().tolist(), fill=color, width=2)

            if self._valid_final_pixels is not None:
                image = image.resize(self._valid_final_pixels, Image.BILINEAR)

            if self.args.reference_frame == "ego":
                image = image.rotate(-90, expand=True)

            if not self.args.dry_run:
                image.save(save_path)
        except Exception as e:
            logging.error(f"Error creating map overlay: {e}")

    def _ego_to_image_coords(self, ego_coords: np.ndarray, height: int, width: int, scale: float) -> np.ndarray:
        x_px = (ego_coords[:, 0] * scale + width / 2)
        y_px = height - (ego_coords[:, 1] * scale + height / 2)
        return np.column_stack((x_px, y_px))

    @staticmethod
    def _calc_lidar_to_basemap_offset(frame, ref_frame="lidar") -> Tuple[List[float], List[float]]:
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(frame['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = frame['lidar2ego_translation']

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(frame['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = frame['ego2global_translation']

        lidar2global = ego2global @ lidar2ego

        if ref_frame == "lidar":
            return list(lidar2global[:3, 3]), list(Quaternion(matrix=lidar2global).q)
        elif ref_frame == "ego":
            return list(ego2global[:3, 3]), list(Quaternion(matrix=ego2global).q)
        else:
            raise ValueError("Unknown reference frame.")

    @staticmethod
    def _quaternion_to_yaw(quat: Quaternion) -> float:
        qw, qx, qy, qz = quat
        return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    
    def _check_crop_args_consistency(self):
        csm = self.args.crop_size_meters
        fisp = self.args.final_image_size_pixels
        if fisp is not None:
            ratio_m = csm[0] / csm[1]
            ratio_px = fisp[0] / fisp[1]
            if abs(ratio_m - ratio_px) > 1e-3:
                logging.warning(f"Aspect ratio mismatch (meters={ratio_m:.3f}, pixels={ratio_px:.3f}). Ignoring resize.")
                self.args.final_image_size_pixels = None
    
    @staticmethod
    def _load_image(image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _load_pickle(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)


# === ENTRY POINT ===
def get_args():
    parser = argparse.ArgumentParser(description="Export ego-aligned satellite images with offset corrections.")

    parser.add_argument("--annotation_pickle_path", type=str, required=True)
    parser.add_argument("--basemap_path", type=str, required=True)
    parser.add_argument("--satmap_path", type=str, required=True)
    parser.add_argument("--per_frame_output_path", type=str, required=True)
    parser.add_argument("--splits", nargs='+', default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--crop_size_meters", nargs=2, type=float, required=True)
    parser.add_argument("--final_image_size_pixels", nargs=2, type=int, default=None)
    parser.add_argument("--img_scaling", type=float, default=0.15)
    parser.add_argument("--offset_grid_dir", type=str, default=None)
    parser.add_argument("--reference_frame", type=str, default="lidar", choices=["lidar", "ego"])
    parser.add_argument("--crop_basemap", action="store_true")
    parser.add_argument("--create_map_overlay", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    basemap_names = [
        "boston-seaport",
        "singapore-queenstown",
        "singapore-hollandvillage",
        "singapore-onenorth"
    ]
    satmap_origins = {
        "boston-seaport": [42.336823030252226, -71.05781902966984],
        "singapore-queenstown": [1.2781458568639803, 103.76739890494545],
        "singapore-hollandvillage": [1.2992851171846798, 103.78217031432737],
        "singapore-onenorth": [1.2881102566234457, 103.78473913384309],
    }

    offset_grids = None
    if args.offset_grid_dir and os.path.isdir(args.offset_grid_dir):
        offset_grids = load_offset_grids(basemap_names, args.offset_grid_dir)
        logging.info(f"Loaded offset grids from: {args.offset_grid_dir}")

    generator = SatImageGenerator(args, basemap_names, satmap_origins, offset_grids)
    generator.run()


if __name__ == "__main__":
    main()
