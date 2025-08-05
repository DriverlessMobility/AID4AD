from PIL import Image
from tqdm import tqdm  # Progress bar
import os

# Allow processing of very large images
Image.MAX_IMAGE_PIXELS = None

# Directory pattern for basemap tiles
TILE_DIR_TEMPLATE = "SatImgTiles/{basemap_name}"

# Define geographic boundaries: (bottom, left, top, right)
AREA_CORNERS = {
    'singapore-queenstown': (1.2781458568639803, 103.76739890494545, 1.311490523682109, 103.79640946097894),
    'singapore-onenorth': (1.2881102566234457, 103.78473913384309, 1.3064236036272767, 103.79898649967284),
    'singapore-hollandvillage': (1.2992851171846798, 103.78217031432737, 1.3257186639669583, 103.8074044117939),
    'boston-seaport': (42.336823030252226, -71.05781902966984, 42.3558855637424, -71.02165456745591)
}

# Tile coordinates in the stitched image
TOP_LEFT_TILE_COORDS = {
    'singapore-queenstown': [(0, 0), (7804, 0), (15608, 0), (0, 7856), (7804, 7856), (15608, 7856),
                             (0, 15712), (7804, 15712), (15608, 15712), (0, 23568), (7804, 23568), (15608, 23568)],
    'singapore-onenorth': [(0, 0), (7804, 0), (0, 7856), (7804, 7856)],
    'singapore-hollandvillage': [(0, 0), (7804, 0), (15608, 0), (0, 7856), (7804, 7856), (15608, 7856),
                                 (0, 15712), (7804, 15712), (15608, 15712)],
    'boston-seaport': [(0, 0), (7791, 1), (15583, 3), (0, 7820), (7789, 7822), (15580, 7824)]
}

# Full image sizes after stitching
STITCHED_IMAGE_SIZES = {
    'singapore-queenstown': (23800, 31760),
    'singapore-onenorth': (15996, 16048),
    'singapore-hollandvillage': (23800, 23904),
    'boston-seaport': (23772, 16016)
}

# Crop boundaries (left, upper, right, lower)
CROP_BOXES = {
    'singapore-queenstown': (194, 5259, 23100, 31592),
    'singapore-onenorth': (194, 1417, 11444, 15881),
    'singapore-hollandvillage': (194, 2861, 20118, 23737),
    'boston-seaport': (200, 772, 21307, 15827)
}

BASEMAP_SIZES = {
    'singapore-queenstown': (3228.6, 3687.1),
    'singapore-onenorth': (1585.6, 2025.0),
    'singapore-hollandvillage': (2808.3, 2922.9),
    'boston-seaport': (2979.5, 2118.1)    
}


def stitch_and_resize_basemaps():
    for basemap_name, tile_positions in TOP_LEFT_TILE_COORDS.items():
        print(f"\nProcessing {basemap_name}...")

        tile_dir = TILE_DIR_TEMPLATE.format(basemap_name=basemap_name)
        stitched_img = Image.new('RGB', STITCHED_IMAGE_SIZES[basemap_name], (255, 255, 255))

        for i, (x_offset, y_offset) in enumerate(tqdm(tile_positions, desc=f"Stitching tiles ({basemap_name})", unit="tile")):
            tile_path = os.path.join(tile_dir, f"{i}.png")
            with Image.open(tile_path) as tile:
                stitched_img.paste(tile, (x_offset, y_offset))

        stitched_img = stitched_img.crop(CROP_BOXES[basemap_name])

        target_size = [round(val/0.15) for val in BASEMAP_SIZES[basemap_name]]

        stitched_img = stitched_img.resize(target_size, Image.Resampling.LANCZOS)

        output_path = os.path.join(tile_dir, "stitched_new.png")
        stitched_img.save(output_path)
        print(f"Saved stitched image to: {output_path}")


if __name__ == "__main__":
    stitch_and_resize_basemaps()
