import os
import json
from glob import glob
from tqdm import tqdm
from nuscenes import NuScenes

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f_json:
        try:
            return json.load(f_json)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for file {json_path}. Skipping...")
            return None

# Paths
nusc = NuScenes(version="v1.0-trainval", dataroot="../nuScenes", verbose=False)
nuAID_image_dir = "../AID4AD_ego_referenced"
map_prior_json_path = "../SatforHDMap_modified/satmap/satellite_map_trainval/map_prior.json"
output_json_path = os.path.join(nuAID_image_dir, "map_prior.json")

# Load files and map them by sample_token
file_paths = glob(os.path.join(nuAID_image_dir, "*", "*.png"))
file_paths = [file for file in file_paths if "overlay" not in file and "basemap" not in file]
nuAID_files_by_sample_token = {}

for file_path in file_paths:
    sample_token = os.path.splitext(os.path.basename(file_path))[0].split("_")[1]
    if sample_token in nuAID_files_by_sample_token:
        print(f"Duplicate sample token found: {sample_token}")
    else:
        nuAID_files_by_sample_token[sample_token] = file_path

# Load original map_prior.json
json_data = load_json(map_prior_json_path)
nuAID_json = {}

# Replace image paths with nuAID image paths
for key, _ in tqdm(json_data.items(), desc="Adapting paths"):
    sample_token = nusc.get('sample_data', key)['sample_token']
    rel_path = nuAID_files_by_sample_token[sample_token]
    rel_path = os.path.join(os.path.basename(os.path.dirname(rel_path)), os.path.basename(rel_path))
    nuAID_json[key] = rel_path

# Save to new json
with open(output_json_path, 'w') as f:
    json.dump(nuAID_json, f, indent=4)

print(f"Saved adapted map_prior.json to: {output_json_path}")
