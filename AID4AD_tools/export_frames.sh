python3 scripts/02_export_frames.py \
  --annotation_pickle_path "annotation_files" \
  --basemap_path "../nuScenes/maps/basemap" \
  --satmap_path "SatImgTiles" \
  --per_frame_output_path "../AID4AD_ego_referenced" \
  --splits train val \
  --crop_size_meters 60 30 \
  --offset_grid_dir "offset_grid_data" \
  --reference_frame ego \
  --crop_basemap \
  --create_map_overlay \
  #--final_image_size_pixels 400 200 \
  #--img_scaling 0.15 \
