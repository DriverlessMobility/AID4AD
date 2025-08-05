# AID4AD combined/fused features
python adaptor.py \
  --version trainval \
  --split train \
  --map_model StreamMapNet \
  --dataroot ../nuscenes \
  --index_file ../adaptor_files/traj_scene_frame_full_train.pkl \
  --map_file ../StreamMapNet_modified/AID4AD_features/bev_train_features_mapping_results.pickle \
  --gt_map_file ../adaptor_files/gt_full_train.pickle \
  --save_path ../trj_data_AID/stream

python adaptor.py \
  --version trainval \
  --split val \
  --map_model StreamMapNet \
  --dataroot ../nuscenes \
  --index_file ../adaptor_files/traj_scene_frame_full_val.pkl \
  --map_file ../StreamMapNet_modified/AID4AD_features/bev_val_features_mapping_results.pickle\
  --gt_map_file ../adaptor_files/gt_full_val.pickle \
  --save_path ../trj_data_AID/stream

# AID4AD only features
python adaptor.py \
  --version trainval \
  --split train \
  --map_model StreamMapNet \
  --dataroot ../nuscenes \
  --index_file ../adaptor_files/traj_scene_frame_full_train.pkl \
  --map_file ../StreamMapNet_modified/AID4AD_only_features/bev_train_features_mapping_results.pickle \
  --gt_map_file ../adaptor_files/gt_full_train.pickle \
  --save_path ../trj_data_AID_only/stream

python adaptor.py \
  --version trainval \
  --split val \
  --map_model StreamMapNet \
  --dataroot ../nuscenes \
  --index_file ../adaptor_files/traj_scene_frame_full_val.pkl \
  --map_file ../StreamMapNet_modified/AID4AD_only_features/bev_val_features_mapping_results.pickle\
  --gt_map_file ../adaptor_files/gt_full_val.pickle \
  --save_path ../trj_data_AID_only/stream
