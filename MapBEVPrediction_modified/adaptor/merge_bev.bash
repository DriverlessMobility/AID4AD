# AID4AD

python adaptor_bev.py \
  --scene_folder ../trj_data_AID/stream/val/data \
  --bev_folder ../StreamMapNet_modified/AID4AD_features/bev_val_features \
  --map_model StreamMapNet

python adaptor_bev.py \
  --scene_folder ../trj_data_AID/stream/train/data \
  --bev_folder ../StreamMapNet_modified/AID4AD_features/bev_train_features \
  --map_model StreamMapNet

# AID4AD only

python adaptor_bev.py \
  --scene_folder ../trj_data_AID_only/stream/val/data \
  --bev_folder ../StreamMapNet_modified/AID4AD_only_features/bev_val_features \
  --map_model StreamMapNet

python adaptor_bev.py \
  --scene_folder ../trj_data_AID_only/stream/train/data \
  --bev_folder ../StreamMapNet_modified/AID4AD_only_features/bev_train_features \
  --map_model StreamMapNet