# run once with val and once with train split in place of val in config
mkdir -p ./AID4AD_features/bev_val_features
mkdir -p ./AID4AD_features/bev_train_features
mkdir -p ./AID4AD_only_features/bev_val_features
mkdir -p ./AID4AD_only_features/bev_train_features

CUDA_VISIBLE_DEVICES=0 python tools/test_save_bev.py plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD_oldval.py \
    checkpoints/AID4AD_combined/best.pth --eval \
    --bev_path ./AID4AD_features/bev_val_features \
    --work-dir work_dirs/trainrun1

CUDA_VISIBLE_DEVICES=0 python tools/test_save_bev.py plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD_oldtrain.py \
    checkpoints/AID4AD_combined/best.pth --eval \
    --bev_path ./AID4AD_features/bev_train_features \
    --work-dir work_dirs/trainrun1

CUDA_VISIBLE_DEVICES=0 python tools/test_save_bev.py plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD_only_oldval.py \
    checkpoints/AID4AD_only/best.pth --eval \
    --bev_path ./AID4AD_only_features/bev_val_features \
    --work-dir work_dirs/nusc_newsplit_480_60x30_24e_AID4AD_only

CUDA_VISIBLE_DEVICES=0 python tools/test_save_bev.py plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD_only_oldtrain.py \
    checkpoints/AID4AD_only/best.pth --eval \
    --bev_path ./AID4AD_only_features/bev_train_features \
    --work-dir work_dirs/nusc_newsplit_480_60x30_24e_AID4AD_only