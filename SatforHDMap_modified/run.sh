CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 train.py \
    --instance_seg --direction_pred \
    --fusion_mode seg-masked-atten --align_fusion \
    --prior_map_root ../AID4AD_ego_referenced \
    --logdir ./runs/AID4AD_oldsplit_exactAlignment 

# CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 train.py \
#     --instance_seg --direction_pred \
#     --fusion_mode seg-masked-atten --align_fusion \
#     --prior_map_root ../AID4AD_ego_referenced \
#     --logdir ./runs/AID4AD_newsplit_exactAlignment \
#     --is_newsplit \
