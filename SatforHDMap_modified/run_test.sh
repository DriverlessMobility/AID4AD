# ___ New split ___

echo "Running test on AID4AD_newsplit"
python test.py \
   --instance_seg --direction_pred \
   --fusion_mode seg-masked-atten --align_fusion \
   --logdir checkpoints/AID4AD_newsplit \
   --modelf checkpoints/AID4AD_newsplit/best.pt \
   --prior_map_root ../AID4AD_ego_referenced \
   --is_newsplit

echo "Running test on Sat4HD_newsplit"
python test.py \
   --instance_seg --direction_pred \
   --fusion_mode seg-masked-atten --align_fusion \
   --logdir checkpoints/Sat4HD_newsplit \
   --modelf checkpoints/Sat4HD_newsplit/best.pt \
   --prior_map_root ./satmap/satellite_map_trainval \
   --is_newsplit

# ___ Old split ___

echo "Running test on AID4AD_oldsplit"
python test.py \
   --instance_seg --direction_pred \
   --fusion_mode seg-masked-atten --align_fusion \
   --logdir checkpoints/AID4AD_oldsplit \
   --modelf checkpoints/AID4AD_oldsplit/best.pt \
   --prior_map_root ../AID4AD_ego_referenced
