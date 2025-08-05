#export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/StreamMapNet_modified"
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD.py \
    checkpoints/AID4AD_combined/best.pth \
    --eval 
    #--bev_path path_to_save_bev_features
