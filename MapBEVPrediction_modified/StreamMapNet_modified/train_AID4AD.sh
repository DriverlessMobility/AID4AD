#export PYTHONPATH="${PYTHONPATH}:/MapBEVPrediction/StreamMapNet_modified"
CUDA_VISIBLE_DEVICES=2 python tools/train.py plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD.py \
    --no-validate --deterministic --work-dir work_dirs/nusc_newsplit_480_60x30_24e_AID4AD
