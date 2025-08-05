# if method is 'bev', choose [MapTR, MapTRv2, MapTRv2_CL, StreamMapNet]
CUDA_VISIBLE_DEVICES=0 python train.py \
  --root ../trj_data_AID/stream \
  --method mapless \
  --map_model MapTR  \
  --embed_dim 128 \
  --gpus 1 \
  --num_workers 32 \
  --max_epochs 96