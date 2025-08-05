# if method is 'bev', choose [MapTR, MapTRv2, MapTRv2_CL, StreamMapNet]
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --root ../trj_data_AID/stream \
  --split val \
  --method base \
  --map_model MapTR \
  --batch_size 32 \
  --ckpt_path checkpoints/hivt_gtmap.ckpt