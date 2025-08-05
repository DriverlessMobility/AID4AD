# if method is 'bev', choose [MapTR, MapTRv2, MapTRv2_CL, StreamMapNet]
CUDA_VISIBLE_DEVICES=1 python eval.py \
  --root ../trj_data_AID_only/stream \
  --split val \
  --method bev \
  --map_model StreamMapNet \
  --batch_size 32 \
  --ckpt_path checkpoints/hivt_aid4ad_only.ckpt