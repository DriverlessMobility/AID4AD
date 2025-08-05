# SatForHDMap

**Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction (ICRA2024)**

**[[Paper](https://arxiv.org/abs/2308.15427)]**

**Abstract:**

High-definition (HD) maps play a crucial role in autonomous driving systems. Recent methods have attempted to construct HD maps in real-time using vehicle onboard sensors. Due to the inherent limitations of onboard sensors, which include sensitivity to detection range and susceptibility to occlusion by nearby vehicles, the performance of these methods significantly declines in complex scenarios and long-range detection tasks. In this paper, we explore a new perspective that boosts HD map construction through the use of satellite maps to complement onboard sensors. We initially generate the satellite map tiles for each sample in nuScenes and release a complementary dataset for further research. To enable better integration of satellite maps with existing methods, we propose a hierarchical fusion module, which includes feature-level fusion and BEV-level fusion. The feature-level fusion, composed of a mask generator and a masked cross-attention mechanism, is used to refine the features from onboard sensors. The BEV-level fusion mitigates the coordinate differences between features obtained from onboard sensors and satellite maps through an alignment module. The experimental results on the augmented nuScenes showcase the seamless integration of our module into three existing HD map construction methods. The satellite maps and our proposed module notably enhance their performance in both HD map semantic segmentation and instance detection tasks.

### Note
We applied our module to the [HDMapNet project](https://github.com/Tsinghua-MARS-Lab/HDMapNet) and parallelized the HDMapNet code for DDP.

### Preparation
1. Follow the [HDMapNet documentation](README_HDMapNet.md), download the nuScenes dataset and install the dependencies.

2. Download the [SatelliteMapTiles(complement for nuScenes)](https://www.kaggle.com/datasets/wjgao0101/satfornuscenes) and put it to `satmap/` folder.

**The final folder structure**
```
SatForHDMap
|-- data/
|-- evaluation/
|-- icon/
|-- model/
|-- postprocess/
|-- preprocess/
|-- dataset/
│   ├── nuScenes_trainval/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
|   |   ├── v1.0-trainval/
|-- satmap
│   ├── map/
│   ├── prior_map_trainval/
│   │   ├── map_prior.json
│   ├── prior_map_test/
```

### Training

Run `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --instance_seg --direction_pred --fusion_mode seg-masked-atten --align_fusion`.

### Evaluation
Follow the [HDMapNet documentation](README_HDMapNet.md).

### Citation
If you found this paper or codebase useful, please cite our paper:
```
@misc{gao2024complementing,
      title={Complementing Onboard Sensors with Satellite Map: A New Perspective for HD Map Construction}, 
      author={Wenjie Gao and Jiawei Fu and Yanqing Shen and Haodong Jing and Shitao Chen and Nanning Zheng},
      year={2024},
      eprint={2308.15427},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```