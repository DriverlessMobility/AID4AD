<div align="center">
  <h1>AID4AD</h1>
  
  <h3> AID4AD: Aerial Image Data for Automated Driving Perception </h3>
  
  [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2508.02140)
</div>

## Introduction

This repository provides the official implementation and evaluation framework for the work  
**“AID4AD: Aerial Image Data for Automated Driving Perception”**.

AID4AD introduces a high-precision dataset and experimental pipeline for evaluating aerial imagery application in automated driving. It enables reproducible research on online map construction, motion prediction, and broader perception tasks using aerial imagery precisely aligned to the nuScenes dataset.

---

## 📈 Results

The following pre-trained models are available to reproduce the main results from the paper:

- SatforHDMap (Map Construction) (TBD)
- StreamMapNet (Map Construction) (TBD)
- HiVT (Motion Prediction) (TBD)

---

## 📦 Dataset

1. Download the dataset archive AID4AD_tiles.zip (TBD) and extract the `SatImgTiles/`, `offset_grid_data/` and `annotation_files/` folders into  
   [`AID4AD_tools/`](AID4AD_tools).
2. Generate the full-area images:
   ```bash
   bash create_dataset.sh
   ```
3. Export frame-wise aerial crops aligned to ego-vehicle coordinates:
   ```bash
   bash export_frames.sh
   ```

---

## 🗂️ Repository Structure

The following folders must be added to the repository by placing the extracted contents of the downloaded checkpoints and datasets into the respective locations.
Note: All included repositories are configured to expect the nuScenes dataset in the path indicated below within the unified repository root, which may differ from the original instructions in each algorithm's repository.

```
AID4AD/
├── nuScenes                     
├── AID4AD_tools
│   ├── annotation_files/    
│   └── SatImgTiles/               
├── MapBEVPrediction_modified/
│   ├── HiVT/
│   │   └── checkpoints/           
│   ├── StreamMapNet_modified/
│   │   └── checkpoints/            
│   ├── trj_data_AID/               
│   └── trj_data_AID_only/       
├── SatforHDMap_modified/
│   └── checkpoints/
```

---

## 🛰️ SatforHDMap Evaluation (Online Map Construction)

1. Set up the environment following the instructions in  
   [`SatforHDMap_modified/README.md`](SatforHDMap_modified/README.md)
2. Place the checkpoint files into  
   [`SatforHDMap_modified/checkpoints/`](SatforHDMap_modified/checkpoints/)
3. Run [create_Sat4HD_json.py](AID4AD_tools/scripts/create_Sat4HD_json.py)
4. Run the evaluation script:  
   ```bash
   bash run_test.sh
   ```

---

## 🗺️ StreamMapNet (Online Map Construction)

1. Set up the environment via  
   [`MapBEVPrediction_modified/README.md`](MapBEVPrediction_modified/README.md)
2. Add checkpoints to:  
   [`StreamMapNet_modified/checkpoints/`](MapBEVPrediction_modified/StreamMapNet_modified/checkpoints/)
3. Run inference:
   ```bash
   bash test_AID4AD.sh
   ```

---

## 🔮 HiVT (Motion Prediction)

You can either generate or download pre-computed BEV features from here (TBD).

### Option A: Download prepared data

1. Merge and extract archive chunks:
   ```bash
   zip --fix trj_data_AID --out joined-trj_data_AID.zip
   unzip joined-trj_data_AID.zip
   ```
2. Place the `trj_data_AID/` and `trj_data_AID_only/` folders into  
   `MapBEVPrediction_modified/`
3. Copy HiVT checkpoints into:  
   `MapBEVPrediction_modified/HiVT/checkpoints/`

### Option B: Generate BEV features manually

1. Save BEV features via:
   ```bash
   bash test_save_bev.sh
   ```
2. Merge predictions with:
   ```bash
   bash adaptor/merge.bash
   bash adaptor/merge_bev.bash
   ```
3. Add HiVT checkpoints to  
   `MapBEVPrediction_modified/HiVT/checkpoints/`

### Evaluate via:

- `test_GT_map.sh`  
- `test_mapless.sh`  
- `test_AID4AD_only.sh`  
- `test_AID4AD_combined.sh`  

(All located in `MapBEVPrediction_modified/HiVT/`)

---

## 📄 Citation

If you use AID4AD in your research, please cite:

```bibtex
@misc{lengerer2025aid4adaerialimagedata,
      title={AID4AD: Aerial Image Data for Automated Driving Perception}, 
      author={Daniel Lengerer and Mathias Pechinger and Klaus Bogenberger and Carsten Markgraf},
      year={2025},
      eprint={2508.02140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02140}, 
}
```

---

## 📜 License

This repository is released under the **Apache 2.0** license.

The **AID4AD dataset**, including the offset-grid-based mapping between the nuScenes local coordinate system and aerial imagery, as well as all associated scripts, is licensed under  
**Creative Commons CC-BY-NC-SA 4.0**.

To support reproducibility, we include aerial image tiles extracted using Google Earth Pro, along with scripts to generate per-frame views from them.

> Use of the aerial imagery remains subject to the [Google Earth Terms of Service](https://earthengine.google.com/terms/) and [Google Attribution Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/).  
> Please ensure proper attribution when using or displaying imagery.