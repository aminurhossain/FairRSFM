# MMEarth-20k: Global Dynamic World Semantic Segmentation

This directory contains the complete, self-contained implementation, dataset specifications, pretrained checkpoints, and evaluation metrics for **MMEarth-20k Dynamic World Semantic Segmentation** using IBM/NASA **Prithvi 2.0 EO 300M TL**.

All components (data splits, HDF5 datasets, DataModules, training methods, output checkpoints, and evaluation logs) are strictly self-contained within `segmentation/mmearth_20k/`.

---

## 📁 Directory Structure

```
segmentation/mmearth_20k/
├── README.md                                             # MMEarth-20k Documentation & Execution Guide
├── train_mmearth_dynamic_world_prithvi_super.py         # Baseline Linear Probing (LP) Training & Eval
├── train_mmearth_dynamic_world_prithvi_super_BOLP.py    # Biome-Orthogonal Linear Probing (BOLP)
├── train_mmearth_dynamic_world_prithvi_super_DBR.py     # Dynamic Biome Reweighting (DBR)
├── mmearth_dynamic_world_datamodule.py                  # PyTorch Lightning DataModule
├── hardcoded/                                            # Dataset HDF5, Tile Info, Splits & Stats
│   ├── data_20k_fixed.h5                                 # 20k HDF5 dataset file
│   ├── data_20k_fixed_splits.json                        # Train/Val/Test fixed split indices
│   ├── data_20k_fixed_band_stats.json                    # Band normalization statistics
│   └── data_20k_fixed_tile_info.json                     # Georeferenced tile metadata
├── output/                                               # Outputs, Checkpoints & Test Metric Logs
│   ├── normal_lp/                                        # Baseline LP Checkpoints & Metrics
│   │   ├── checkpoints/
│   │   ├── test_metrics.json
│   │   └── test_biome_metrics.csv
│   ├── bolp_k5/                                          # BOLP (k=5) Checkpoints & Metrics
│   │   ├── checkpoints/
│   │   └── test_metrics.json
│   └── dbr_alpha0.5_ema0.8/                              # DBR Checkpoints & Metrics
│       ├── checkpoints/
│       └── test_metrics.json
└── all_files_dataset/                                    # Dataset Creation & Analysis Utilities
    ├── 20k_hard_coded.py
    ├── 20k_om_imbalanced.py
    ├── analyze_mmearth_biomes.py
    ├── check_mmearth_dynamic_world_loader.py
    ├── geobench_prithvi_datamodule.py
    ├── make_mmearth20k_imbalanced_subset.py
    ├── make_mmearth20k_random_splits.py
    ├── make_mmearth5k_subset.py
    ├── plot_mmearth20k_split_biomes.py
    ├── data_20k_imbalanced.h5
    ├── data_20k_imbalanced_splits.json
    ├── data_20k_imbalanced_tile_info.json
    ├── data_20k_imbalanced_band_stats.json
    └── data_20k_imbalanced_subset_manifest.json
```

---

## 🚀 Key Methods & Training Commands

### 1. Baseline Linear Probing (LP)
Trains a linear head on top of frozen Prithvi 2.0 features for Dynamic World 10-class segmentation:
```bash
python segmentation/mmearth_20k/train_mmearth_dynamic_world_prithvi_super.py \
    --data-root segmentation/mmearth_20k/hardcoded \
    --out-dir segmentation/mmearth_20k/output
```

### 2. Biome-Orthogonal Linear Probing (BOLP)
Mitigates geographic sub-biome representation bias during feature projection:
```bash
python segmentation/mmearth_20k/train_mmearth_dynamic_world_prithvi_super_BOLP.py \
    --data-root segmentation/mmearth_20k/hardcoded \
    --out-dir segmentation/mmearth_20k/output \
    --bolp-k 5
```

### 3. Dynamic Biome Reweighting (DBR)
Dynamically reweights loss functions during training using validation feedback:
```bash
python segmentation/mmearth_20k/train_mmearth_dynamic_world_prithvi_super_DBR.py \
    --data-root segmentation/mmearth_20k/hardcoded \
    --out-dir segmentation/mmearth_20k/output \
    --dbr-alpha 0.5 \
    --dbr-ema 0.8
```

---

## 📊 Evaluation & Testing

To test a trained checkpoint:
```bash
python segmentation/mmearth_20k/train_mmearth_dynamic_world_prithvi_super_BOLP.py \
    --ckpt-path segmentation/mmearth_20k/output/bolp_k5/checkpoints/last.ckpt \
    --data-root segmentation/mmearth_20k/hardcoded
```

---

## 📌 Dataset Notes
- **Input Bands**: Sentinel-2 L2A (12 bands) + Sentinel-1 RTC (2 bands) + DEM + Dynamic World Target Mask (10 land-cover classes).
- **Target Resolution**: 128x128 patches georeferenced across global biomes.
