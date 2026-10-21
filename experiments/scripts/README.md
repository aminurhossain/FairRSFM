# Methods Guide: Debiasing Algorithms & Training

This directory contains the training and debiasing implementations for the 48 benchmark configurations (3 Foundation Models × 4 Datasets × 4 Methods).

---

## 🧠 Debiasing Algorithms

| Method | Type | Description | Key Hyperparameters |
| :--- | :--- | :--- | :--- |
| **ERM** | Baseline | Standard Empirical Risk Minimization / Linear Probing with uniform sample weighting. | `--lr`, `--weight_decay` |
| **GroupDRO** | Minimax Optimization | Group Distributionally Robust Optimization. Maintains online exponentiated group weights updated via step size $\eta_q$: $q_g \leftarrow q_g \exp(\eta_q \ell_g)$. | `--group_dro_step_size` (0.01) |
| **DBR** | Dynamic Reweighting | Dynamic Biome Reweighting. Updates running exponential moving average (EMA) group validation losses and reweights loss during training: $w_g \propto \exp(\alpha \cdot \bar{\ell}_g)$. | `--dbr_alpha` (0.5), `--dbr_ema` (0.8 - 0.9) |
| **BOLP** | Subspace Decomposition | Biome-Orthogonal Linear Probing. Projects feature representations into a biome-invariant subspace while isolating rank-$k$ biome-specific variations: $W_g = W_{\text{shared}} + U_g V_g^\top$. | `--bolp_k` (1 - 5) |

---

## 📁 Directory Structure

```text
methods/
├── prithvi/
│   ├── classification/
│   │   ├── bigearthnet/       # bigearthnet_erm.py, bigearthnet_groupdro.py, bigearthnet_dbr.py, bigearthnet_bolp.py
│   │   └── eurosat/           # eurosat_erm.py, eurosat_groupdro.py, eurosat_dbr.py, eurosat_bolp.py
│   └── segmentation/
│       ├── sa_crop/           # sa_crop_erm.py, sa_crop_groupdro.py, sa_crop_dbr.py, sa_crop_bolp.py
│       └── mmearth_20k/       # mmearth_erm.py, mmearth_groupdro.py, mmearth_dbr.py, mmearth_bolp.py
│
├── satmae/
│   ├── bigearthnet/           # bigearthnet_erm.py, bigearthnet_groupdro.py, bigearthnet_dbr.py, bigearthnet_bolp.py
│   ├── eurosat/               # eurosat_erm.py, eurosat_groupdro.py, eurosat_dbr.py, eurosat_bolp.py
│   ├── sa_crop/               # sa_crop_erm.py, sa_crop_groupdro.py, sa_crop_dbr.py, sa_crop_bolp.py
│   └── mmearth/               # mmearth_erm.py, mmearth_groupdro.py, mmearth_dbr.py, mmearth_bolp.py
│
└── dofa/
    ├── bigearthnet/           # erm.py, groupdro.py, dbr.py, bolp.py
    ├── eurosat/               # erm.py, groupdro.py, dbr.py, bolp.py
    ├── sa_crop/               # Linear probe (erm.py, groupdro.py, dbr.py, bolp.py)
    ├── sa_crop_upernet/       # Dense UPerNet decoder (erm.py, groupdro.py, dbr.py, bolp.py)
    ├── mmearth/               # Linear probe (erm.py, groupdro.py, dbr.py, bolp.py)
    └── mmearth_upernet/       # Dense UPerNet decoder (erm.py, groupdro.py, dbr.py, bolp.py)
```

---

## 🚀 Execution Examples

### 1. Prithvi EO V2 300M

```bash
# EuroSAT ERM
python methods/prithvi/classification/eurosat/eurosat_erm.py \
    --seed 42 \
    --lr 1e-3 \
    --max_epochs 50

# BigEarthNet GroupDRO
python methods/prithvi/classification/bigearthnet/bigearthnet_groupdro.py \
    --seed 42 \
    --group_dro_step_size 0.01 \
    --max_epochs 50

# South Africa Crop Type BOLP (Segmentation)
python methods/prithvi/segmentation/sa_crop/sa_crop_bolp.py \
    --seed 42 \
    --bolp_k 1 \
    --max_epochs 50
```

### 2. SatMAE (Stanford SustainLab)

```bash
# BigEarthNet DBR
python methods/satmae/bigearthnet/bigearthnet_dbr.py \
    --seed 42 \
    --dbr_alpha 0.5 \
    --dbr_ema 0.9 \
    --batch_size 64

# South Africa Crop Type ERM (Custom 1x1 Conv Linear Probe)
python methods/satmae/sa_crop/sa_crop_erm.py \
    --seed 42 \
    --batch_size 32 \
    --lr 1e-3
```

### 3. DOFA (Dynamic One-For-All)

```bash
# EuroSAT BOLP
python methods/dofa/eurosat/bolp.py \
    --seed 42 \
    --bolp_k 2 \
    --batch_size 128

# MMEarth UPerNet GroupDRO
python methods/dofa/mmearth_upernet/groupdro.py \
    --seed 42 \
    --group_dro_step_size 0.01 \
    --batch_size 16
```

---

## ⚙️ Standard Training Arguments

All modular scripts accept standard command-line options:
* `--seed`: Random seed for reproducibility (e.g. 42).
* `--batch_size`: Batch size per GPU.
* `--lr`: Initial learning rate (typically `1e-3` for linear probe, `1e-4` for decoders).
* `--weight_decay`: L2 regularization coefficient (default `1e-4`).
* `--max_epochs`: Maximum number of training epochs.
* `--out_dir`: Custom directory to save logs, metrics, and checkpoints.
