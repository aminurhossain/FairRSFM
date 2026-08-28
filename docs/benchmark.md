# FairRSFM Benchmark Protocol

[Back to the main README](../README.md)

## Scope

FairRSFM evaluates whether remote sensing foundation model representations transfer consistently across ecological regimes. The benchmark augments four georeferenced downstream datasets with raw biome and macro-group metadata, freezes each pretrained encoder, and reports both conventional task scores and biome-aware robustness metrics.

The benchmark covers:

- single-label land-cover classification;
- multi-label land-cover classification;
- crop-type semantic segmentation; and
- dense land-cover semantic segmentation.

## Dataset Suite

| Dataset | Source suite | Task | Classes | Train | Validation | Test | Sensor / labels | Primary score |
|---|---|---|---:|---:|---:|---:|---|---|
| m-EuroSAT | GEO-Bench | Single-label classification | 10 | 2,000 | 1,000 | 1,000 | Sentinel-2 land cover | Macro-F1 |
| m-BigEarthNet | GEO-Bench | Multi-label classification | 43 | 20,000 | 1,000 | 1,000 | Sentinel-2 multi-label land cover | F1@opt |
| m-SA-Crop-Type | GEO-Bench | Semantic segmentation | 10 | 3,000 | 1,000 | 1,000 | Sentinel-2 crop type | mIoU |
| MMEarth20K | MMEarth subset | Semantic segmentation | 9 | 16,000 | 2,000 | 2,000 | Sentinel-2 with Dynamic World maps | mIoU |

### MMEarth20K construction

MMEarth20K is a 20,000-sample subset with Dynamic World-derived 10 m land-cover label maps. Its fixed 80/10/10 split supports dense biome-aware evaluation while retaining all six macro-groups. The sample composition is intentionally imbalanced across groups so that evaluation can measure ecological robustness under unequal coverage.

## Biome Metadata

Every sample is represented by

```text
(x_i, y_i, d_i, location_i, b_i, g_i)
```

where:

| Field | Meaning |
|---|---|
| `x_i` | Input image or multispectral patch |
| `y_i` | Classification label, multi-label vector, or segmentation mask |
| `d_i` | Dataset identifier |
| `location_i` | Patch centroid `(latitude, longitude)` |
| `b_i` | Raw terrestrial biome label in `{1, ..., 14}` |
| `g_i` | Derived biome macro-group label in `{1, ..., 6}` |

The macro-group is obtained through a deterministic mapping `g_i = psi(b_i)`. Raw labels remain available for fine-grained diagnostics; macro-groups are the primary units for robustness evaluation and debiasing.

Unknown, water-only, and unmatched samples are reported separately. They are excluded from primary worst-group summaries unless explicitly stated. When a dataset covers only part of the taxonomy, metrics are computed over the active macro-groups present in that split.

## Frozen Foundation Models

| Configuration | Prithvi-EO-2.0 | SatMAE | DOFA |
|---|---|---|---|
| Backbone | Spatio-temporal ViT | ViT-Large | ViT-Base |
| Parameters | 300M | 304M | 86M |
| Embedding dimension | 1024 | 1024 | 768 |
| Patch size | 16 x 16 | 16 x 16 | 16 x 16 |
| Input bands | 6 HLS bands | 10 Sentinel-2 bands | 9-12 wavelength bands |
| Spectral range | Official HLS configuration | Official Sentinel-2 configuration | 0.44-2.20 micrometers |
| Encoder status | Frozen, zero updates | Frozen, zero updates | Frozen, zero updates |
| Classification head | Linear / MLP probe | Linear probe | BatchNorm1d + Linear |
| Segmentation head | FCN decoder | Convolutional decoder | Multi-level UPerNet |

Official pretraining configurations, checkpoints, band orderings, and normalization statistics are used for every backbone.

## Controlled Transfer Protocol

FairRSFM enforces a strictly frozen encoder across classification and segmentation. Backpropagation updates only:

- a task-specific linear or MLP head for classification; or
- a lightweight FCN, convolutional, or UPerNet-style decoder for segmentation.

This design isolates the properties of the pretrained representation. Any change in robustness under ERM, DBR, BOLP, or GroupDRO is attributable to the mitigation method and downstream head rather than backbone fine-tuning.

## Training Configuration

| Setting | Value |
|---|---|
| Input size | 224 x 224 |
| Optimizer | AdamW |
| Adam betas | 0.9, 0.999 |
| Base learning rate | `1e-3` |
| Final cosine learning rate | `1e-6` |
| Weight decay | `1e-4` |
| Schedule | Cosine annealing |
| Epochs | 50 |
| Early-stopping patience | 15 |
| Classification batch size | 64 |
| Segmentation batch size | 32 |
| Random seeds | 3 independent runs |
| Checkpoint selection | Best validation overall metric |

Test reporting includes the overall task score, worst-group performance, expected calibration error, normalized failure range, equalized-odds disparity, and demographic-parity disparity.

## Task-Specific Scoring

### m-EuroSAT

Single-label classification is evaluated with macro-F1 over the classes present in each active macro-group.

### m-BigEarthNet

Multi-label classification uses macro-F1 at class-specific optimal thresholds (`F1@opt`). Thresholds are selected on validation data over `[0.1, 0.95]` and then fixed for test evaluation.

### m-SA-Crop-Type and MMEarth20K

Segmentation is evaluated with global and group-wise mean intersection-over-union. Each patch is assigned to a biome macro-group from its centroid, and mIoU is averaged over classes present in that group.

## Reproducibility Principles

- Fixed dataset definitions and splits.
- Official pretrained checkpoints and input statistics.
- Frozen encoders for every task and method.
- Identical optimization budgets across mitigation methods.
- Validation-selected checkpoints.
- Three independent seeds with mean and standard deviation reporting.
- Explicit handling of unmatched samples and inactive biome groups.

## Related Documentation

- [Biome taxonomy and labeling](biome-groups.md)
- [Methods and evaluation metrics](methods.md)
- [Complete experimental results](results.md)
