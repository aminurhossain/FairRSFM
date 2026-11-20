# Training scripts

The canonical matrix contains 48 configurations per seed: three backbones,
four datasets, and four methods. The eight DOFA UPerNet segmentation variants
are additional configurations.

All scripts use the shared objectives in
[`protocol.py`](../../src/fairrsfm/training/protocol.py) and algorithms in
[`methods/`](../../src/fairrsfm/methods/). The mathematical contract is in
[`docs/methods.md`](../../docs/methods.md).

| Method | Shared rule |
| --- | --- |
| ERM | Mean task loss over retained training images/patches. |
| DBR | Epoch group training-loss EMA, beta 0.9; weights proportional to `(EMA loss + 1e-6)^0.5`, normalized to mean 1; inverse-frequency initialization. |
| GroupDRO | Official update-before-objective exponential group probabilities, eta 0.01; inverse group-frequency sampling with replacement. |
| BOLP | Training-only group centroids, equal group global mean, residual centering, rank `min(number of groups - 1, dimension)`; transform only the final encoder layer. |

All methods train on the same known-biome subset. Validation and test retain
unknown-biome samples for overall scoring. Segmentation uses valid-pixel mean
cross-entropy per patch, EuroSAT uses per-image cross-entropy, and BigEarthNet
uses per-image BCE averaged over labels. Backbones stay frozen.

Per-epoch validation and test reporting remain enabled. Validation selects
checkpoints and BigEarthNet thresholds; test metrics do not update methods.
Checkpoint state includes the method configuration, group statistics, and BOLP
projectors. Historical checkpoints cannot resume as the new shared protocol.

## Directory structure

```text
experiments/scripts/
├── prithvi/
│   ├── classification/{eurosat,bigearthnet}/
│   └── segmentation/{sa_crop,mmearth_20k}/
├── satmae/{eurosat,bigearthnet,sa_crop,mmearth}/
└── dofa/{eurosat,bigearthnet,sa_crop,mmearth,sa_crop_upernet,mmearth_upernet}/
```

## Commands from the repository root

Use the unified entry point to translate each adapter's CLI flags:

```bash
python experiments/train.py --model prithvi --dataset eurosat --method erm \
  --seed 42 --epochs 50 --batch-size 64 --gpu 0 --output_dir runs/example_prithvi_erm

python experiments/train.py --model satmae --dataset bigearthnet --method dbr \
  --seed 42 --epochs 50 --batch-size 64 --gpu 0 --output_dir runs/example_satmae_dbr

python experiments/train.py --model dofa --dataset sa_crop --head upernet --method bolp \
  --seed 42 --epochs 50 --batch-size 16 --gpu 0 --output_dir runs/example_dofa_upernet_bolp
```

Add `--smoke-test` for bounded real-data checks; these produce no benchmark
results. Adapter-specific algorithm flags differ: consult the selected
script's `--help` before overriding defaults. Heads, optimizer settings, batch
sizes, and stopping rules retain their adapter-specific defaults.

For isolated multi-run output directories and source manifests, use
[`run_matrix.py`](../run_matrix.py). See
[`cleanup-status.md`](../../docs/cleanup-status.md) for verification coverage.
