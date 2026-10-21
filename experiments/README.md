# Experiments

This directory provides the execution runners and reproducible configurations for all 48 FairRSFM benchmark configurations (3 Foundation Models × 4 Datasets × 4 Mitigation Methods).

```text
experiments/
|-- train.py          # Unified CLI training launcher
|-- configs/
|   |-- backbones/    # Prithvi-EO-2.0, SatMAE, and DOFA specs
|   |-- datasets/     # EuroSAT, BigEarthNet, SA-Crop, MMEarth specs
|   |-- methods/      # ERM, DBR, BOLP, and GroupDRO hyperparameters
|   `-- data_mappings/# Biome lookup JSON mappings
|-- scripts/          # Canonical experiment scripts
`-- outputs/          # Local logs, checkpoints, and predictions (not tracked in git)
```

## Running Training

You can launch any experiment via the unified training dispatcher:

```bash
# General syntax:
python experiments/train.py --model <prithvi|satmae|dofa> \
                            --dataset <eurosat|bigearthnet|sa_crop|mmearth> \
                            --method <erm|dbr|bolp|groupdro> \
                            --seed <seed>

# Examples:
# 1. Train SatMAE on BigEarthNet with GroupDRO
python experiments/train.py --model satmae --dataset bigearthnet --method groupdro --seed 42

# 2. Train Prithvi on EuroSAT with DBR
python experiments/train.py --model prithvi --dataset eurosat --method dbr --seed 42

# 3. Train DOFA on South Africa Crop Type with BOLP
python experiments/train.py --model dofa --dataset sa_crop --method bolp --seed 42
```

### Standard CLI Arguments

* `--model`: Foundation model backbone (`prithvi`, `satmae`, `dofa`).
* `--dataset`: Benchmark task (`eurosat`, `bigearthnet`, `sa_crop`, `mmearth`).
* `--method`: Debiasing algorithm (`erm`, `dbr`, `bolp`, `groupdro`).
* `--seed`: Random seed (default: `42`).
* `--batch_size`: Batch size per GPU.
* `--epochs`: Maximum training epochs.
* `--lr`: Learning rate for task head probe.
* `--gpu`: GPU device ID (e.g. `0`).
* `--output_dir`: Output directory for checkpoints and logs (defaults to `experiments/outputs/`).

## Output Policy

The `outputs/` directory is intentionally excluded from version control. Checkpoints and evaluation predictions will be written to `outputs/` during runs without cluttering git tracking.
