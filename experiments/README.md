# Experiments

This directory contains the training launcher, experiment scripts, and
configuration drafts for the benchmark combinations. Training scripts currently
hold several settings directly; the YAML files are reference material until
configuration loading is consolidated.

```text
experiments/
|-- train.py          # Unified CLI training launcher
|-- configs/
|   |-- backbones/    # Prithvi-EO-2.0, SatMAE, and DOFA specs
|   |-- datasets/     # EuroSAT, BigEarthNet, SA-Crop, MMEarth specs
|   |-- methods/      # ERM, DBR, BOLP, and GroupDRO hyperparameters
|   `-- data_mappings/# Biome lookup JSON mappings
|-- scripts/          # Canonical experiment scripts
`-- run_prithvi_matrix.py  # Isolated Prithvi matrix launcher
```

## Running Training

### Prithvi ERM and BOLP audit matrix

From the repository root, launch one full 50-epoch seed across all four datasets.
The two entries in `--gpus` create two concurrent workers. Repeating `0` runs
both on the same GPU, so check available GPU memory before starting.

```bash
python experiments/run_prithvi_matrix.py --methods erm bolp --seeds 42 \
    --epochs 50 --gpus 0 0 --run-id prithvi_erm_bolp_audit_20261022
```

The launcher writes a manifest, command, status, checkpoints, metrics, and logs
under `runs/prithvi_matrix/<run-id>/`. It refuses an existing run ID. Use
`--dry-run` to inspect the queue. After checking the first seed and checkpoint
evaluations, use `--seeds 42 43 44` with a new run ID for the three-seed
comparison. The two workers share GPU memory when both IDs are `0`.

The metric definitions are in [`../docs/metric-definitions.md`](../docs/metric-definitions.md).
Checkpoint evaluation still requires a separate audit before results are used
in a paper; see [`../docs/cleanup-status.md`](../docs/cleanup-status.md).

### Single run

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
* `--output_dir`: Output directory when supported by the selected training script; the Prithvi matrix passes an isolated directory for every job.

## Output Policy

Training outputs go under the ignored `runs/` directory. The Prithvi matrix
launcher creates one folder per dataset, method, and seed.
