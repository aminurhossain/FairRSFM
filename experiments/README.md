# Experiments

This directory is reserved for reproducible FairRSFM experiment definitions.
Each released experiment will specify the dataset, frozen backbone, mitigation
method, task head, random seed, and training configuration needed to reproduce
the corresponding paper result.

```text
experiments/
|-- configs/
|   |-- backbones/  # Prithvi-EO-2.0, SatMAE, and DOFA
|   |-- datasets/   # Dataset and split definitions
|   |-- methods/    # ERM, DBR, BOLP, and GroupDRO
|   `-- tasks/      # Classification and segmentation runs
`-- outputs/        # Local logs, checkpoints, and predictions (not tracked)
```

Experiment configurations and launch commands are in preparation. Reported
hyperparameters and results are already available in the [benchmark
protocol](../docs/benchmark.md) and [complete result tables](../docs/results.md).

## Output Policy

The `outputs/` directory is intentionally excluded from version control except
for its ignore file. Published artifacts and benchmark metadata will be linked
from the main repository README rather than committed as generated files.
