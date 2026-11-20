# Repository experiment status

This note describes the benchmark code after the cleanup and reporting pass.
The metric definitions and class-presence rules are specified in
[`metric-definitions.md`](metric-definitions.md).

## Canonical benchmark matrix

The repository has 48 canonical runs per seed: Prithvi, SatMAE, and DOFA on
EuroSAT, BigEarthNet, SA-Crop, and MMEarth, using ERM, DBR, BOLP, and GroupDRO.
The eight DOFA UPerNet scripts remain available as additional segmentation
variants. All 56 scripts now use the shared training objectives and reporting
contract. The method choices are specified in [`methods.md`](methods.md):

- Common per-image task losses; valid-pixel mean loss per segmentation patch.
- The same known-biome training subset for all four methods; full validation
  and test splits remain available for overall scoring.
- DBR uses epoch group training-loss EMA with beta 0.9 and alpha 0.5.
- GroupDRO uses the official update-before-objective rule and group-balanced
  sampling with replacement.
- BOLP fits training-only centroids and transforms only the final encoder
  layer, with residual centering and default rank `min(number of groups - 1, d)`.

Method state and training population hashes are checkpointed. Resume rejects
changed method settings, changed training group assignments, and missing head
or method state, including in adapters that allow omitted frozen weights.

Every training callback evaluates validation and test data at the end of each
epoch. Validation metrics select checkpoints and, for BigEarthNet, fit the
epoch's class thresholds. The test pass uses those same thresholds and writes
its own epoch record. Final selected-checkpoint test reports are also saved;
group confidence intervals resample images for classification and patches for
segmentation, never individual pixels.

The shared reporting implementation is in
[`src/fairrsfm/training/reporting.py`](../src/fairrsfm/training/reporting.py).
It writes `val_results.json`, `test_results.json`, `training_config.json`, and
selected-checkpoint reports with JSON-safe values. Per-group diagnostics
include true and predicted class counts, GT-present macro-F1, union macro-F1,
class-matched macro-F1, and support flags. The global and group macro-F1
contract averages classes present in that group's ground truth; absent classes
are excluded. Unknown biome samples contribute to overall results and are
excluded from group fairness summaries.

The primary checkpoint monitors are `val/benchmark_macro_f1` for EuroSAT and
BigEarthNet and `val/benchmark_miou` for SA-Crop and MMEarth. MMEarth keeps
TerraTorch's native `val/mIoU` log separate from the benchmark monitor.

## Launching the matrix

`experiments/run_matrix.py` is the canonical entry point. For one seed and all
48 paths, provide one GPU entry per simultaneous worker, for example:

```bash
python experiments/run_matrix.py \
  --models prithvi satmae dofa \
  --datasets eurosat bigearthnet sa_crop mmearth \
  --methods erm dbr bolp groupdro \
  --seeds 42 \
  --epochs 50 \
  --gpus 0 0 \
  --run-id ijrs_full_seed42
```

Repeated GPU IDs allow two workers to share one GPU. Use a unique run ID; the
launcher refuses to overwrite an existing output directory and records the
source hashes and job status in the run manifest. Add `--dry-run` to inspect
the 48 resolved jobs without starting training. Add `--smoke-test` for bounded
real-model checks; smoke results are not benchmark results.

## Verification completed

- The callback suite exercises the actual callback implementations across all
  53 standard training scripts and checks that test reporting runs after
  validation, saves JSON, reports per-group class support, and restores model
  train/eval modes. The three Prithvi MMEarth callbacks have Lightning lifecycle
  tests for the checkpoint monitor and both split records.
- The regression suite passes 107 tests, covering metric definitions, all 16
  task/method objective combinations, final-layer BOLP behavior, known-biome
  subsets, GroupDRO sampling, DBR epoch boundaries/resume, missing-state
  rejection, unknown-biome overall evaluation, MMEarth checkpoint monitors,
  both DOFA UPerNet BOLP adapters, the generic probe's final-layer hook, and
  safeguards in all twelve legacy standalone evaluators.
- Seven bounded real-data GPU smoke jobs completed successfully during this
  refactor. Each includes training, per-epoch validation/test reporting, and
  selected-checkpoint evaluation:

  | Run ID | Checks |
  | --- | --- |
  | `protocol_smoke_20261023_v1` | Prithvi EuroSAT: ERM, DBR, GroupDRO, BOLP |
  | `protocol_adapter_smoke_20261023_v1` | SatMAE BigEarthNet: BOLP |
  | `protocol_segmentation_smoke_20261023_v1` | Prithvi MMEarth: BOLP |
  | `protocol_dofa_smoke_20261023_v1` | DOFA SA-Crop: BOLP |

  These cover all three backbones and four tasks, but do not verify every
  combination. Smoke fitting/evaluation is batch-limited and its outputs must
  not be used as paper results. Run manifests and per-job configurations record
  the source versions used while the refactor was in progress.
- A prior Prithvi ERM+BOLP smoke matrix completed eight real-data GPU jobs.
  It predates this cross-backbone reporting pass and does not verify all 48
  model/method/data combinations.
- The earlier Prithvi ERM+BOLP matrix is partial: EuroSAT ERM and BOLP have
  successful status files; BigEarthNet ERM and BOLP have checkpoints through
  roughly epochs 29 and 31 but no completion status; SA-Crop and MMEarth have
  no job outputs in that run. These checkpoints predate the shared reporting
  changes and must not be mixed with future standardized DBR results.

## Remaining work before paper tables

1. Full 48-run execution remains deferred. Before paper training, complete the
   remaining adapter smoke coverage, including the optional UPerNet variants.
   CPU callback tests do not establish that every model restores its weights
   or runs its complete data pipeline.
2. The twelve legacy standalone evaluator scripts now reject unified
   checkpoints. This prevents their older architecture reconstruction from
   silently changing a final-layer BOLP model or metric path. For every new
   run, use the automatically written `test_selected.json` and
   `test_selected_worst.json`; those reports use the exact training adapter,
   preprocessing, BOLP state, threshold selection, and shared metric contract.
   The standalone scripts remain for historical checkpoints only.
3. Use old Prithvi checkpoints only for diagnostics;
   rerun under the unified protocol before using results in paper tables.
4. Run the paper comparison with the agreed seeds and update result tables only
   after checking checkpoint, threshold, split, and seed provenance.
