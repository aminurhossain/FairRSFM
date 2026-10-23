# Repository cleanup status

This records the implementation boundary before running new IJRS experiments.

## Completed in the current cleanup pass

- Added a mathematical metric contract in `docs/metric-definitions.md` and a
  corresponding copy in the original `FairRSFM/evaluation` folder.
- BigEarthNet training uses validation-selected thresholds, the same
  `0.10, 0.15, ..., 0.90` grid, and per-label binary ECE across all 12
  model-method scripts. Threshold vectors are saved in validation records.
- BigEarthNet checkpoint evaluation requires validation thresholds and checks
  their epoch against Lightning checkpoint metadata when present.
- All 12 checkpoint evaluators import and show CLI help. Shared scoring keys,
  MMEarth band/ignore settings, SA-Crop ignore settings, and the BigEarthNet
  output schema were repaired. Checkpoint loading rejects missing task weights;
  evaluator JSON stores undefined metrics as `null`.
- `.gitignore` still excludes local runs, checkpoints, logs, and scratch data.
- Added `experiments/run_prithvi_matrix.py` for isolated Prithvi matrices with
  two bounded workers, run manifests, per-job status, and non-overwriting output
  directories. The ERM and BOLP training scripts now accept those output paths.
  MMEarth training saves validation metrics each epoch and evaluates test
  metrics only after selecting a checkpoint from validation.
- New Prithvi classification BOLP checkpoints store biome projection metadata.
  Their evaluators now apply the same token-shaped encoder transform used by
  training, with biome IDs from the test samples. A synthetic hook test checks
  shape and group residualization.
- Consolidated the duplicate root training entry point and top-level MMEarth
  data module into compatibility shims. Removed empty Git placeholders and the
  unused SA-Crop BOLP callbacks that selected checkpoints from test metrics.
  Existing local run artifacts remain ignored and untouched.

## Audit before the next Prithvi matrix

- The 2026-10-22 ERM+BOLP tmux run was stopped and only its
  `runs/prithvi_matrix/prithvi_erm_bolp_audit_20261022` directory was removed.
- Prithvi SA-Crop ERM and DBR retain their per-epoch test evaluation, as
  requested. They also save the validation-selected checkpoint's test result
  separately in `test_selected.json`.
- Prithvi MMEarth ERM, BOLP, and DBR checkpoint selection now logs the exact
  present-ground-truth-class global mIoU written to the validation JSON.
  The benchmark monitor uses `val/benchmark_miou` so it does not collide with
  TerraTorch's inherited `val/mIoU` logging.
- Prithvi MMEarth validation and final selected-test evaluation now pass tile
  IDs and biome metadata to BOLP, matching the training forward pass. The
  previous calls applied projection without biome residualization.
- The DOFA and SatMAE EuroSAT/BigEarthNet checkpoint evaluators now calculate
  global macro-F1 with the same present-class rule as group macro-F1.
- Static code inspection found all four SatMAE MMEarth training scripts also
  evaluate test data on every validation epoch. This behavior is retained by
  request. They do not yet save a final test result for the validation-selected
  checkpoint.

## Work before paper-grade reruns

### 2026-10-23 matrix failure review

All launcher logs and nested training logs in `prithvi_erm_bolp_20261023`
were inspected. BigEarthNet ERM and SA-Crop BOLP exited successfully.
SA-Crop ERM completed training and wrote `test_selected.json`, then failed
on the supplementary worst-group checkpoint test. Its fitted checkpoints
remain available; retraining is unnecessary to recover that evaluation.

Repairs:

- EuroSAT: convert shared-class NumPy IDs to Python integers before JSON
  serialization. The diagnostics regression test now serializes the payload.
- MMEarth: benchmark checkpoint monitor is `val/benchmark_miou`; inherited
  TerraTorch logging retains `val/mIoU`. Three CPU Lightning integration tests
  verify the real benchmark callbacks save the checkpoint with the reported
  validation score without a logging-name collision.
- BigEarthNet ERM/BOLP: batch size is configurable with `--batch-size` and
  defaults to 64 rather than 256. BOLP feature fitting is also capped by that
  smaller default. Available shared-GPU memory still determines feasibility.
- SA-Crop ERM/BOLP/DBR: snapshot both selected checkpoint paths before testing
  because Lightning restores historical callback state during a test call.
  Export primary checkpoint weights before evaluating the supplementary one.

Five jobs failed before completing training: EuroSAT ERM/BOLP, BigEarthNet
BOLP, and MMEarth ERM/BOLP. Existing logs and outputs were preserved.

### GPU smoke checks after the repairs

The `prithvi_smoke_20261023_v1` matrix completed all eight GPU jobs successfully
with two workers sharing GPU 0. Checks use real datasets and pretrained weights,
one epoch, two Lightning train/validation/test batches, and up to eight batches
for projection fitting and direct callback evaluations. They exercise training,
metric JSON writes and checkpoint selection. Segmentation additionally reloads
the selected checkpoint and writes final test outputs. Smoke scores are not
benchmark results and must not be used in paper tables.

EuroSAT BOLP's saved checkpoint was also restored by the standalone evaluator
on the full test split; support diagnostics and 1,000-replicate confidence
intervals were verified. MMEarth BOLP's override needed additional updates to
the inherited train/validation/test metric collections; its repeat GPU check
passed and is stored under `prithvi_smoke_20261023_mmearth_bolp_v2`. The repeat
produced finite native test mIoU without the uninitialized-metric warning.
The complete regression suite also passed all 16 tests.

To repeat the bounded GPU matrix in a fresh directory, use
`python experiments/run_prithvi_matrix.py --methods erm bolp --gpus 0 0 --run-id UNIQUE_SMOKE_ID --smoke-test`.
Omit `--smoke-test` for full training. Full training still needs to finish
before the resulting experiments can be considered validated end to end.

1. Reconstruct and verify remaining BOLP checkpoint evaluators against their
   training architectures. Prithvi classification has a matching hook but still
   needs a real checkpoint evaluation; Prithvi segmentation and the SatMAE/DOFA
   BOLP evaluator paths remain to be checked. The load check makes mismatches
   fail explicitly.
2. Run one real checkpoint evaluation for every backbone, dataset, and method;
   the CLI checks and synthetic BigEarthNet scoring test do not validate
   backbone restoration or the full data pipeline. The repository contains
   local slim checkpoints but no pretrained backbone weights under
   `checkpoints/` in this workspace. The Prithvi 300M backbone weights are
   available in the older local `FairRSFM_unified/checkpoints` directory.
3. Consolidate training model/method code and YAML config handling. The
   launcher still dispatches to separate scripts, and DBR implementations have
   method differences that need an explicit protocol decision.
4. Rerun the comparison with the fixed metric contract and three seeds, then
   replace the historical result tables only after checking checkpoint,
   threshold, split, and seed provenance.
