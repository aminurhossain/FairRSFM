# FairRSFM metric definitions

This document records the metrics implemented in the original DOFA experiment
scripts and specifies the consistent protocol for new
FairRSFM experiments. Scores in code are fractions in `[0, 1]`; tables multiply
them by 100. Compute each metric for each seed on the selected checkpoint, then
report the mean and sample standard deviation across seeds (`ddof=1`).
Undefined checkpoint-evaluation metrics are stored as JSON `null`.

## Evaluation units and classes

EuroSAT and BigEarthNet use images as observations. Segmentation uses valid
pixels to compute confusion counts, while biome assignment remains at patch
level. Let `C_g` be classes with at least one positive ground-truth observation
in group `g`. The overall score pools observations first and uses the classes
present in that pooled set. Thus overall macro-F1 or mIoU need not lie between
the group scores, since their class sets can differ.

An absent class is **excluded from the group macro average**, rather than
assigned F1 = 0. For EuroSAT, predicting an absent class still creates a
false negative for the true class. For BigEarthNet, a false positive on a label
with no positive ground truth in that group does not enter that group's
macro-F1; report class support alongside group scores when comparing groups.
If a multi-label group has no positive labels at all, macro-F1 is undefined.

For EuroSAT, also report **union macro-F1** over classes appearing in either
ground truth or predictions. A predicted-only class enters that average with
F1 = 0. This is a diagnostic; the ground-truth-present score remains primary.
Report each group's true and predicted count per class, so a zero F1 can be
identified as a missed present class or a predicted-only class. A supplementary
class-matched macro-F1 uses the intersection of ground-truth-present class
sets across groups; it is undefined if that intersection is empty. Flag groups
with fewer than 30 images as low support, rather than silently merging or
dismissing them. This threshold is descriptive, not an exclusion rule.

For each class `c`, define `TP_c`, `FP_c`, `FN_c`, and `TN_c` from predictions
and ground truth. Segmentation excludes pixels with the task's `ignore_index`
*before* computing any of these counts. The ignore index must be recorded with
each dataset; SA-Crop and MMEarth do not necessarily share one.

## Task performance

| Metric | Definition | Use |
| --- | --- | --- |
| Class precision | `TP_c / (TP_c + FP_c)` | Classification and segmentation |
| Class recall / TPR | `TP_c / (TP_c + FN_c)` | Classification and segmentation |
| Class F1 | `2 TP_c / (2 TP_c + FP_c + FN_c)` | Classification and segmentation |
| Macro-F1 | `mean_{c in C} F1_c` | EuroSAT overall and per group; BigEarthNet overall and per group |
| Micro-F1 | `2 sum_c TP_c / (2 sum_c TP_c + sum_c FP_c + sum_c FN_c)` | Secondary classification metric |
| Exact accuracy | `mean_i 1[all predicted labels_i = all true labels_i]` | BigEarthNet |
| Accuracy | `mean_i 1[predicted class_i = true class_i]` | EuroSAT |
| Class IoU | `TP_c / (TP_c + FP_c + FN_c)` | Segmentation |
| mIoU | `mean_{c in C} IoU_c` | SA-Crop and MMEarth, pooled over valid pixels |

For BigEarthNet, predictions are `1[p_{i,c} >= tau_c]`. Select each `tau_c`
using *validation* labels and sigmoid probabilities by maximizing class F1 on
the candidate grid `0.10, 0.15, ..., 0.90` (17 values). Save the threshold
vector with the selected run and apply it unchanged to test predictions.
Historical SatMAE results used another grid and need rerunning for comparison.

The standalone BigEarthNet evaluators require `--thresholds-json` pointing to
the run's `val_results.json`. If that file has multiple epochs, pass
`--thresholds-epoch` only if the checkpoint lacks epoch metadata. The evaluator
checks that the selected validation record matches the checkpoint epoch. Keep
the checkpoint and threshold history from the same seed and method.

Average precision for class `c` is the area under its precision-recall step
curve when observations are ranked by predicted probability. `mAP` is the mean
of these values over classes with at least one positive ground-truth example.
It uses probabilities, not thresholded predictions. The original DOFA scripts
also report this for EuroSAT using one-vs-rest class indicators.

## Group robustness

Let `M_g` be the primary task score for active group `g`, and let `G` contain
groups with test observations. Use the same group set for each metric within a
run.

| Metric | Definition |
| --- | --- |
| Worst group | `min_{g in G} M_g` |
| Best group | `max_{g in G} M_g` |
| Group mean | `(1 / |G|) sum_g M_g` (unweighted by group size) |
| Group range | `max_g M_g - min_g M_g` |
| PER | `min_g M_g / max_g M_g`, when the denominator is positive |
| NFR | `(max_g M_g - min_g M_g) / group_mean`, when the denominator is positive |
| Gini | `sum_{i=1}^{K} (2i-K-1) M_(i) / (K sum_i M_(i))`, with group scores sorted ascending |

`NFR` is normalized by the group mean in the DOFA scripts. It must not silently
switch to the overall score as denominator. The within-run standard deviation
of group scores is `sqrt(sum_g (M_g - group_mean)^2 / (K-1))`; it is distinct
from standard deviation across random seeds.

## Calibration

Use ten equal-width probability bins `B_b` partitioning `[0,1]`, including
both endpoints exactly once. For observations with confidence `q_i` and binary
outcome `a_i`,

`ECE = sum_b (|B_b| / N) |mean_{i in B_b} a_i - mean_{i in B_b} q_i|`.

For EuroSAT, `q_i` is the maximum softmax probability and `a_i` indicates
whether its predicted class is correct. For segmentation, each valid pixel is
one observation; `q_i` is its maximum softmax probability and `a_i` indicates
correct pixel classification. For BigEarthNet, compute binary ECE separately
for each label with `q_{i,c}` equal to its sigmoid probability and `a_{i,c}`
equal to its ground-truth 0/1 label. Macro-average ECE over labels with at
least one positive observation in the scored group.

The original DOFA BigEarthNet scripts reduce the multi-label vector with
`max(axis=1)` before ECE. That quantity is **legacy only**: a maximum over
binary labels is not a valid per-label calibration target. New results must
use the per-label definition above and label any legacy ECE values separately.

ECE disparity is `max_g ECE_g - min_g ECE_g` over groups where ECE is defined.

## Parity metrics

For each class `c` and group `g`, compute one-vs-rest
`TPR_{g,c} = TP_{g,c}/(TP_{g,c}+FN_{g,c})` and
`FPR_{g,c} = FP_{g,c}/(FP_{g,c}+TN_{g,c})`.
The DOFA implementation compares Youden indices
`J_{g,c} = TPR_{g,c} - FPR_{g,c}`:

`EOdd = max_{g != h} mean_{c in C_g intersect C_h} |J_{g,c} - J_{h,c}|`.

Undefined rates and classes absent from either group's ground truth are
excluded from that pair. This is the implemented *Youden-index disparity*,
which should be named explicitly because it differs from the standard
equalized-odds metric that compares TPR and FPR separately.

Let `PPR_{g,c}` be the fraction of observations predicted as class `c` in
group `g`. For BigEarthNet it is the fraction predicted positive for label
`c`; for segmentation it is the fraction of valid pixels predicted as `c`.

`DPM = max_{g != h} mean_{c=1..C} |PPR_{g,c} - PPR_{h,c}|`.

Both parity metrics are computed from fixed predictions after validation
threshold selection. They describe differences across groups; they do not
establish that biome membership caused those differences.

## Uncertainty

For seed scores `s_1, ..., s_R`, report their arithmetic mean and sample
standard deviation `sqrt(sum_r (s_r - mean)^2 / (R-1))`. For a test-sampling
confidence interval, resample **patches/images within each group** with
replacement and recompute the pooled task metric for each bootstrap sample.
For segmentation, resample whole patches rather than individual pixels so
spatially correlated pixels stay together. Report the percentile interval and
the number of bootstrap replicates.
