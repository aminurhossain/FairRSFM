# Methods and Evaluation Metrics

[Back to the main README](../README.md)

## Problem Setup

Let the training set be

$$
\mathcal{D}_{\mathrm{tr}} = \{(x_i, y_i, g_i)\}_{i=1}^{N},
$$

where `x_i` is an input image, `y_i` is its task label, and `g_i` is the biome macro-group. A frozen RSFM encoder `phi` produces

$$
z_i = \phi(x_i) \in \mathbb{R}^{d}.
$$

Every method keeps `phi` fixed and trains only the downstream head `h_theta`.
The common per-image loss is cross-entropy for EuroSAT and binary
cross-entropy averaged over labels for BigEarthNet. For segmentation, average
cross-entropy over valid pixels within each patch, then average patches.
Patches with no valid pixels contribute neither a task-loss observation nor
a group-loss observation. These task-loss definitions apply to all four
methods and every backbone.

All four methods train on the same subset of training images with known biome
labels. Validation and test retain all images: unknown-biome images count toward
overall metrics but not group fairness summaries. The retained training IDs and
group assignments are hashed in each run's configuration and checkpoint.

## Compared Strategies

| Method | Stage | Data retained | Backbone updates | Additional trainable parameters |
|---|---|---:|---:|---:|
| ERM | Standard task objective | Known-biome training subset | No | No |
| DBR | Loss reweighting | Known-biome training subset | No | No |
| BOLP | Representation transformation | Known-biome training subset | No | No |
| GroupDRO | Group-robust objective | Known-biome training subset | No | No |

## Empirical Risk Minimization

ERM is the frozen-feature baseline:

$$
\mathcal{L}_{\mathrm{ERM}} = \frac{1}{N}\sum_{i=1}^{N}\ell(h_{\theta}(z_i), y_i).
$$

After the common training subset is selected, ERM does not use biome labels in
its objective or sampling.

## Dynamic Biome Reweighting

Dynamic Biome Reweighting (DBR) is a loss-level mitigation method. It begins
with inverse-frequency group weights, then raises weights for groups with
higher **training loss**. A per-group EMA smooths the loss signal. Validation
and test metrics do not update the weights.

Let `p_g = n_g / N` be the empirical frequency of group `g`. Initial normalized weights are

$$
w_g^{(0)} = \frac{1/p_g}{\frac{1}{|\mathcal{G}|}\sum_{g'\in\mathcal{G}}1/p_{g'}}.
$$

At the end of epoch `t`, let `L_g^(t)` be the mean per-example task loss for
group `g`. For segmentation, compute loss per patch over valid pixels, then
average patches within each group. Smooth the group loss using

$$
\bar{L}_g^{(t)} = \beta \bar{L}_g^{(t-1)} + (1-\beta)L_g^{(t)}.
$$

Set the next epoch's unnormalized group weight proportional to smoothed loss:

$$
\tilde{w}_g^{(t+1)} = (\bar{L}_g^{(t)} + \epsilon)^\alpha,
\qquad
w_g^{(t+1)} =
\frac{\tilde{w}_g^{(t+1)}}
{\frac{1}{|\mathcal{G}|}\sum_{g'}\tilde{w}_{g'}^{(t+1)}}.
$$

Groups absent from an epoch retain their previous EMA loss. Unknown-biome
samples are excluded from the common training subset. Initialize each group's EMA from its first observed
training epoch. The cleaned benchmark uses `alpha=0.5`, `beta=0.9`, and updates
weights once per epoch with `epsilon=1e-6` and no clipping. Checkpoints persist
EMA losses, weights, partial-epoch sums/counts, and the last updated epoch.
If validation checkpointing precedes the epoch-end update, resume completes
that update before the next epoch's training batches.

The training objective is

$$
\mathcal{L}_{\mathrm{DBR}} =
\frac{1}{N}\sum_{i=1}^{N}w_{g_i}^{(t)}\ell(h_{\theta}(\phi(x_i)), y_i).
$$

DBR is non-adversarial, parameter-free beyond the task head, and applicable to classification and segmentation.

The source audit records the historical differences and the shared replacement
in [`dbr-protocol-audit.md`](dbr-protocol-audit.md).

## Biome-Orthogonal Linear Probing

Biome-Orthogonal Linear Probing (BOLP) acts on the geometry of frozen embeddings.

### 1. Group centroids

For each macro-group,

$$
\mu_g = \frac{1}{n_g}\sum_{i:g_i=g}z_i,
\qquad
\mu_0 = \frac{1}{|\mathcal{G}|}\sum_g \mu_g.
$$

### 2. Inter-group mean-difference matrix

$$
B = [\mu_1-\mu_0,\ldots,\mu_{|\mathcal{G}|}-\mu_0]
\in \mathbb{R}^{d\times|\mathcal{G}|}.
$$

### 3. Biome-associated subspace

Using the thin singular value decomposition

$$
B = U\Sigma V^\top,
$$

the leading `k` left singular vectors `U_k` span the dominant inter-group directions. Since the columns of `B` sum to zero, at most `|G|-1` directions are non-trivial. The universal default is `k = min(|G|-1, d)`, with residual centering enabled. Any smaller rank is an explicitly labeled ablation.

### 4. Orthogonal projector

$$
P_k = I_d - U_k U_k^\top.
$$

### 5. Debiased embedding

For a sample with known group label `g_i`, BOLP removes the group-mean shift and projects out the biome-associated subspace:

$$
\hat{z}_i = P_k(z_i - \mu_{g_i} + \mu_0).
$$

The downstream head is then trained with

$$
\mathcal{L}_{\mathrm{BOLP}} =
\frac{1}{N}\sum_i \ell(h_{\theta}(\hat{z}_i), y_i).
$$

The centroids, global mean, and projector are computed from training embeddings
and frozen. Only the **final encoder layer** is transformed. For encoders that
return several feature layers, earlier layers pass to the decoder unchanged.
Fit centroids from spatial/token means per image, giving each image equal
weight within its group; apply the fitted centering and projector to every
token or spatial position of the final layer during training, validation, and
test. If a group label is unavailable during inference, group residualization
is skipped and the global projector is still applied. Centroids and projector
buffers, group order, rank, and residualization settings are checkpointed.

## GroupDRO

GroupDRO is included as a group-robust optimization baseline. It emphasizes groups with higher loss so that training targets robust performance rather than average empirical risk. In FairRSFM it uses the same biome macro-groups, frozen encoder, task head, data, optimization budget, and checkpoint-selection protocol as the other methods.

The algorithm follows the standard robust objective in Sagawa et al.'s
[official implementation](https://github.com/kohpangwei/group_DRO/blob/master/loss.py).
Initialize adversarial probabilities uniformly. At each training batch,
calculate group mean task losses (zero for absent groups), update
`q_g <- q_g * exp(eta * loss_g)` and normalize over all training groups, then
compute `sum_g q_g * loss_g` using the updated probabilities. Use `eta=0.01`,
zero generalization adjustment, no loss normalization, and the standard
robust objective rather than its optional BTL variant. The official example
commands use group-balanced sampling: each retained training image is sampled
with weight `1/n_g`, with replacement, for `N` draws per epoch. The same known-
biome training subset is used by all four methods. GroupDRO probabilities are
checkpointed and validation/test forwards never update them.

## Shared implementation

The 48 canonical scripts and eight optional DOFA UPerNet variants use
`src/fairrsfm/training/protocol.py` for training objectives and task losses, and
`src/fairrsfm/methods/` for the method algorithms. Backbone adapters retain
their heads and optimizer settings. Per-epoch validation **and test** reporting
remain enabled; only validation selects checkpoints and BigEarthNet thresholds.

## Evaluation Metrics

The full mathematical contract, including task-specific ECE, parity metrics,
class presence, sample standard deviation, and bootstrap units, is recorded in
[metric-definitions.md](metric-definitions.md). A copy is kept beside the
original DOFA code at `FairRSFM/evaluation/METRIC_DEFINITIONS.md`.

Let `G_d` be the active biome macro-groups in dataset `d`, with `K = |G_d|`, and let `M_g` be the task score for group `g`.

### Tier 1: Task performance

#### Single-label classification

For m-EuroSAT, group performance is macro-F1 over classes present in the group:

$$
\mathrm{Macro\text{-}F1}_g =
\frac{1}{|C_g|}\sum_{c\in C_g}\mathrm{F1}_{g,c}.
$$

#### Multi-label classification

For m-BigEarthNet, class thresholds `tau_c` are selected on validation data from `0.10, 0.15, ..., 0.90` and fixed for test evaluation:

$$
\mathrm{F1@opt}_g =
\frac{1}{|C_g|}\sum_{c\in C_g}\mathrm{F1}_{g,c}(\tau_c).
$$

#### Segmentation

For m-SA-Crop-Type and MMEarth20K,

$$
\mathrm{mIoU}_g =
\frac{1}{|C_g|}\sum_{c\in C_g}
\frac{\mathrm{TP}_{g,c}}
{\mathrm{TP}_{g,c}+\mathrm{FP}_{g,c}+\mathrm{FN}_{g,c}}.
$$

### Tier 2: Group robustness

#### Worst-group score

$$
M_{\mathrm{worst}} = \min_{g\in\mathcal{G}_d}M_g.
$$

Higher is better.

#### Normalized failure range

With group mean `M_bar`,

$$
\bar{M}=\frac{1}{K}\sum_g M_g,
\qquad
\mathrm{NFR}=
\frac{\max_gM_g-\min_gM_g}{\bar{M}},\quad\bar{M}>0.
$$

Lower is better. NFR measures the relative spread between the strongest and weakest active groups.

### Tier 3: Calibration and parity

#### Expected calibration error

Confidence scores are divided into ten equal-width bins:

$$
\mathrm{ECE} =
\sum_{m=1}^{10}\frac{|B_m|}{N}
|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)|.
$$

Lower is better.

#### Equalized-odds disparity

Using Youden's index `J_(g,c) = TPR_(g,c) - FPR_(g,c)`, FairRSFM computes

$$
\mathrm{EOdd} =
\max_{g,g'\in\mathcal{G}_d}
\frac{1}{|C|}\sum_{c\in C}|J_{g,c}-J_{g',c}|.
$$

Lower is better.

This implemented quantity is a Youden-index disparity; it is distinct from
standard equalized odds, which compares TPR and FPR separately.

#### Demographic-parity metric

Let `PPR_(g,c) = P(y_hat_c = 1 | g)` be the predicted-positive rate. Then

$$
\mathrm{DPM} =
\max_{g,g'\in\mathcal{G}_d}
\frac{1}{|C|}\sum_{c\in C}
|\mathrm{PPR}_{g,c}-\mathrm{PPR}_{g',c}|.
$$

Lower is better.

## Interpretation

No single mitigation method is expected to dominate every field. FairRSFM therefore reports overall task accuracy, worst-group robustness, calibration, and parity together. A method can improve the weakest group while reducing the average, or improve EOdd while worsening DPM. These trade-offs are part of the benchmark rather than hidden by a single composite score.

## Related Documentation

- [Benchmark protocol](benchmark.md)
- [Biome taxonomy and labeling](biome-groups.md)
- [Complete experimental results](results.md)
