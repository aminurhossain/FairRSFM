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

Every method keeps `phi` fixed and trains only the downstream head `h_theta`. The task loss is cross-entropy for single-label classification, binary cross-entropy for multi-label classification, and a decoder-level loss for segmentation.

## Compared Strategies

| Method | Stage | Data retained | Backbone updates | Additional trainable parameters |
|---|---|---:|---:|---:|
| ERM | Standard task objective | All | No | No |
| DBR | Loss reweighting | All | No | No |
| BOLP | Representation transformation | All | No | No |
| GroupDRO | Group-robust objective | All | No | No |

## Empirical Risk Minimization

ERM is the frozen-feature baseline:

$$
\mathcal{L}_{\mathrm{ERM}} = \frac{1}{N}\sum_{i=1}^{N}\ell(h_{\theta}(z_i), y_i).
$$

It does not use biome labels during optimization. Biome metadata is used only to compute group-wise evaluation metrics.

## Dynamic Biome Reweighting

Dynamic Biome Reweighting with Validation Feedback (DBR) is a loss-level mitigation method. It begins with inverse-frequency weights and adapts them using current validation performance.

Let `p_g = n_g / N` be the empirical frequency of group `g`. Initial normalized weights are

$$
w_g^{(0)} = \frac{1/p_g}{\frac{1}{|\mathcal{G}|}\sum_{g'\in\mathcal{G}}1/p_{g'}}.
$$

At update step `t`, let `M_g^(t)` be the validation metric for group `g` and let

$$
\bar{M}^{(t)} = \frac{1}{|\mathcal{G}|}\sum_g M_g^{(t)}.
$$

DBR increases the weight of below-mean groups and decreases the weight of above-mean groups:

$$
\tilde{w}^{(t+1)}_g = w^{(t)}_g
\left(1 + \alpha \frac{\bar{M}^{(t)} - M_g^{(t)}}{\bar{M}^{(t)} + \epsilon}\right),
$$

followed by unit-mean normalization,

$$
w^{(t+1)}_g =
\frac{\tilde{w}^{(t+1)}_g}
{\frac{1}{|\mathcal{G}|}\sum_{g'}\tilde{w}^{(t+1)}_{g'}}.
$$

The training objective is

$$
\mathcal{L}_{\mathrm{DBR}} =
\frac{1}{N}\sum_{i=1}^{N}w_{g_i}^{(t)}\ell(h_{\theta}(\phi(x_i)), y_i).
$$

DBR is non-adversarial, parameter-free beyond the task head, and applicable to classification and segmentation.

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

the leading `k` left singular vectors `U_k` span the dominant inter-group directions. Since the columns of `B` sum to zero, at most `|G|-1` directions are non-trivial.

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

The centroids, global mean, and projector are computed from training embeddings and frozen. If a group label is unavailable during inference, group residualization is skipped and the global projector is still applied.

## GroupDRO

GroupDRO is included as a group-robust optimization baseline. It emphasizes groups with higher loss so that training targets robust performance rather than average empirical risk. In FairRSFM it uses the same biome macro-groups, frozen encoder, task head, data, optimization budget, and checkpoint-selection protocol as the other methods.

## Evaluation Metrics

Let `G_d` be the active biome macro-groups in dataset `d`, with `K = |G_d|`, and let `M_g` be the task score for group `g`.

### Tier 1: Task performance

#### Single-label classification

For m-EuroSAT, group performance is macro-F1 over classes present in the group:

$$
\mathrm{Macro\text{-}F1}_g =
\frac{1}{|C_g|}\sum_{c\in C_g}\mathrm{F1}_{g,c}.
$$

#### Multi-label classification

For m-BigEarthNet, class thresholds `tau_c` are selected on validation data over `[0.1, 0.95]` and fixed for test evaluation:

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
\frac{\max_gM_g-\min_gM_g}{\bar{M}+\epsilon}.
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
