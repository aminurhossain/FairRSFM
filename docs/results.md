# Complete Experimental Results

[Back to the main README](../README.md)

## Reporting Conventions

- Values are percentages and reported as `mean +/- standard deviation` over three independent seeds.
- **Overall** is Macro-F1 for m-EuroSAT, F1@opt for m-BigEarthNet, and global mIoU for segmentation datasets.
- **Worst** is the minimum score over active biome macro-groups.
- Higher Overall and Worst are better.
- Lower ECE, NFR, EOdd, and DPM are better.
- All encoders are frozen; only the downstream head or lightweight decoder is trained.
- Worst-group scores are computed within each seed and then summarized across seeds. They therefore need not equal the minimum of the rounded biome-wise means shown later.
- The values below were transcribed from the revised submission tables and independently checked in table order.

## Global and Robustness Results

### Prithvi-EO-2.0

| Dataset | Method | Overall | Worst | ECE | NFR | EOdd | DPM |
|---|---|---:|---:|---:|---:|---:|---:|
| m-EuroSAT | ERM | 90.98 +/- 2.36 | 83.72 +/- 0.87 | 4.02 +/- 1.64 | 5.96 +/- 3.28 | 20.50 +/- 1.75 | 9.91 +/- 0.21 |
|  | BOLP | 93.42 +/- 0.35 | 84.34 +/- 0.00 | **2.37 +/- 0.04** | 9.82 +/- 0.30 | **17.06 +/- 0.16** | **9.61 +/- 0.03** |
|  | DBR | 91.36 +/- 1.01 | **84.35 +/- 0.50** | 5.50 +/- 2.29 | **5.85 +/- 0.70** | 21.23 +/- 1.39 | 10.90 +/- 0.17 |
|  | GroupDRO | **94.14 +/- 0.47** | 84.34 +/- 0.00 | 3.17 +/- 0.40 | 10.34 +/- 0.51 | 17.31 +/- 0.12 | 10.04 +/- 0.10 |
| m-BigEarthNet | ERM | 60.09 +/- 0.99 | 46.12 +/- 1.51 | 2.12 +/- 0.28 | 31.14 +/- 1.99 | **42.93 +/- 1.90** | **12.66 +/- 0.27** |
|  | BOLP | **62.75 +/- 0.16** | **50.27 +/- 0.63** | 2.07 +/- 0.08 | 20.92 +/- 0.39 | 45.93 +/- 0.76 | 13.29 +/- 0.11 |
|  | DBR | 58.05 +/- 0.32 | 48.04 +/- 0.63 | 3.70 +/- 0.13 | 29.43 +/- 1.45 | 49.34 +/- 0.51 | 13.40 +/- 0.40 |
|  | GroupDRO | 54.42 +/- 0.60 | 46.82 +/- 0.60 | **1.71 +/- 0.00** | **19.19 +/- 8.16** | 46.99 +/- 3.03 | 15.34 +/- 0.03 |
| MMEarth20K | ERM | **47.99 +/- 0.32** | 33.75 +/- 0.58 | 3.47 +/- 0.29 | 28.19 +/- 0.64 | 22.51 +/- 2.36 | 9.78 +/- 0.19 |
|  | BOLP | 43.31 +/- 0.85 | 30.64 +/- 1.20 | 2.76 +/- 0.22 | 30.66 +/- 3.17 | **19.46 +/- 1.67** | **8.13 +/- 0.53** |
|  | DBR | 47.84 +/- 0.22 | 34.76 +/- 0.42 | 2.14 +/- 0.13 | 26.42 +/- 0.62 | 23.15 +/- 3.22 | 9.73 +/- 0.12 |
|  | GroupDRO | 46.99 +/- 0.62 | **37.16 +/- 0.26** | **1.69 +/- 0.08** | **15.34 +/- 1.91** | 24.45 +/- 0.20 | 10.50 +/- 0.15 |
| m-SA-Crop-Type | ERM | 27.30 +/- 0.15 | 18.47 +/- 0.16 | 2.98 +/- 0.18 | 35.31 +/- 0.54 | 24.85 +/- 0.62 | 7.68 +/- 0.25 |
|  | BOLP | **28.27 +/- 0.21** | 19.38 +/- 0.19 | 5.94 +/- 1.18 | 29.95 +/- 2.08 | 22.09 +/- 0.66 | 7.01 +/- 0.31 |
|  | DBR | 27.18 +/- 1.86 | **19.81 +/- 0.42** | **2.84 +/- 0.09** | **28.45 +/- 5.10** | 20.31 +/- 2.42 | 6.43 +/- 0.27 |
|  | GroupDRO | 28.26 +/- 0.14 | 19.09 +/- 0.26 | 6.42 +/- 0.20 | 39.63 +/- 0.94 | **16.91 +/- 0.64** | **6.19 +/- 0.01** |

### SatMAE

| Dataset | Method | Overall | Worst | ECE | NFR | EOdd | DPM |
|---|---|---:|---:|---:|---:|---:|---:|
| m-EuroSAT | ERM | **93.89 +/- 0.13** | 74.45 +/- 0.51 | **3.18 +/- 0.21** | 20.71 +/- 0.38 | 23.82 +/- 0.57 | 10.15 +/- 0.16 |
|  | BOLP | 90.82 +/- 1.31 | 76.63 +/- 2.24 | 16.39 +/- 3.73 | 16.59 +/- 3.23 | 20.82 +/- 4.79 | **9.42 +/- 0.50** |
|  | DBR | 91.75 +/- 1.60 | 81.46 +/- 2.25 | 7.35 +/- 2.82 | 11.37 +/- 3.64 | 22.54 +/- 3.90 | 10.59 +/- 0.22 |
|  | GroupDRO | 91.26 +/- 1.03 | **89.71 +/- 1.51** | 6.36 +/- 0.93 | **5.67 +/- 1.71** | **15.12 +/- 0.60** | 10.18 +/- 0.07 |
| m-BigEarthNet | ERM | **53.30 +/- 0.16** | 41.33 +/- 0.36 | 11.29 +/- 0.29 | 20.13 +/- 1.07 | 38.92 +/- 5.10 | 14.03 +/- 0.22 |
|  | BOLP | 51.55 +/- 0.17 | **43.07 +/- 0.17** | 10.82 +/- 0.29 | 17.03 +/- 0.37 | 38.71 +/- 2.78 | **7.62 +/- 0.25** |
|  | DBR | 52.74 +/- 0.28 | 42.68 +/- 0.11 | **10.14 +/- 1.35** | **13.79 +/- 1.34** | 39.38 +/- 0.69 | 13.24 +/- 0.19 |
|  | GroupDRO | 51.63 +/- 0.17 | 41.45 +/- 0.12 | 11.89 +/- 0.52 | 15.13 +/- 0.66 | **37.87 +/- 3.18** | 12.58 +/- 0.75 |
| MMEarth20K | ERM | 41.93 +/- 0.40 | 30.78 +/- 0.57 | 2.94 +/- 0.68 | 24.07 +/- 2.99 | 26.01 +/- 0.70 | 9.71 +/- 0.28 |
|  | BOLP | 40.86 +/- 1.38 | 30.27 +/- 0.14 | 2.18 +/- 1.05 | 23.51 +/- 2.11 | **22.84 +/- 0.96** | **8.86 +/- 0.29** |
|  | DBR | **42.39 +/- 0.52** | **32.74 +/- 0.04** | **1.50 +/- 0.27** | 19.65 +/- 1.28 | 25.26 +/- 0.68 | 9.76 +/- 0.10 |
|  | GroupDRO | 41.40 +/- 0.82 | 31.74 +/- 0.42 | 2.53 +/- 2.29 | **17.47 +/- 0.20** | 24.15 +/- 2.30 | 10.26 +/- 0.11 |
| m-SA-Crop-Type | ERM | 28.01 +/- 0.51 | 18.07 +/- 0.23 | 3.47 +/- 0.67 | 33.47 +/- 0.84 | 22.51 +/- 0.82 | 6.84 +/- 0.01 |
|  | BOLP | 28.64 +/- 0.06 | 18.44 +/- 0.11 | **1.98 +/- 0.38** | 33.91 +/- 0.32 | **18.80 +/- 0.67** | **5.11 +/- 0.11** |
|  | DBR | **28.79 +/- 0.46** | **19.23 +/- 0.07** | 2.83 +/- 1.00 | 31.00 +/- 1.00 | 21.13 +/- 0.33 | 6.85 +/- 0.31 |
|  | GroupDRO | 26.60 +/- 1.06 | 18.43 +/- 0.12 | 3.14 +/- 0.47 | **28.02 +/- 3.63** | 21.28 +/- 0.14 | 7.50 +/- 0.36 |

### DOFA

| Dataset | Method | Overall | Worst | ECE | NFR | EOdd | DPM |
|---|---|---:|---:|---:|---:|---:|---:|
| m-EuroSAT | ERM | 89.63 +/- 0.57 | 79.88 +/- 7.84 | 10.86 +/- 2.41 | 11.98 +/- 9.36 | 14.09 +/- 4.13 | 10.72 +/- 0.38 |
|  | BOLP | 84.46 +/- 0.56 | 76.80 +/- 0.23 | 11.81 +/- 1.06 | 12.96 +/- 1.33 | 12.71 +/- 1.03 | **6.25 +/- 0.50** |
|  | DBR | **90.98 +/- 0.71** | **88.27 +/- 1.11** | **9.83 +/- 1.78** | 4.26 +/- 0.38 | **10.36 +/- 1.04** | 10.76 +/- 0.07 |
|  | GroupDRO | 89.40 +/- 0.31 | 87.61 +/- 0.93 | 11.22 +/- 1.13 | **2.66 +/- 1.94** | 13.04 +/- 2.84 | 11.04 +/- 0.12 |
| m-BigEarthNet | ERM | **47.67 +/- 0.44** | 37.90 +/- 0.44 | 3.47 +/- 0.43 | 22.25 +/- 1.19 | 40.23 +/- 0.68 | 12.39 +/- 0.63 |
|  | BOLP | 42.79 +/- 0.22 | 36.18 +/- 0.86 | 3.61 +/- 0.73 | 22.11 +/- 3.29 | **39.48 +/- 0.88** | **7.40 +/- 0.37** |
|  | DBR | 45.89 +/- 0.57 | 38.25 +/- 0.14 | **2.72 +/- 0.46** | **14.45 +/- 0.84** | 40.09 +/- 1.98 | 12.18 +/- 0.62 |
|  | GroupDRO | 42.97 +/- 0.51 | **38.74 +/- 0.14** | 3.85 +/- 0.32 | 24.20 +/- 5.69 | 41.94 +/- 1.29 | 11.62 +/- 0.47 |
| MMEarth20K | ERM | **39.67 +/- 1.95** | 27.64 +/- 0.96 | 5.78 +/- 2.85 | 30.11 +/- 1.68 | 25.44 +/- 2.36 | 10.20 +/- 0.09 |
|  | BOLP | 38.20 +/- 0.17 | 28.31 +/- 0.41 | **4.48 +/- 1.07** | 27.04 +/- 3.08 | 25.65 +/- 0.49 | 10.00 +/- 0.17 |
|  | DBR | 38.68 +/- 2.95 | **28.54 +/- 0.40** | 7.84 +/- 3.14 | **24.39 +/- 3.16** | 22.04 +/- 3.73 | **9.75 +/- 0.33** |
|  | GroupDRO | 39.35 +/- 0.69 | 28.15 +/- 0.44 | 5.37 +/- 3.81 | 25.69 +/- 2.53 | **21.83 +/- 0.46** | 10.65 +/- 0.31 |
| m-SA-Crop-Type | ERM | 27.40 +/- 0.94 | 19.91 +/- 0.38 | 12.67 +/- 7.02 | 28.64 +/- 1.64 | 22.29 +/- 3.30 | 5.72 +/- 0.11 |
|  | BOLP | 27.18 +/- 0.59 | 20.53 +/- 0.23 | 13.51 +/- 7.18 | 24.97 +/- 1.93 | **18.99 +/- 2.31** | **5.54 +/- 0.11** |
|  | DBR | **28.20 +/- 0.49** | **21.91 +/- 0.60** | 7.66 +/- 4.86 | **22.04 +/- 1.15** | 20.90 +/- 1.54 | 6.08 +/- 0.18 |
|  | GroupDRO | 26.45 +/- 0.70 | 19.20 +/- 0.53 | **3.13 +/- 0.47** | 28.69 +/- 4.77 | 22.54 +/- 0.95 | 5.58 +/- 0.22 |

## Biome-Wise Results

### Classification and m-SA-Crop-Type

The abbreviated columns are High-Amplitude Phenological (`Pheno.`), Aseasonal High-Biomass (`Biomass`), Transitional Herbaceous and Scrub (`Trans.`), and Xeric and Mineralogical (`Xeric`).

| Backbone | Method | Euro Pheno. | Euro Biomass | Euro Trans. | BEN Pheno. | BEN Biomass | BEN Trans. | SA Trans. | SA Xeric |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prithvi-EO-2.0 | ERM | 88.96 +/- 3.38 | 84.56 +/- 0.31 | 86.71 +/- 3.79 | 59.67 +/- 0.48 | 61.14 +/- 6.05 | 46.04 +/- 1.85 | 28.55 +/- 0.19 | 18.47 +/- 0.16 |
|  | BOLP | 92.28 +/- 0.53 | 84.34 +/- 0.00 | 93.02 +/- 0.41 | 61.85 +/- 0.72 | 54.03 +/- 1.68 | 50.27 +/- 0.77 | 27.70 +/- 0.67 | 19.38 +/- 0.24 |
|  | DBR | 88.57 +/- 0.96 | 85.00 +/- 0.00 | 85.96 +/- 2.74 | 55.59 +/- 0.18 | 64.53 +/- 0.13 | 48.03 +/- 0.78 | 26.49 +/- 1.93 | 19.81 +/- 0.42 |
|  | GroupDRO | 92.82 +/- 0.70 | 84.34 +/- 0.00 | 93.68 +/- 0.50 | 49.59 +/- 0.67 | 56.78 +/- 5.15 | 46.82 +/- 0.60 | 28.52 +/- 0.11 | 19.09 +/- 0.26 |
| SatMAE | ERM | 93.99 +/- 0.14 | 74.45 +/- 0.51 | 93.62 +/- 0.13 | 52.06 +/- 0.22 | 47.97 +/- 3.69 | 41.33 +/- 0.36 | 27.45 +/- 0.48 | 18.07 +/- 0.23 |
|  | BOLP | 91.74 +/- 0.97 | 78.41 +/- 4.76 | 85.38 +/- 3.97 | 51.85 +/- 0.20 | 43.65 +/- 0.87 | 43.18 +/- 0.14 | 28.15 +/- 0.07 | 18.44 +/- 0.11 |
|  | DBR | 91.95 +/- 1.36 | 83.26 +/- 4.80 | 89.00 +/- 3.54 | 49.95 +/- 0.69 | 43.67 +/- 0.53 | 42.68 +/- 0.11 | 28.16 +/- 0.45 | 19.23 +/- 0.07 |
|  | GroupDRO | 90.16 +/- 1.03 | 94.87 +/- 0.00 | 90.22 +/- 1.99 | 49.26 +/- 0.35 | 44.78 +/- 2.50 | 41.53 +/- 0.08 | 25.93 +/- 1.17 | 18.43 +/- 0.12 |
| DOFA | ERM | 89.90 +/- 0.53 | 79.88 +/- 7.84 | 87.75 +/- 0.56 | 47.39 +/- 0.82 | 42.63 +/- 1.94 | 37.90 +/- 0.44 | 26.58 +/- 0.88 | 19.91 +/- 0.38 |
|  | BOLP | 87.32 +/- 0.98 | 79.46 +/- 3.26 | 76.90 +/- 0.21 | 44.87 +/- 0.34 | 36.78 +/- 1.21 | 36.56 +/- 0.71 | 26.39 +/- 0.53 | 20.53 +/- 0.23 |
|  | DBR | 91.52 +/- 0.38 | 89.54 +/- 2.76 | 88.49 +/- 1.08 | 43.97 +/- 0.61 | 40.12 +/- 2.48 | 38.27 +/- 0.14 | 27.33 +/- 0.46 | 21.91 +/- 0.60 |
|  | GroupDRO | 89.26 +/- 0.38 | 89.11 +/- 2.04 | 88.03 +/- 0.83 | 42.96 +/- 1.10 | 49.34 +/- 2.72 | 38.74 +/- 0.14 | 25.64 +/- 0.73 | 19.20 +/- 0.53 |

### MMEarth20K

| Backbone | Method | Pheno. | Biomass | Trans. | Cryospheric | Hydrological | Xeric |
|---|---|---:|---:|---:|---:|---:|---:|
| Prithvi-EO-2.0 | ERM | 44.19 +/- 1.39 | 41.20 +/- 1.35 | 44.54 +/- 1.04 | 43.74 +/- 0.56 | 37.28 +/- 2.14 | 35.32 +/- 2.69 |
|  | BOLP | 40.57 +/- 0.93 | 35.96 +/- 2.17 | 41.38 +/- 2.26 | 39.71 +/- 0.63 | 32.43 +/- 3.10 | 32.82 +/- 1.91 |
|  | DBR | 43.58 +/- 1.24 | 40.52 +/- 1.05 | 44.03 +/- 1.56 | 43.78 +/- 1.23 | 38.49 +/- 3.13 | 36.87 +/- 2.85 |
|  | GroupDRO | 42.77 +/- 0.33 | 40.27 +/- 0.76 | 41.53 +/- 0.08 | 43.44 +/- 0.56 | 37.16 +/- 0.26 | 40.30 +/- 0.24 |
| SatMAE | ERM | 40.89 +/- 0.81 | 34.89 +/- 0.55 | 35.86 +/- 0.35 | 37.42 +/- 0.50 | 30.78 +/- 0.57 | 34.11 +/- 1.08 |
|  | BOLP | 39.91 +/- 1.31 | 33.13 +/- 2.08 | 35.29 +/- 1.23 | 38.09 +/- 0.93 | 30.27 +/- 0.14 | 33.76 +/- 0.58 |
|  | DBR | 41.07 +/- 0.61 | 35.57 +/- 0.30 | 36.84 +/- 0.29 | 38.43 +/- 0.64 | 32.74 +/- 0.04 | 34.25 +/- 0.08 |
|  | GroupDRO | 38.81 +/- 0.40 | 35.71 +/- 0.29 | 33.97 +/- 0.70 | 37.89 +/- 1.89 | 31.93 +/- 0.36 | 32.92 +/- 1.08 |
| DOFA | ERM | 37.80 +/- 0.87 | 33.73 +/- 0.61 | 35.97 +/- 1.93 | 35.30 +/- 1.99 | 27.64 +/- 0.96 | 32.46 +/- 3.26 |
|  | BOLP | 37.14 +/- 1.52 | 32.75 +/- 0.75 | 34.60 +/- 0.45 | 33.36 +/- 0.16 | 28.31 +/- 0.41 | 29.48 +/- 1.22 |
|  | DBR | 36.15 +/- 2.65 | 32.24 +/- 2.04 | 36.05 +/- 2.24 | 35.79 +/- 1.76 | 28.54 +/- 0.40 | 32.90 +/- 3.06 |
|  | GroupDRO | 36.56 +/- 0.96 | 34.81 +/- 0.98 | 35.24 +/- 0.77 | 35.66 +/- 0.74 | 28.15 +/- 0.44 | 30.81 +/- 1.62 |

## Interpretation

### Aggregate performance is insufficient

Under Prithvi ERM, m-EuroSAT reaches 90.98 overall Macro-F1 but only 83.72 on its weakest active macro-group. The m-BigEarthNet gap is larger: 60.09 overall versus 46.12 worst-group F1@opt. m-SA-Crop-Type obtains 27.30 global mIoU while its weakest group reaches 18.47.

### BOLP is strong for Prithvi classification

On m-EuroSAT, BOLP improves overall performance from 90.98 to 93.42 and worst-group performance from 83.72 to 84.34. On m-BigEarthNet, it improves overall F1@opt from 60.09 to 62.75, worst-group performance from 46.12 to 50.27, and NFR from 31.14 to 20.92.

### DBR generalizes across architectures

DBR produces the strongest overall and worst-group m-SA-Crop-Type results for SatMAE and DOFA. It also gives SatMAE the strongest MMEarth20K overall and worst-group scores, and substantially improves DOFA's m-EuroSAT robustness.

### GroupDRO can strongly improve the weakest group

SatMAE GroupDRO raises m-EuroSAT worst-group Macro-F1 from 74.45 to 89.71 and reduces NFR from 20.71 to 5.67. Prithvi GroupDRO gives the strongest MMEarth20K worst-group score, 37.16.

### Dense prediction remains task dependent

BOLP does not uniformly improve segmentation. On Prithvi MMEarth20K, it reduces both overall and worst-group mIoU, indicating that biome-associated feature directions can also carry useful land-cover information. FairRSFM therefore treats mitigation as a measurable trade-off rather than assuming one method will dominate every task.

## Limitations

- Patch-centroid biome labels can be noisy near ecological boundaries.
- Datasets cover different subsets of the taxonomy, so worst-group values are relative to their active groups.
- Pixel-level fairness metrics for segmentation require choices that may differ from classification interpretations.
- Biome robustness captures only one aspect of geographic bias and does not replace analysis of sensor, country, resolution, temporal, source, or socio-economic disparities.
- Unequal satellite imagery availability can itself create regional coverage bias outside the biome taxonomy.

## Related Documentation

- [Benchmark protocol](benchmark.md)
- [Biome taxonomy and labeling](biome-groups.md)
- [Methods and evaluation metrics](methods.md)
