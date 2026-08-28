# Biome Taxonomy and Labeling

[Back to the main README](../README.md)

## Why Biomes?

Earth observation data are spatially and ecologically structured. The same semantic class can have different spectral, textural, phenological, and hydrological signatures across forests, grasslands, deserts, tundra, Mediterranean regions, and wetlands.

FairRSFM defines **biome bias** as a systematic disparity in downstream RSFM performance across biome-derived ecological groups. Biomes provide a globally defined and physically interpretable grouping variable for evaluating this disparity.

## Labeling Pipeline

1. Read the sample's georeferenced patch extent.
2. Compute its centroid latitude and longitude.
3. Spatially join the centroid with the [RESOLVE Ecoregions 2017](https://developers.google.com/earth-engine/datasets/catalog/RESOLVE_ECOREGIONS_2017) reference map.
4. Store the resulting raw biome ID as sample metadata.
5. Map the raw biome ID to one of six macro-groups.

For segmentation, the entire patch receives the biome label of its centroid. This keeps the protocol consistent across classification and dense prediction, but it can introduce noise for patches crossing ecological boundaries.

## Fourteen Raw Terrestrial Biomes

| ID | Raw terrestrial biome |
|---:|---|
| 1 | Tropical and Subtropical Moist Broadleaf Forests |
| 2 | Tropical and Subtropical Dry Broadleaf Forests |
| 3 | Tropical and Subtropical Coniferous Forests |
| 4 | Temperate Broadleaf and Mixed Forests |
| 5 | Temperate Conifer Forests |
| 6 | Boreal Forests / Taiga |
| 7 | Tropical and Subtropical Grasslands, Savannas, and Shrublands |
| 8 | Temperate Grasslands, Savannas, and Shrublands |
| 9 | Flooded Grasslands and Savannas |
| 10 | Montane Grasslands and Shrublands |
| 11 | Tundra |
| 12 | Mediterranean Forests, Woodlands, and Scrub |
| 13 | Deserts and Xeric Shrublands |
| 14 | Mangroves |

Directly treating all 14 labels as primary groups can produce unstable estimates when a downstream dataset contains very few samples from some biomes. FairRSFM therefore retains the raw labels for diagnosis and uses six consolidated macro-groups for primary evaluation.

## Six Macro-Groups

| ID | Macro-group | Included raw biomes | Physical and remote sensing rationale |
|---:|---|---|---|
| 1 | Aseasonal High-Biomass | 1 Tropical Moist Forests; 3 Tropical Conifer Forests; 5 Temperate Conifer Forests | Dense, high-biomass vegetation with relatively persistent canopy structure. |
| 2 | High-Amplitude Phenological | 2 Tropical Dry Forests; 4 Temperate Broadleaf and Mixed Forests; 6 Boreal Forests / Taiga | Forested regions characterized by strong seasonal or phenological variation. |
| 3 | Transitional Herbaceous and Scrub | 7 Tropical Grasslands / Savannas / Shrublands; 8 Temperate Grasslands / Savannas / Shrublands; 12 Mediterranean Forests / Woodlands / Scrub | Open or mixed vegetation with strong grass, shrub, soil, and canopy heterogeneity. |
| 4 | Cryospheric and Short-Cycle | 10 Montane Grasslands and Shrublands; 11 Tundra | Temperature-restricted ecosystems with short growing periods or cryospheric influence. |
| 5 | Xeric and Mineralogical | 13 Deserts and Xeric Shrublands | Vegetation-sparse surfaces dominated by albedo, exposed soil, and mineral background. |
| 6 | Hydrologically Modulated | 9 Flooded Grasslands and Savannas; 14 Mangroves | Water-influenced ecosystems where inundation strongly affects spectral response. |

## Global Distribution

<p align="center">
  <img src="../assets/global-biome-macro-groups.png" alt="Global distribution of FairRSFM biome macro-groups" width="100%">
</p>

The macro-groups are intended as ecological analysis strata rather than administrative or demographic groups. They capture important environmental variation but do not constitute a complete geographic fairness audit.

## MMEarth20K Group Composition

| Macro-group | Total | Train (80%) | Validation (10%) | Test (10%) |
|---|---:|---:|---:|---:|
| Aseasonal High-Biomass | 5,600 | 4,480 | 560 | 560 |
| High-Amplitude Phenological | 5,100 | 4,080 | 510 | 510 |
| Transitional Herbaceous and Scrub | 3,600 | 2,880 | 360 | 360 |
| Cryospheric and Short-Cycle | 2,800 | 2,240 | 280 | 280 |
| Xeric and Mineralogical | 1,700 | 1,360 | 170 | 170 |
| Hydrologically Modulated | 1,200 | 960 | 120 | 120 |
| **Total** | **20,000** | **16,000** | **2,000** | **2,000** |

<p align="center">
  <img src="../assets/macro-group-distribution.png" alt="MMEarth20K group composition by split" width="95%">
</p>

The distribution figure retains the zero-based plotting indices used by the
experiment artifact: Groups 0-5 correspond in order to taxonomy IDs 1-6 in the
table above.

## Dataset Coverage

Not every downstream dataset covers every macro-group:

- m-EuroSAT and m-BigEarthNet primarily cover the two forest groups and Transitional Herbaceous and Scrub.
- m-SA-Crop-Type covers Transitional Herbaceous and Scrub plus Xeric and Mineralogical.
- MMEarth20K covers all six macro-groups.

All worst-group and disparity metrics are computed over the active matched groups present in the relevant dataset split.

## Interpretation Boundaries

- Centroid labeling may be noisy near biome boundaries.
- A patch can contain mixed ecological conditions even when assigned one label.
- Sparse raw biome classes motivate macro-group consolidation but reduce fine-grained resolution.
- Biome groups do not capture country, sensor, resolution, temporal, socio-economic, or imagery-availability bias.
- Worst-group scores are dataset-specific and should be interpreted relative to the groups actually covered.

## Related Documentation

- [Benchmark protocol](benchmark.md)
- [Methods and evaluation metrics](methods.md)
- [Complete experimental results](results.md)
