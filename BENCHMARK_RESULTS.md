# FairRSFM 48-Run Benchmark Summary (1 Full Epoch)
**Last Updated:** 2026-10-21 16:22:24  
**Completed Runs:** 26 / 48  
**Currently Running:** 1  

| # | Model | Dataset | Method | Status | Overall Perf (F1/mIoU) | Worst Group | PER | Gini | Eq. Odds | Log |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PRITHVI | eurosat | ERM | ✅ SUCCESS | **0.6406** | 0.3968 | 0.5690 | 0.1182 | 0.3152 | `prithvi_eurosat_erm.log` |
| 2 | PRITHVI | eurosat | DBR | ✅ SUCCESS | **0.4718** | 0.3984 | 0.7453 | 0.0638 | 0.1616 | `prithvi_eurosat_dbr.log` |
| 3 | PRITHVI | eurosat | BOLP | ✅ SUCCESS | **0.6844** | 0.4611 | 0.6713 | 0.0837 | 0.3145 | `prithvi_eurosat_bolp.log` |
| 4 | PRITHVI | eurosat | GROUPDRO | ✅ SUCCESS | **0.5437** | 0.4715 | 0.8501 | 0.0366 | 0.3239 | `prithvi_eurosat_groupdro.log` |
| 5 | PRITHVI | bigearthnet | ERM | ✅ SUCCESS | **0.3781** | 0.2836 | 0.7746 | 0.0585 | 0.3484 | `prithvi_bigearthnet_erm.log` |
| 6 | PRITHVI | bigearthnet | DBR | ✅ SUCCESS | **0.3134** | 0.2604 | 0.5981 | 0.1201 | 0.3090 | `prithvi_bigearthnet_dbr.log` |
| 7 | PRITHVI | bigearthnet | BOLP | ✅ SUCCESS | **0.3606** | 0.2541 | 0.7331 | 0.0710 | 0.2832 | `prithvi_bigearthnet_bolp.log` |
| 8 | PRITHVI | bigearthnet | GROUPDRO | ✅ SUCCESS | - | 0.2643 | 0.6300 | 0.1079 | 0.2531 | `prithvi_bigearthnet_groupdro.log` |
| 9 | PRITHVI | sa_crop | ERM | ✅ SUCCESS | **0.1611** | 0.1294 | 0.8057 | 0.0538 | 0.0839 | `prithvi_sa_crop_erm.log` |
| 10 | PRITHVI | sa_crop | DBR | ✅ SUCCESS | **0.1425** | 0.1365 | 0.8381 | 0.0440 | 0.1098 | `prithvi_sa_crop_dbr.log` |
| 11 | PRITHVI | sa_crop | BOLP | ✅ SUCCESS | **0.1836** | 0.1504 | 0.8092 | 0.0527 | 0.1330 | `prithvi_sa_crop_bolp.log` |
| 12 | PRITHVI | sa_crop | GROUPDRO | ✅ SUCCESS | **16.40%** | - | 0.8102 | 0.0524 | 0.0453 | `prithvi_sa_crop_groupdro.log` |
| 13 | PRITHVI | mmearth | ERM | ❌ FAILED | - | - | - | - | - | `prithvi_mmearth_erm.log` |
| 14 | PRITHVI | mmearth | DBR | ❌ FAILED | - | - | - | - | - | `prithvi_mmearth_dbr.log` |
| 15 | PRITHVI | mmearth | BOLP | ❌ FAILED | - | - | - | - | - | `prithvi_mmearth_bolp.log` |
| 16 | PRITHVI | mmearth | GROUPDRO | ❌ FAILED | - | - | - | - | - | `prithvi_mmearth_groupdro.log` |
| 17 | SATMAE | eurosat | ERM | ✅ SUCCESS | **0.8223** | 0.5470 | 0.6627 | 0.0869 | 0.4814 | `satmae_eurosat_erm.log` |
| 18 | SATMAE | eurosat | DBR | ✅ SUCCESS | **0.8170** | 0.5470 | 0.6654 | 0.0862 | 0.4961 | `satmae_eurosat_dbr.log` |
| 19 | SATMAE | eurosat | BOLP | ✅ SUCCESS | **0.7951** | 0.4222 | 0.5069 | 0.1431 | 0.4988 | `satmae_eurosat_bolp.log` |
| 20 | SATMAE | eurosat | GROUPDRO | ✅ SUCCESS | **0.8002** | 0.7222 | 0.8484 | 0.0369 | 0.3838 | `satmae_eurosat_groupdro.log` |
| 21 | SATMAE | bigearthnet | ERM | ✅ SUCCESS | - | 0.3355 | 0.7641 | 0.0594 | 0.3591 | `satmae_bigearthnet_erm.log` |
| 22 | SATMAE | bigearthnet | DBR | ✅ SUCCESS | - | 0.3263 | 0.7726 | 0.0577 | 0.3758 | `satmae_bigearthnet_dbr.log` |
| 23 | SATMAE | bigearthnet | BOLP | ✅ SUCCESS | - | 0.2440 | 0.5681 | 0.1215 | 0.3725 | `satmae_bigearthnet_bolp.log` |
| 24 | SATMAE | bigearthnet | GROUPDRO | ✅ SUCCESS | - | 0.2724 | 0.6551 | 0.0973 | 0.3686 | `satmae_bigearthnet_groupdro.log` |
| 25 | SATMAE | sa_crop | ERM | ✅ SUCCESS | **0.3683** | 0.1699 | 0.6830 | 0.0942 | 0.1985 | `satmae_sa_crop_erm.log` |
| 26 | SATMAE | sa_crop | DBR | ✅ SUCCESS | **0.3680** | 0.1694 | 0.6809 | 0.0949 | 0.1977 | `satmae_sa_crop_dbr.log` |
| 27 | SATMAE | sa_crop | BOLP | ✅ SUCCESS | **0.3585** | 0.1534 | 0.6107 | 0.1208 | 0.1324 | `satmae_sa_crop_bolp.log` |
| 28 | SATMAE | sa_crop | GROUPDRO | ✅ SUCCESS | **0.3603** | 0.2183 | 0.9282 | 0.0186 | 0.1590 | `satmae_sa_crop_groupdro.log` |
| 29 | SATMAE | mmearth | ERM | ✅ SUCCESS | **0.4986** | 0.2539 | 0.6604 | 0.0691 | 0.2026 | `satmae_mmearth_erm.log` |
| 30 | SATMAE | mmearth | DBR | ✅ SUCCESS | **0.4985** | 0.2530 | 0.6562 | 0.0697 | 0.2020 | `satmae_mmearth_dbr.log` |
| 31 | SATMAE | mmearth | BOLP | 🔄 RUNNING | - | - | - | - | - | `satmae_mmearth_bolp.log` |
