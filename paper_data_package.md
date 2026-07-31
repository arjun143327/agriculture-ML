# Paper Data Package

## Flagged Summary

- Genuine bug fixed in code path: spatial feature-importance seeds previously reused the same deterministic GroupKFold split, which can force spatial SD to 0.0000.
- Genuine bug fixed in scope: Wheat 2017 anomaly diagnostic was missing and is now computed explicitly.
- Coverage fix: Wilcoxon outputs are now generated for both 5-fold and 10-fold comparisons across all four datasets.
- Uncertain legacy item: a distinct single-holdout pilot table could not be verified from local scripts/files, so it is flagged rather than invented.

## 1. Main Results Tables

| Dataset                 | Model         | Split          | RMSE            | R2               |
| ----------------------- | ------------- | -------------- | --------------- | ---------------- |
| CY-Bench Maize (Europe) | Null Baseline | Random         | 2.401 +/- 0.073 | -0.005 +/- 0.007 |
| CY-Bench Maize (Europe) | Null Baseline | Spatial        | 2.437 +/- 0.114 | -0.009 +/- 0.011 |
| CY-Bench Maize (Europe) | Null Baseline | Temporal       | 2.497 +/- 0.502 | -0.560 +/- 1.126 |
| CY-Bench Maize (Europe) | Null Baseline | Spatiotemporal | 3.062 +/- 0.262 | -0.900 +/- 0.305 |
| CY-Bench Maize (Europe) | Ridge         | Random         | 1.680 +/- 0.031 | 0.507 +/- 0.031  |
| CY-Bench Maize (Europe) | Ridge         | Spatial        | 1.860 +/- 0.271 | 0.390 +/- 0.225  |
| CY-Bench Maize (Europe) | Ridge         | Temporal       | 1.751 +/- 0.356 | 0.108 +/- 0.892  |
| CY-Bench Maize (Europe) | Ridge         | Spatiotemporal | 3.018 +/- 0.417 | -1.008 +/- 0.963 |
| CY-Bench Maize (Europe) | Random Forest | Random         | 1.274 +/- 0.046 | 0.716 +/- 0.027  |
| CY-Bench Maize (Europe) | Random Forest | Spatial        | 1.558 +/- 0.141 | 0.576 +/- 0.105  |
| CY-Bench Maize (Europe) | Random Forest | Temporal       | 1.522 +/- 0.423 | 0.293 +/- 0.864  |
| CY-Bench Maize (Europe) | Random Forest | Spatiotemporal | 2.695 +/- 0.281 | -0.550 +/- 0.557 |
| CY-Bench Maize (Europe) | XGBoost       | Random         | 1.253 +/- 0.023 | 0.725 +/- 0.025  |
| CY-Bench Maize (Europe) | XGBoost       | Spatial        | 1.742 +/- 0.232 | 0.463 +/- 0.183  |
| CY-Bench Maize (Europe) | XGBoost       | Temporal       | 1.521 +/- 0.417 | 0.306 +/- 0.761  |
| CY-Bench Maize (Europe) | XGBoost       | Spatiotemporal | 2.789 +/- 0.306 | -0.662 +/- 0.602 |
| CY-Bench Wheat (Europe) | Null Baseline | Random         | 1.242 +/- 0.037 | -0.007 +/- 0.009 |
| CY-Bench Wheat (Europe) | Null Baseline | Spatial        | 1.203 +/- 0.137 | -0.066 +/- 0.092 |
| CY-Bench Wheat (Europe) | Null Baseline | Temporal       | 1.176 +/- 0.164 | -0.394 +/- 0.410 |
| CY-Bench Wheat (Europe) | Null Baseline | Spatiotemporal | 1.357 +/- 0.072 | -0.746 +/- 0.382 |
| CY-Bench Wheat (Europe) | Ridge         | Random         | 0.886 +/- 0.044 | 0.486 +/- 0.045  |
| CY-Bench Wheat (Europe) | Ridge         | Spatial        | 0.971 +/- 0.146 | 0.275 +/- 0.246  |
| CY-Bench Wheat (Europe) | Ridge         | Temporal       | 0.904 +/- 0.207 | 0.116 +/- 0.568  |
| CY-Bench Wheat (Europe) | Ridge         | Spatiotemporal | 1.271 +/- 0.101 | -0.552 +/- 0.411 |
| CY-Bench Wheat (Europe) | Random Forest | Random         | 0.709 +/- 0.032 | 0.671 +/- 0.020  |
| CY-Bench Wheat (Europe) | Random Forest | Spatial        | 0.877 +/- 0.147 | 0.434 +/- 0.104  |
| CY-Bench Wheat (Europe) | Random Forest | Temporal       | 0.827 +/- 0.133 | 0.265 +/- 0.377  |
| CY-Bench Wheat (Europe) | Random Forest | Spatiotemporal | 1.147 +/- 0.064 | -0.240 +/- 0.229 |
| CY-Bench Wheat (Europe) | XGBoost       | Random         | 0.701 +/- 0.020 | 0.678 +/- 0.020  |
| CY-Bench Wheat (Europe) | XGBoost       | Spatial        | 0.884 +/- 0.092 | 0.420 +/- 0.079  |
| CY-Bench Wheat (Europe) | XGBoost       | Temporal       | 0.851 +/- 0.156 | 0.220 +/- 0.419  |
| CY-Bench Wheat (Europe) | XGBoost       | Spatiotemporal | 1.179 +/- 0.021 | -0.308 +/- 0.212 |
| CY-Bench Maize (Zambia) | Null Baseline | Random         | 0.940 +/- 0.061 | -0.005 +/- 0.004 |
| CY-Bench Maize (Zambia) | Null Baseline | Spatial        | 0.941 +/- 0.071 | -0.068 +/- 0.082 |
| CY-Bench Maize (Zambia) | Null Baseline | Temporal       | 0.980 +/- 0.154 | -0.307 +/- 0.252 |
| CY-Bench Maize (Zambia) | Null Baseline | Spatiotemporal | 1.081 +/- 0.047 | -0.261 +/- 0.206 |
| CY-Bench Maize (Zambia) | Ridge         | Random         | 0.753 +/- 0.072 | 0.356 +/- 0.047  |
| CY-Bench Maize (Zambia) | Ridge         | Spatial        | 0.766 +/- 0.062 | 0.293 +/- 0.060  |
| CY-Bench Maize (Zambia) | Ridge         | Temporal       | 0.769 +/- 0.191 | 0.180 +/- 0.318  |
| CY-Bench Maize (Zambia) | Ridge         | Spatiotemporal | 0.828 +/- 0.025 | 0.264 +/- 0.103  |
| CY-Bench Maize (Zambia) | Random Forest | Random         | 0.685 +/- 0.065 | 0.466 +/- 0.051  |
| CY-Bench Maize (Zambia) | Random Forest | Spatial        | 0.788 +/- 0.076 | 0.250 +/- 0.098  |
| CY-Bench Maize (Zambia) | Random Forest | Temporal       | 0.722 +/- 0.186 | 0.264 +/- 0.303  |
| CY-Bench Maize (Zambia) | Random Forest | Spatiotemporal | 0.801 +/- 0.052 | 0.308 +/- 0.114  |
| CY-Bench Maize (Zambia) | XGBoost       | Random         | 0.725 +/- 0.073 | 0.402 +/- 0.063  |
| CY-Bench Maize (Zambia) | XGBoost       | Spatial        | 0.831 +/- 0.087 | 0.171 +/- 0.056  |
| CY-Bench Maize (Zambia) | XGBoost       | Temporal       | 0.762 +/- 0.189 | 0.177 +/- 0.336  |
| CY-Bench Maize (Zambia) | XGBoost       | Spatiotemporal | 0.832 +/- 0.052 | 0.251 +/- 0.140  |
| SustainBench Soybean    | Null Baseline | Random         | 0.690 +/- 0.003 | -0.000 +/- 0.000 |
| SustainBench Soybean    | Null Baseline | Spatial        | 0.699 +/- 0.030 | -0.001 +/- 0.001 |
| SustainBench Soybean    | Null Baseline | Temporal       | 0.752 +/- 0.081 | -0.384 +/- 0.578 |
| SustainBench Soybean    | Null Baseline | Spatiotemporal | 0.812 +/- 0.009 | -0.691 +/- 0.153 |
| SustainBench Soybean    | Ridge         | Random         | 0.683 +/- 0.128 | -0.014 +/- 0.398 |
| SustainBench Soybean    | Ridge         | Spatial        | 0.670 +/- 0.137 | 0.040 +/- 0.434  |
| SustainBench Soybean    | Ridge         | Temporal       | 0.670 +/- 0.079 | -0.120 +/- 0.557 |
| SustainBench Soybean    | Ridge         | Spatiotemporal | 0.761 +/- 0.010 | -0.487 +/- 0.149 |
| SustainBench Soybean    | Random Forest | Random         | 0.537 +/- 0.012 | 0.394 +/- 0.023  |
| SustainBench Soybean    | Random Forest | Spatial        | 0.544 +/- 0.021 | 0.392 +/- 0.029  |
| SustainBench Soybean    | Random Forest | Temporal       | 0.653 +/- 0.072 | -0.058 +/- 0.497 |
| SustainBench Soybean    | Random Forest | Spatiotemporal | 0.742 +/- 0.016 | -0.411 +/- 0.108 |
| SustainBench Soybean    | XGBoost       | Random         | 0.556 +/- 0.010 | 0.351 +/- 0.019  |
| SustainBench Soybean    | XGBoost       | Spatial        | 0.561 +/- 0.020 | 0.356 +/- 0.016  |
| SustainBench Soybean    | XGBoost       | Temporal       | 0.668 +/- 0.082 | -0.111 +/- 0.548 |
| SustainBench Soybean    | XGBoost       | Spatiotemporal | 0.747 +/- 0.015 | -0.433 +/- 0.129 |

## 2. Wilcoxon Significance Tests

| Dataset                 | Folds | SpatialWorseFolds | Statistic | PValue       | WPlus | WMinus | RankBiserialR       |
| ----------------------- | ----- | ----------------- | --------- | ------------ | ----- | ------ | ------------------- |
| CY-Bench Maize (Europe) | 5     | 5                 | 15.0      | 0.03125      | 15.0  | 0.0    | 1.0                 |
| CY-Bench Maize (Europe) | 10    | 10                | 55.0      | 0.0009765625 | 55.0  | 0.0    | 1.0                 |
| CY-Bench Wheat (Europe) | 5     | 4                 | 14.0      | 0.0625       | 14.0  | 1.0    | 0.8666666666666667  |
| CY-Bench Wheat (Europe) | 10    | 9                 | 53.0      | 0.0029296875 | 53.0  | 2.0    | 0.9272727272727272  |
| CY-Bench Maize (Zambia) | 5     | 4                 | 13.0      | 0.09375      | 13.0  | 2.0    | 0.7333333333333333  |
| CY-Bench Maize (Zambia) | 10    | 5                 | 37.0      | 0.1875       | 37.0  | 18.0   | 0.34545454545454546 |
| SustainBench Soybean    | 5     | 3                 | 9.0       | 0.40625      | 9.0   | 6.0    | 0.2                 |
| SustainBench Soybean    | 10    | 8                 | 37.0      | 0.1875       | 37.0  | 18.0   | 0.34545454545454546 |

## 3. Feature Importance Stability (Top 8 by random-mean importance)

### CY-Bench Maize (Europe)

| feature        | rand_mean            | rand_sd               | spat_mean            | spat_sd                |
| -------------- | -------------------- | --------------------- | -------------------- | ---------------------- |
| fpar_mean      | 0.343316912651062    | 0.013940158300101757  | 0.34309136867523193  | 0.004374309908598661   |
| bulk_density   | 0.1751929074525833   | 0.002894648350775242  | 0.19398756325244904  | 0.009418611414730549   |
| awc            | 0.08573772758245468  | 0.0036136191338300705 | 0.0851769745349884   | 0.003980190958827734   |
| fpar_max       | 0.0705922320485115   | 0.0037963036447763443 | 0.06526961177587509  | 0.00024635280715301633 |
| rad_mean       | 0.06459302455186844  | 0.00634451350197196   | 0.05686655640602112  | 0.0015342606930062175  |
| et0_mean       | 0.04948599636554718  | 0.006066946312785149  | 0.030169708654284477 | 0.001002224045805633   |
| drainage_class | 0.04032773897051811  | 0.011630409397184849  | 0.0617874450981617   | 0.004358299076557159   |
| ndvi_mean      | 0.030062677338719368 | 0.006621718872338533  | 0.030068064108490944 | 0.0012020384892821312  |

### CY-Bench Wheat (Europe)

| feature        | rand_mean            | rand_sd               | spat_mean            | spat_sd               |
| -------------- | -------------------- | --------------------- | -------------------- | --------------------- |
| fpar_mean      | 0.24744975566864014  | 0.03944757208228111   | 0.24451588094234467  | 0.009884456172585487  |
| drainage_class | 0.224493607878685    | 0.05711963400244713   | 0.23384636640548706  | 0.02944098226726055   |
| bulk_density   | 0.1248418465256691   | 0.0027615330182015896 | 0.15823812782764435  | 0.003706118557602167  |
| awc            | 0.062112707644701004 | 0.006612355820834637  | 0.057180989533662796 | 0.0012069374788552523 |
| ndvi_mean      | 0.054028164595365524 | 0.003268731525167823  | 0.0410442128777504   | 0.003383080940693617  |
| fpar_max       | 0.051578279584646225 | 0.0008423045510426164 | 0.04790143296122551  | 0.003302172990515828  |
| rad_mean       | 0.03990553691983223  | 0.00563460448756814   | 0.031862009316682816 | 0.0050438325852155685 |
| vpd_mean       | 0.031574033200740814 | 0.006114746909588575  | 0.027562880888581276 | 0.0017814745660871267 |

### CY-Bench Maize (Zambia)

| feature      | rand_mean           | rand_sd               | spat_mean           | spat_sd               |
| ------------ | ------------------- | --------------------- | ------------------- | --------------------- |
| tavg_mean    | 0.29622790217399597 | 0.03230930119752884   | 0.21465694904327393 | 0.028507910668849945  |
| tmax_mean    | 0.10113394260406494 | 0.027896760031580925  | 0.11679136008024216 | 0.021978646516799927  |
| bulk_density | 0.07934102416038513 | 0.005611823406070471  | 0.10081833600997925 | 0.006920053157955408  |
| cwb_mean     | 0.07555881887674332 | 0.00954530667513609   | 0.063944973051548   | 0.001514287549071014  |
| vpd_mean     | 0.06233276054263115 | 0.001171501469798386  | 0.1078493595123291  | 0.015755031257867813  |
| ndvi_mean    | 0.0593080036342144  | 0.015572071075439453  | 0.04940536618232727 | 0.002959647448733449  |
| fpar_max     | 0.05597289279103279 | 0.0056042755022645    | 0.05604889988899231 | 0.002810786012560129  |
| awc          | 0.04657972976565361 | 0.0025785593315958977 | 0.04710584878921509 | 0.0013039924670010805 |

## 4. Zambia 2012 and 2014 Anomaly Diagnostics

| year | historical_mean_excl_2012_2014 | historical_std_excl_2012_2014 | year_mean          | regions | records | z_score            | direction        |
| ---- | ------------------------------ | ----------------------------- | ------------------ | ------- | ------- | ------------------ | ---------------- |
| 2012 | 1.7061793427230043             | 0.8966266724777039            | 2.3814366197183103 | 71      | 71      | 0.7531086211492302 | positive anomaly |
| 2014 | 1.7061793427230043             | 0.8966266724777039            | 2.4094366197183095 | 71      | 71      | 0.7843367798238154 | positive anomaly |

| year   | mean_yield         |
| ------ | ------------------ |
| 2001.0 | 1.1516338028169015 |
| 2002.0 | 1.0719718309859154 |
| 2003.0 | 1.5570422535211268 |
| 2004.0 | 1.8419718309859154 |
| 2005.0 | 1.1121549295774646 |
| 2006.0 | 1.7503521126760564 |
| 2007.0 | 1.4812816901408452 |
| 2008.0 | 1.3949295774647887 |
| 2009.0 | 1.6924084507042254 |
| 2010.0 | 2.0086338028169015 |
| 2011.0 | 2.20225352112676   |
| 2012.0 | 2.38143661971831   |
| 2013.0 | 2.0333661971830987 |
| 2014.0 | 2.40943661971831   |
| 2015.0 | 1.9152394366197183 |
| 2016.0 | 2.1767323943661974 |
| 2017.0 | 2.2027183098591547 |

## 5. Wheat 2017 Diagnostics

### Overall

| regions_evaluated | historical_mean_excl_2017 | historical_std_excl_2017 | yield_2017_mean   | difference_mean    | mean_region_z_score | direction        |
| ----------------- | ------------------------- | ------------------------ | ----------------- | ------------------ | ------------------- | ---------------- |
| 46                | 3.9309742458714236        | 0.8137854200832377       | 4.909456521739131 | 0.9784822758677063 | 1.2134881567562994  | positive anomaly |

### By Country

| country_code | regions | historical_mean_excl_2017 | yield_2017_mean   | difference_mean      | mean_region_z_score | direction        |
| ------------ | ------- | ------------------------- | ----------------- | -------------------- | ------------------- | ---------------- |
| AT           | 9       | 5.45016855865153          | 5.254888888888889 | -0.19527966976264174 | -0.2022355324678227 | negative anomaly |
| BG           | 1       | 4.553999999999999         | 6.224             | 1.6700000000000008   | 3.0332047320749225  | positive anomaly |
| HR           | 2       | 4.360723484848485         | 4.9315            | 0.5707765151515154   | 1.3283131515045017  | positive anomaly |
| HU           | 4       | 4.296121527777777         | 5.59325           | 1.2971284722222225   | 1.4083954218496046  | positive anomaly |
| RO           | 30      | 3.377112173380455         | 4.669366666666666 | 1.2922544932862117   | 1.5439054093505944  | positive anomaly |

## 6. Wheat Temporal Walk-Forward Yearly Breakdown

| year   | rmse               | r2                   | n    |
| ------ | ------------------ | -------------------- | ---- |
| 2006.0 | 0.6699223239077792 | 0.5736386229481547   | 40.0 |
| 2007.0 | 0.7489694645891493 | 0.30672008359057057  | 23.0 |
| 2008.0 | 0.8688625371086667 | 0.2611410078232542   | 44.0 |
| 2009.0 | 0.8271542190667951 | 0.5034002722884303   | 50.0 |
| 2010.0 | 0.8129159579573737 | 0.18670580106579793  | 57.0 |
| 2011.0 | 0.9007373643571644 | 0.014801173980233062 | 55.0 |
| 2012.0 | 0.640412132215567  | 0.6626422174619482   | 54.0 |
| 2013.0 | 0.6279063410446961 | 0.5498058148228188   | 58.0 |
| 2014.0 | 0.7783906242769528 | 0.47189330144749597  | 53.0 |
| 2015.0 | 0.8194625363643919 | 0.4461695974517005   | 42.0 |
| 2016.0 | 0.8386147878994158 | 0.46142844862867805  | 67.0 |
| 2017.0 | 1.173164011857109  | -0.7968108975971762  | 48.0 |
| 2018.0 | 0.8709676735841414 | -0.2672328698368933  | 70.0 |
| 2019.0 | 0.8338232670506385 | 0.02897154977593619  | 64.0 |
| 2020.0 | 0.9965097690658882 | 0.5688440425572465   | 64.0 |

## 7. Dataset Descriptive Statistics

| Dataset                 | Regions | YearRange | ValidRecords | PredictorCount | Predictors                                                                                                                                   | TargetMean | TargetSD | TargetMin | TargetMax |
| ----------------------- | ------- | --------- | ------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | --------- | --------- |
| CY-Bench Maize (Europe) | 79      | 2001-2020 | 1543         | 15             | prec_sum, tmin_mean, tmax_mean, tavg_mean, rad_mean, et0_mean, vpd_mean, cwb_mean, ndvi_mean, ndvi_max, fpar_mean, fpar_max, awc, bulk_de... | 5.327      | 2.434    | 0.233     | 12.966    |
| CY-Bench Wheat (Europe) | 76      | 2001-2020 | 951          | 15             | prec_sum, tmin_mean, tmax_mean, tavg_mean, rad_mean, et0_mean, vpd_mean, cwb_mean, ndvi_mean, ndvi_max, fpar_mean, fpar_max, awc, bulk_de... | 4.143      | 1.198    | 0.403     | 8.14      |
| CY-Bench Maize (Zambia) | 71      | 2001-2017 | 1207         | 15             | awc, bulk_density, drainage_class, prec_sum, tmin_mean, tmax_mean, tavg_mean, rad_mean, et0_mean, vpd_mean, cwb_mean, ndvi_mean, ndvi_max... | 1.787      | 0.934    | 0.009     | 7.919     |
| SustainBench Soybean    | 538     | 2005-2016 | 5792         | 27             | band_0_mean, band_1_mean, band_2_mean, band_3_mean, band_4_mean, band_5_mean, band_6_mean, band_7_mean, band_8_mean, band_0_std, band_1_s... | 2.97       | 0.7      | 0.599     | 4.795     |

## 8. Existing Rolling Spatiotemporal Artifacts

- `rolling_st_CY-Bench_Maize_Europe.csv`
- `rolling_st_CY-Bench_Wheat_Europe.csv`
- `rolling_st_CY-Bench_Maize_Zambia.csv`
- `rolling_st_SustainBench_Soybean.csv`
- `figures/figure_rolling_spatiotemporal.png`

## 9. Outstanding / Not Verified

- Could not find a separate local script or artifact that unambiguously defines the single-holdout pilot table. Not regenerated here.
- The rolling spatiotemporal figure/data were already present locally and are referenced as existing artifacts.
- Outstanding citations still needed: (a) external confirmation of 2018 Romania/Hungary above-average maize yield, (b) Kapoor & Narayanan 2023 Patterns full citation, (c) the 2026 '2,047 Benchmark Datasets' leakage landscape paper full citation.
