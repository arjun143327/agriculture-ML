### 2. Feature Importance

**CY-Bench Wheat**
Feature                 Rand mean  Rand SD  Spat mean  Spat SD
------------------------------------------------------------
fpar_mean                 0.2474   0.0394     0.2445   0.0099
drainage_class            0.2245   0.0571     0.2338   0.0294
bulk_density              0.1248   0.0028     0.1582   0.0037
awc                       0.0621   0.0066     0.0572   0.0012
ndvi_mean                 0.0540   0.0033     0.0410   0.0034
fpar_max                  0.0516   0.0008     0.0479   0.0033
rad_mean                  0.0399   0.0056     0.0319   0.0050
vpd_mean                  0.0316   0.0061     0.0276   0.0018

**CY-Bench Zambia Maize**
Feature                 Rand mean  Rand SD  Spat mean  Spat SD
------------------------------------------------------------
tavg_mean                 0.2962   0.0323     0.2147   0.0285
tmax_mean                 0.1011   0.0279     0.1168   0.0220
bulk_density              0.0793   0.0056     0.1008   0.0069
cwb_mean                  0.0756   0.0095     0.0639   0.0015
vpd_mean                  0.0623   0.0012     0.1078   0.0158
ndvi_mean                 0.0593   0.0156     0.0494   0.0030
fpar_max                  0.0560   0.0056     0.0560   0.0028
awc                       0.0466   0.0026     0.0471   0.0013

### 3. Full Rolling-Window Spatiotemporal Data Table

| Dataset | Window (start-end year) | RF R² | Ridge R² | XGB R² | Anomaly year(s) in window? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CY-Bench Maize | 2006-2008 | 0.520 | 0.495 | 0.458 | No |
| CY-Bench Maize | 2007-2009 | 0.542 | 0.521 | 0.457 | No |
| CY-Bench Maize | 2008-2010 | 0.554 | 0.452 | 0.410 | No |
| CY-Bench Maize | 2009-2011 | 0.462 | 0.415 | 0.358 | No |
| CY-Bench Maize | 2010-2012 | 0.526 | 0.523 | 0.487 | No |
| CY-Bench Maize | 2011-2013 | 0.546 | 0.540 | 0.463 | No |
| CY-Bench Maize | 2012-2014 | 0.540 | 0.520 | 0.499 | No |
| CY-Bench Maize | 2013-2015 | 0.379 | 0.143 | 0.296 | No |
| CY-Bench Maize | 2014-2016 | 0.327 | 0.367 | 0.198 | No |
| CY-Bench Maize | 2015-2017 | 0.343 | 0.305 | 0.269 | No |
| CY-Bench Maize | 2016-2018 | 0.142 | 0.013 | -0.030 | 2018 |
| CY-Bench Maize | 2017-2019 | -0.515 | -0.816 | -0.693 | 2018 |
| CY-Bench Maize | 2018-2020 | -0.096 | -0.123 | -0.287 | 2018 |
| CY-Bench Wheat | 2006-2008 | 0.154 | 0.416 | 0.166 | No |
| CY-Bench Wheat | 2007-2009 | 0.093 | 0.342 | -0.072 | No |
| CY-Bench Wheat | 2008-2010 | 0.198 | 0.133 | 0.180 | No |
| CY-Bench Wheat | 2009-2011 | 0.182 | 0.166 | 0.137 | No |
| CY-Bench Wheat | 2010-2012 | 0.184 | 0.226 | 0.154 | No |
| CY-Bench Wheat | 2011-2013 | 0.178 | 0.340 | 0.072 | No |
| CY-Bench Wheat | 2012-2014 | 0.253 | 0.324 | 0.324 | No |
| CY-Bench Wheat | 2013-2015 | 0.273 | 0.365 | 0.266 | No |
| CY-Bench Wheat | 2014-2016 | 0.272 | 0.454 | 0.244 | No |
| CY-Bench Wheat | 2015-2017 | 0.172 | 0.085 | 0.055 | No |
| CY-Bench Wheat | 2016-2018 | -0.145 | -0.618 | -0.459 | 2018 |
| CY-Bench Wheat | 2017-2019 | -1.049 | -1.756 | -1.278 | 2018 |
| CY-Bench Wheat | 2018-2020 | 0.072 | -0.027 | -0.090 | 2018 |
| CY-Bench Zambia Maize | 2006-2008 | -0.065 | -0.005 | -0.262 | No |
| CY-Bench Zambia Maize | 2007-2009 | -0.290 | -0.224 | -0.634 | No |
| CY-Bench Zambia Maize | 2008-2010 | 0.004 | 0.131 | -0.246 | No |
| CY-Bench Zambia Maize | 2009-2011 | 0.246 | 0.265 | 0.135 | No |
| CY-Bench Zambia Maize | 2010-2012 | 0.063 | 0.051 | -0.073 | 2012 |
| CY-Bench Zambia Maize | 2011-2013 | 0.138 | 0.141 | 0.054 | 2012 |
| CY-Bench Zambia Maize | 2012-2014 | -0.092 | 0.031 | -0.161 | 2012, 2014 |
| CY-Bench Zambia Maize | 2013-2015 | 0.221 | 0.262 | 0.111 | 2014 |
| CY-Bench Zambia Maize | 2014-2016 | 0.262 | 0.200 | 0.255 | 2014 |
| CY-Bench Zambia Maize | 2015-2017 | 0.357 | 0.424 | 0.282 | No |
| SustainBench Soybean | 2010-2012 | 0.167 | 0.074 | 0.120 | 2012 |
| SustainBench Soybean | 2011-2013 | 0.193 | 0.105 | 0.118 | 2012 |
| SustainBench Soybean | 2012-2014 | 0.307 | 0.274 | 0.209 | 2012 |
| SustainBench Soybean | 2013-2015 | 0.175 | 0.208 | 0.110 | No |
| SustainBench Soybean | 2014-2016 | -0.324 | -0.326 | -0.335 | No |
