# Significance Testing: Prompt Conditions (normal / missing / wrong)

Per-example scores are paired within each dataset/subset (the same queries are reused across normal/missing/wrong), then pooled across subsets within each dataset group before testing. p-values are Holm-Bonferroni corrected within each group across all metric x comparison tests of that type (continuous or binary). `*` p<.05, `**` p<.01, `***` p<.001 (corrected).


## Open-Ended

### Continuous metrics (paired t-test / Wilcoxon signed-rank)

| Metric | Comparison | n | Mean A | Mean B | Mean diff | Cohen's d_z | t p (Holm) | Wilcoxon p (Holm) | Rank-biserial r |
|---|---|---|---|---|---|---|---|---|---|
| Factscore (precision) | normal vs missing | 3661 | 0.8359 | 0.8356 | +0.0018 | +0.027 | 3.08e-01 | 4.44e-02* | +0.058 |
| Factscore (precision) | normal vs wrong | 3661 | 0.8359 | 0.7879 | +0.0428 | +0.773 | 0.00e+00*** | 0.00e+00*** | +0.824 |
| Factscore (precision) | missing vs wrong | 3661 | 0.8356 | 0.7879 | +0.0410 | +0.550 | 2.91e-211*** | 2.71e-275*** | +0.686 |
| Vital Precision | normal vs missing | 3726 | 0.8269 | 0.8288 | -0.0009 | -0.007 | 1.00e+00 | 1.00e+00 | -0.019 |
| Vital Precision | normal vs wrong | 3726 | 0.8269 | 0.7540 | +0.0531 | +0.381 | 3.15e-111*** | 3.73e-142*** | +0.597 |
| Vital Precision | missing vs wrong | 3726 | 0.8288 | 0.7540 | +0.0540 | +0.333 | 2.62e-86*** | 5.54e-115*** | +0.533 |
| Linear-Decay Precision | normal vs missing | 3726 | 0.8359 | 0.8381 | +0.0005 | +0.004 | 1.00e+00 | 1.00e+00 | +0.010 |
| Linear-Decay Precision | normal vs wrong | 3726 | 0.8359 | 0.7321 | +0.0831 | +0.521 | 8.66e-196*** | 2.23e-298*** | +0.718 |
| Linear-Decay Precision | missing vs wrong | 3726 | 0.8381 | 0.7321 | +0.0826 | +0.483 | 1.30e-170*** | 6.93e-263*** | +0.674 |
| Nuggets Recall (strict-all) | normal vs missing | 3726 | 0.2435 | 0.1861 | +0.0537 | +0.795 | 0.00e+00*** | 0.00e+00*** | +0.922 |
| Nuggets Recall (strict-all) | normal vs wrong | 3726 | 0.2435 | 0.2323 | +0.0125 | +0.318 | 1.57e-79*** | 1.50e-84*** | +0.529 |
| Nuggets Recall (strict-all) | missing vs wrong | 3726 | 0.1861 | 0.2323 | -0.0412 | -0.591 | 7.28e-244*** | 1.62e-262*** | -0.799 |
| Vital Recall | normal vs missing | 3726 | 0.4090 | 0.2991 | +0.1169 | +0.878 | 0.00e+00*** | 0.00e+00*** | +0.941 |
| Vital Recall | normal vs wrong | 3726 | 0.4090 | 0.3789 | +0.0350 | +0.397 | 1.23e-119*** | 5.54e-115*** | +0.663 |
| Vital Recall | missing vs wrong | 3726 | 0.2991 | 0.3789 | -0.0820 | -0.589 | 6.23e-242*** | 1.98e-222*** | -0.762 |
| Linear-Decay Recall | normal vs missing | 3726 | 0.7103 | 0.6106 | +0.1338 | +0.748 | 0.00e+00*** | 0.00e+00*** | +0.869 |
| Linear-Decay Recall | normal vs wrong | 3726 | 0.7103 | 0.6643 | +0.0624 | +0.452 | 1.19e-151*** | 2.88e-139*** | +0.609 |
| Linear-Decay Recall | missing vs wrong | 3726 | 0.6106 | 0.6643 | -0.0714 | -0.342 | 6.22e-91*** | 1.25e-102*** | -0.450 |

### Binary error metrics (McNemar's test)

| Metric | Comparison | n | Rate A | Rate B | Risk diff | Discordant (1,0)/(0,1) | McNemar p (Holm) |
|---|---|---|---|---|---|---|---|
| Any vital subclaim wrong | normal vs missing | 3726 | 0.5270 | 0.4548 | +0.0722 | 493/270 | 3.68e-15*** |
| Any vital subclaim wrong | normal vs wrong | 3726 | 0.5270 | 0.7379 | -0.2109 | 165/1097 | 1.10e-150*** |
| Any vital subclaim wrong | missing vs wrong | 3726 | 0.4548 | 0.7379 | -0.2831 | 170/1325 | 5.95e-195*** |
| Any vital nugget unsupported | normal vs missing | 3726 | 0.8722 | 0.9118 | -0.0396 | 21/111 | 2.83e-14*** |
| Any vital nugget unsupported | normal vs wrong | 3726 | 0.8722 | 0.8858 | -0.0136 | 24/60 | 1.34e-04*** |
| Any vital nugget unsupported | missing vs wrong | 3726 | 0.9118 | 0.8858 | +0.0260 | 93/39 | 7.94e-06*** |


## Single-Answer

### Continuous metrics (paired t-test / Wilcoxon signed-rank)

| Metric | Comparison | n | Mean A | Mean B | Mean diff | Cohen's d_z | t p (Holm) | Wilcoxon p (Holm) | Rank-biserial r |
|---|---|---|---|---|---|---|---|---|---|
| Factscore (precision) | normal vs missing | 2717 | 0.8247 | 0.8253 | -0.0001 | -0.002 | 9.37e-01 | 1.00e+00 | -0.016 |
| Factscore (precision) | normal vs wrong | 2717 | 0.8247 | 0.7657 | +0.0590 | +0.749 | 5.59e-264*** | 6.55e-239*** | +0.755 |
| Factscore (precision) | missing vs wrong | 2717 | 0.8253 | 0.7657 | +0.0592 | +0.565 | 1.32e-164*** | 8.32e-182*** | +0.648 |
| Vital Precision | normal vs missing | 3000 | 0.7281 | 0.7222 | +0.0059 | +0.020 | 5.56e-01 | 1.00e+00 | +0.006 |
| Vital Precision | normal vs wrong | 3000 | 0.7281 | 0.4873 | +0.2408 | +0.786 | 7.89e-315*** | 2.78e-263*** | +0.849 |
| Vital Precision | missing vs wrong | 3000 | 0.7222 | 0.4873 | +0.2349 | +0.697 | 6.05e-259*** | 9.84e-222*** | +0.769 |
| Linear-Decay Precision | normal vs missing | 3000 | 0.7299 | 0.7414 | -0.0116 | -0.056 | 9.03e-03** | 1.48e-01 | -0.053 |
| Linear-Decay Precision | normal vs wrong | 3000 | 0.7299 | 0.5183 | +0.2116 | +0.865 | 0.00e+00*** | 2.08e-293*** | +0.823 |
| Linear-Decay Precision | missing vs wrong | 3000 | 0.7414 | 0.5183 | +0.2231 | +0.879 | 0.00e+00*** | 9.78e-295*** | +0.825 |
| Nuggets Recall (strict-all) | normal vs missing | 3000 | 0.2771 | 0.1913 | +0.0858 | +0.537 | 3.91e-166*** | 3.99e-210*** | +0.880 |
| Nuggets Recall (strict-all) | normal vs wrong | 3000 | 0.2771 | 0.2349 | +0.0421 | +0.352 | 2.83e-77*** | 3.31e-100*** | +0.685 |
| Nuggets Recall (strict-all) | missing vs wrong | 3000 | 0.1913 | 0.2349 | -0.0436 | -0.303 | 1.23e-58*** | 6.43e-74*** | -0.551 |
| Vital Recall | normal vs missing | 3000 | 0.5244 | 0.2745 | +0.2499 | +0.711 | 9.49e-268*** | 1.40e-220*** | +0.942 |
| Vital Recall | normal vs wrong | 3000 | 0.5244 | 0.3670 | +0.1574 | +0.505 | 2.02e-149*** | 1.14e-137*** | +0.877 |
| Vital Recall | missing vs wrong | 3000 | 0.2745 | 0.3670 | -0.0925 | -0.284 | 5.99e-52*** | 1.68e-51*** | -0.518 |
| Linear-Decay Recall | normal vs missing | 3000 | 0.6256 | 0.5388 | +0.0868 | +0.388 | 1.99e-92*** | 9.69e-116*** | +0.683 |
| Linear-Decay Recall | normal vs wrong | 3000 | 0.6256 | 0.5313 | +0.0943 | +0.427 | 2.49e-110*** | 5.84e-113*** | +0.721 |
| Linear-Decay Recall | missing vs wrong | 3000 | 0.5388 | 0.5313 | +0.0075 | +0.029 | 3.39e-01 | 1.00e+00 | +0.002 |

### Binary error metrics (McNemar's test)

| Metric | Comparison | n | Rate A | Rate B | Risk diff | Discordant (1,0)/(0,1) | McNemar p (Holm) |
|---|---|---|---|---|---|---|---|
| Any vital subclaim wrong | normal vs missing | 3000 | 0.4503 | 0.4217 | +0.0287 | 388/302 | 2.43e-03** |
| Any vital subclaim wrong | normal vs wrong | 3000 | 0.4503 | 0.8953 | -0.4450 | 59/1394 | 1.28e-267*** |
| Any vital subclaim wrong | missing vs wrong | 3000 | 0.4217 | 0.8953 | -0.4737 | 57/1478 | 7.46e-287*** |
| Any vital nugget unsupported | normal vs missing | 3000 | 0.5430 | 0.6810 | -0.1380 | 54/468 | 1.46e-72*** |
| Any vital nugget unsupported | normal vs wrong | 3000 | 0.5430 | 0.6843 | -0.1413 | 42/466 | 5.56e-78*** |
| Any vital nugget unsupported | missing vs wrong | 3000 | 0.6810 | 0.6843 | -0.0033 | 258/268 | 6.95e-01 |


## All

### Continuous metrics (paired t-test / Wilcoxon signed-rank)

| Metric | Comparison | n | Mean A | Mean B | Mean diff | Cohen's d_z | t p (Holm) | Wilcoxon p (Holm) | Rank-biserial r |
|---|---|---|---|---|---|---|---|---|---|
| Factscore (precision) | normal vs missing | 6378 | 0.8324 | 0.8324 | +0.0010 | +0.012 | 6.80e-01 | 8.88e-01 | +0.018 |
| Factscore (precision) | normal vs wrong | 6378 | 0.8324 | 0.7808 | +0.0497 | +0.744 | 0.00e+00*** | 0.00e+00*** | +0.783 |
| Factscore (precision) | missing vs wrong | 6378 | 0.8324 | 0.7808 | +0.0487 | +0.547 | 0.00e+00*** | 0.00e+00*** | +0.660 |
| Vital Precision | normal vs missing | 6726 | 0.7957 | 0.7952 | +0.0021 | +0.010 | 6.80e-01 | 8.88e-01 | -0.012 |
| Vital Precision | normal vs wrong | 6726 | 0.7957 | 0.6698 | +0.1368 | +0.553 | 0.00e+00*** | 0.00e+00*** | +0.741 |
| Vital Precision | missing vs wrong | 6726 | 0.7952 | 0.6698 | +0.1347 | +0.497 | 0.00e+00*** | 0.00e+00*** | +0.673 |
| Linear-Decay Precision | normal vs missing | 6726 | 0.8024 | 0.8076 | -0.0049 | -0.030 | 4.44e-02* | 8.88e-01 | -0.018 |
| Linear-Decay Precision | normal vs wrong | 6726 | 0.8024 | 0.6646 | +0.1404 | +0.663 | 0.00e+00*** | 0.00e+00*** | +0.757 |
| Linear-Decay Precision | missing vs wrong | 6726 | 0.8076 | 0.6646 | +0.1453 | +0.651 | 0.00e+00*** | 0.00e+00*** | +0.739 |
| Nuggets Recall (strict-all) | normal vs missing | 6726 | 0.2541 | 0.1878 | +0.0680 | +0.571 | 0.00e+00*** | 0.00e+00*** | +0.898 |
| Nuggets Recall (strict-all) | normal vs wrong | 6726 | 0.2541 | 0.2332 | +0.0257 | +0.298 | 1.44e-125*** | 6.87e-184*** | +0.601 |
| Nuggets Recall (strict-all) | missing vs wrong | 6726 | 0.1878 | 0.2332 | -0.0423 | -0.387 | 1.17e-205*** | 2.55e-299*** | -0.678 |
| Vital Recall | normal vs missing | 6726 | 0.4454 | 0.2913 | +0.1762 | +0.670 | 0.00e+00*** | 0.00e+00*** | +0.933 |
| Vital Recall | normal vs wrong | 6726 | 0.4454 | 0.3752 | +0.0896 | +0.396 | 7.53e-214*** | 7.56e-263*** | +0.776 |
| Vital Recall | missing vs wrong | 6726 | 0.2913 | 0.3752 | -0.0867 | -0.360 | 1.91e-179*** | 1.68e-236*** | -0.644 |
| Linear-Decay Recall | normal vs missing | 6726 | 0.6836 | 0.5879 | +0.1128 | +0.560 | 0.00e+00*** | 0.00e+00*** | +0.803 |
| Linear-Decay Recall | normal vs wrong | 6726 | 0.6836 | 0.6223 | +0.0766 | +0.425 | 1.46e-243*** | 1.19e-251*** | +0.653 |
| Linear-Decay Recall | missing vs wrong | 6726 | 0.5879 | 0.6223 | -0.0362 | -0.154 | 1.23e-35*** | 5.38e-61*** | -0.279 |

### Binary error metrics (McNemar's test)

| Metric | Comparison | n | Rate A | Rate B | Risk diff | Discordant (1,0)/(0,1) | McNemar p (Holm) |
|---|---|---|---|---|---|---|---|
| Any vital subclaim wrong | normal vs missing | 6726 | 0.5028 | 0.4443 | +0.0585 | 881/572 | 1.29e-15*** |
| Any vital subclaim wrong | normal vs wrong | 6726 | 0.5028 | 0.7876 | -0.2848 | 224/2491 | 0.00e+00*** |
| Any vital subclaim wrong | missing vs wrong | 6726 | 0.4443 | 0.7876 | -0.3433 | 227/2803 | 0.00e+00*** |
| Any vital nugget unsupported | normal vs missing | 6726 | 0.7682 | 0.8389 | -0.0706 | 75/579 | 1.59e-85*** |
| Any vital nugget unsupported | normal vs wrong | 6726 | 0.7682 | 0.8221 | -0.0539 | 66/526 | 6.67e-79*** |
| Any vital nugget unsupported | missing vs wrong | 6726 | 0.8389 | 0.8221 | +0.0167 | 351/307 | 9.37e-02 |
