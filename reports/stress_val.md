# False-positive stress test (val split, negatives only)

Operating threshold: **0.46**. Negative images tested: **171**.

| category | tested | false ₹50 at threshold | 95% CI | max false score | >=0.25 | >=0.40 | >=0.50 | >=0.60 | >=0.70 | >=0.80 | >=0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 30 | 0 | 0%-11% | 0.31 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100 | 31 | 0 | 0%-11% | 0.41 | 7 | 1 | 0 | 0 | 0 | 0 | 0 |
| 20 | 23 | 0 | 0%-14% | 0.28 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 200 | 23 | 0 | 0%-14% | 0.14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 500 | 28 | 0 | 0%-12% | 0.26 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 36 | 0 | 0%-10% | 0.37 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |

Strongest false-₹50 score per image, quantiles: p50=0.10, p90=0.26, p95=0.32, p99=0.39, p100=0.41

Hardest negatives (annotated JPEGs in `reports\stress`):

- webcam_s02_100_unknown_0011 (100): 0.41 near-miss - review
- webcam_s02_100_unknown_0016 (100): 0.39 near-miss - review
- webcam_s02_100_unknown_0015 (100): 0.38 near-miss - review
- webcam_s02_background_na_0008 (background): 0.37
- webcam_s02_100_unknown_0012 (100): 0.36
- webcam_s02_100_unknown_0010 (100): 0.36
- webcam_s02_100_unknown_0004 (100): 0.35
- webcam_s02_background_na_0009 (background): 0.35
- webcam_s02_background_na_0027 (background): 0.34
- webcam_s02_10_unknown_0006 (10): 0.31
- webcam_s02_100_unknown_0021 (100): 0.31
- webcam_s02_10_unknown_0012 (10): 0.30
- webcam_s02_background_na_0007 (background): 0.29
- webcam_s02_background_na_0017 (background): 0.28
- webcam_s02_20_unknown_0007 (20): 0.28
- webcam_s02_20_unknown_0021 (20): 0.27
- webcam_s02_10_unknown_0015 (10): 0.26
- webcam_s02_500_unknown_0019 (500): 0.26
- webcam_s02_background_na_0024 (background): 0.26
- webcam_s02_10_unknown_0022 (10): 0.26

Small categories give wide confidence intervals; a 0% observed rate is not a guarantee.
