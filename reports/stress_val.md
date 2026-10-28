# False-positive stress test (val split, negatives only)

Operating threshold: **0.27**. Negative images tested: **159**.

| category | tested | false ₹50 at threshold | 95% CI | max false score | >=0.25 | >=0.40 | >=0.50 | >=0.60 | >=0.70 | >=0.80 | >=0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 27 | 0 | 0%-12% | 0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100 | 31 | 0 | 0%-11% | 0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | 19 | 0 | 0%-17% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 200 | 19 | 0 | 0%-17% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 500 | 27 | 0 | 0%-12% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 36 | 0 | 0%-10% | 0.13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Strongest false-₹50 score per image, quantiles: p50=0.00, p90=0.00, p95=0.00, p99=0.03, p100=0.13

Hardest negatives (annotated JPEGs in `reports\split3_model\stress\stress`):

- webcam_s02_background_na_0022 (background): 0.13
- webcam_s02_background_na_0017 (background): 0.03
- webcam_s02_100_unknown_0015 (100): 0.02
- webcam_s02_10_unknown_0003 (10): 0.02
- webcam_s02_100_unknown_0012 (100): 0.01
- webcam_s02_background_na_0029 (background): 0.01
- webcam_s02_10_unknown_0023 (10): 0.01
- webcam_s02_100_unknown_0010 (100): 0.01
- hand_b_face_background_na_0001 (background): 0.00
- scan_val_100_back_0001 (100): 0.00
- scan_val_100_back_0002 (100): 0.00
- scan_val_100_back_0003 (100): 0.00
- scan_val_100_back_0004 (100): 0.00
- scan_val_100_back_0005 (100): 0.00
- scan_val_100_back_0006 (100): 0.00
- scan_val_100_back_0007 (100): 0.00
- scan_val_100_back_0008 (100): 0.00
- scan_val_10_back_0001 (10): 0.00
- scan_val_10_back_0002 (10): 0.00
- scan_val_10_back_0003 (10): 0.00

Small categories give wide confidence intervals; a 0% observed rate is not a guarantee.
