# False-positive stress test (val split, negatives only)

Operating threshold: **0.62**. Negative images tested: **159**.

| category | tested | false ₹50 at threshold | 95% CI | max false score | >=0.25 | >=0.40 | >=0.50 | >=0.60 | >=0.70 | >=0.80 | >=0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 27 | 0 | 0%-12% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100 | 31 | 0 | 0%-11% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | 19 | 0 | 0%-17% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 200 | 19 | 0 | 0%-17% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 500 | 27 | 0 | 0%-12% | 0.60 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| background | 36 | 0 | 0%-10% | 0.02 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Strongest false-₹50 score per image, quantiles: p50=0.00, p90=0.00, p95=0.00, p99=0.01, p100=0.60

Hardest negatives (annotated JPEGs in `reports\stress`):

- webcam_s02_500_unknown_0016 (500): 0.60 near-miss - review
- webcam_s02_background_na_0025 (background): 0.02
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
- scan_val_10_back_0004 (10): 0.00
- scan_val_10_back_0005 (10): 0.00
- scan_val_10_back_0006 (10): 0.00
- scan_val_10_front_0001 (10): 0.00
- scan_val_200_back_0001 (200): 0.00
- scan_val_500_back_0001 (500): 0.00

Small categories give wide confidence intervals; a 0% observed rate is not a guarantee.
