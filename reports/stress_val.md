# False-positive stress test (val split, negatives only)

Operating threshold: **0.54**. Negative images tested: **22**.

| category | tested | false ₹50 at threshold | 95% CI | max false score | >=0.25 | >=0.40 | >=0.50 | >=0.60 | >=0.70 | >=0.80 | >=0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 7 | 0 | 0%-35% | 0.07 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100 | 8 | 0 | 0%-32% | 0.45 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 200 | 1 | 0 | 0%-79% | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 500 | 5 | 0 | 0%-43% | 0.34 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 1 | 0 | 0%-79% | 0.03 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Strongest false-₹50 score per image, quantiles: p50=0.03, p90=0.22, p95=0.34, p99=0.43, p100=0.45

Hardest negatives (annotated JPEGs in `reports\stress`):

- scan_val_100_back_0004 (100): 0.45 near-miss - review
- scan_val_500_back_0004 (500): 0.34
- scan_val_100_back_0003 (100): 0.23
- scan_val_100_back_0001 (100): 0.16
- scan_val_100_back_0002 (100): 0.13
- scan_val_500_back_0003 (500): 0.09
- scan_val_500_back_0002 (500): 0.08
- scan_val_10_back_0002 (10): 0.07
- scan_val_10_back_0001 (10): 0.07
- scan_val_10_back_0005 (10): 0.04
- hand_b_face_background_na_0001 (background): 0.03
- scan_val_100_back_0007 (100): 0.03
- scan_val_10_back_0004 (10): 0.03
- scan_val_10_back_0003 (10): 0.03
- scan_val_10_front_0001 (10): 0.02
- scan_val_100_back_0006 (100): 0.02
- scan_val_10_back_0006 (10): 0.01
- scan_val_100_back_0005 (100): 0.01
- scan_val_100_back_0008 (100): 0.00
- scan_val_200_back_0001 (200): 0.00

Small categories give wide confidence intervals; a 0% observed rate is not a guarantee.
