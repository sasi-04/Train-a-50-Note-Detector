# Dataset quality audit

Dataset root: `dataset`  
Manifest hash: `c61ece36d59f192b`  
Status: **READY FOR TRAINING** (0 errors, 5 warnings)

## Totals

- Images in manifest: 738 (on disk: 738)
- Target front / back / unknown side: 28 / 17 / 75

## Images per denomination

| value | count |
|---|---|
| 10 | 95 |
| 100 | 99 |
| 20 | 96 |
| 200 | 95 |
| 500 | 94 |
| 50_new | 120 |
| background | 139 |

## Images per split

| value | count |
|---|---|
| test | 203 |
| train | 335 |
| val | 200 |

## Split x denomination

| split | 50_new | 50_old | 10 | 20 | 100 | 200 | 500 | object | background | mixed | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 36 | 0 | 23 | 31 | 23 | 29 | 23 | 0 | 38 | 0 | 0 |
| train | 55 | 0 | 42 | 42 | 45 | 43 | 43 | 0 | 65 | 0 | 0 |
| val | 29 | 0 | 30 | 23 | 31 | 23 | 28 | 0 | 36 | 0 | 0 |

## Capture sessions

| value | count |
|---|---|
| hand_a | 20 |
| hand_b_close | 12 |
| hand_b_face | 13 |
| scan_test_blue | 6 |
| scan_test_crop | 11 |
| scan_train | 33 |
| scan_val | 22 |
| webcam_s01 | 282 |
| webcam_s02 | 165 |
| webcam_s03 | 174 |

## Image source

| value | count |
|---|---|
| phone | 117 |
| webcam | 621 |

## Lighting condition

| value | count |
|---|---|
| unknown | 738 |

## Resolution (long side, px)

| value | count |
|---|---|
| <= 1600 | 738 |

## Boxes per image

| value | count |
|---|---|
| 0 | 618 |
| 1 | 120 |

## Integrity

- Near-duplicate pairs inside one split: 156
- Cross-split near-duplicate pairs (leakage): 0
- Very blurry images (Laplacian variance < 20): 5

## Errors (0)


## Warnings (5)

- unsupported file in images/: .gitkeep
- 156 near-duplicate pairs within the same split/session (expected for burst captures; consider thinning them)
- only 28 front-side target images (recommended >= 100)
- only 17 back-side target images (recommended >= 100)
- no images of the OLD Rs.50 note (not the target; recommended as a hard negative)

_Generated from the files on disk. Recommended minimums are guidelines, not statistical guarantees._
