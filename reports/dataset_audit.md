# Dataset quality audit

Dataset root: `dataset`  
Manifest hash: `7db1d0e102c3300e`  
Status: **READY FOR TRAINING** (0 errors, 5 warnings)

## Totals

- Images in manifest: 801 (on disk: 801)
- Target front / back / unknown side: 28 / 17 / 153

## Images per denomination

| value | count |
|---|---|
| 10 | 85 |
| 100 | 88 |
| 20 | 120 |
| 200 | 86 |
| 500 | 85 |
| 50_new | 198 |
| background | 139 |

## Images per split

| value | count |
|---|---|
| test | 259 |
| train | 354 |
| val | 188 |

## Split x denomination

| split | 50_new | 50_old | 10 | 20 | 100 | 200 | 500 | object | background | mixed | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 68 | 0 | 22 | 63 | 19 | 28 | 21 | 0 | 38 | 0 | 0 |
| train | 101 | 0 | 36 | 38 | 38 | 39 | 37 | 0 | 65 | 0 | 0 |
| val | 29 | 0 | 27 | 19 | 31 | 19 | 27 | 0 | 36 | 0 | 0 |

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
| webcam_s01 | 255 |
| webcam_s02 | 153 |
| webcam_s03 | 164 |
| webcam_s04 | 46 |
| webcam_s05 | 66 |

## Image source

| value | count |
|---|---|
| phone | 117 |
| webcam | 684 |

## Lighting condition

| value | count |
|---|---|
| unknown | 801 |

## Resolution (long side, px)

| value | count |
|---|---|
| <= 1600 | 801 |

## Boxes per image

| value | count |
|---|---|
| 0 | 139 |
| 1 | 662 |

## Integrity

- Near-duplicate pairs inside one split: 191
- Cross-split near-duplicate pairs (leakage): 0
- Very blurry images (Laplacian variance < 20): 5

## Errors (0)


## Warnings (5)

- unsupported file in images/: .gitkeep
- 191 near-duplicate pairs within the same split/session (expected for burst captures; consider thinning them)
- only 28 front-side target images (recommended >= 100)
- only 17 back-side target images (recommended >= 100)
- no images of the OLD Rs.50 note (not the target; recommended as a hard negative)

_Generated from the files on disk. Recommended minimums are guidelines, not statistical guarantees._
