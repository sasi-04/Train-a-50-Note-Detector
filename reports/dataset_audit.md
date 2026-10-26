# Dataset quality audit

Dataset root: `dataset`  
Manifest hash: `258f672f30b057e4`  
Status: **READY FOR TRAINING** (0 errors, 11 warnings)

## Totals

- Images in manifest: 117 (on disk: 117)
- Target front / back / unknown side: 28 / 17 / 6

## Images per denomination

| value | count |
|---|---|
| 10 | 12 |
| 100 | 15 |
| 20 | 12 |
| 200 | 12 |
| 500 | 10 |
| 50_new | 51 |
| background | 5 |

## Images per split

| value | count |
|---|---|
| test | 29 |
| train | 53 |
| val | 35 |

## Split x denomination

| split | 50_new | 50_old | 10 | 20 | 100 | 200 | 500 | object | background | mixed | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 14 | 0 | 1 | 8 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| train | 24 | 0 | 4 | 4 | 7 | 5 | 5 | 0 | 4 | 0 | 0 |
| val | 13 | 0 | 7 | 0 | 8 | 1 | 5 | 0 | 1 | 0 | 0 |

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

## Image source

| value | count |
|---|---|
| phone | 117 |

## Lighting condition

| value | count |
|---|---|
| unknown | 117 |

## Resolution (long side, px)

| value | count |
|---|---|
| <= 1600 | 117 |

## Boxes per image

| value | count |
|---|---|
| 0 | 66 |
| 1 | 51 |

## Integrity

- Near-duplicate pairs inside one split: 79
- Cross-split near-duplicate pairs (leakage): 0
- Very blurry images (Laplacian variance < 20): 4

## Errors (0)


## Warnings (11)

- unsupported file in images/: .gitkeep
- 79 near-duplicate pairs within the same split/session (expected for burst captures; consider thinning them)
- only 28 front-side target images (recommended >= 100)
- only 17 back-side target images (recommended >= 100)
- only 12 images of Rs.10 (recommended >= 30); per-denomination error rates will be unreliable
- only 12 images of Rs.20 (recommended >= 30); per-denomination error rates will be unreliable
- only 15 images of Rs.100 (recommended >= 30); per-denomination error rates will be unreliable
- only 12 images of Rs.200 (recommended >= 30); per-denomination error rates will be unreliable
- only 10 images of Rs.500 (recommended >= 30); per-denomination error rates will be unreliable
- only 5 object/background negatives (recommended >= 50)
- no images of the OLD Rs.50 note (not the target; recommended as a hard negative)

_Generated from the files on disk. Recommended minimums are guidelines, not statistical guarantees._
