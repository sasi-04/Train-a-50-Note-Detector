# Dataset quality audit

Dataset root: `dataset`  
Manifest hash: `None`  
Status: **NOT READY** (0 errors, 1 warnings)

## Blockers

- no images in the dataset
- no front-side images of the new Rs.50 note
- no back-side images of the new Rs.50 note
- no negative images of the Rs.10 note
- no negative images of the Rs.20 note
- no negative images of the Rs.100 note
- no negative images of the Rs.200 note
- no negative images of the Rs.500 note
- no ordinary-object / empty-background negatives
- fewer than 3 capture sessions (cannot form leak-free train/val/test)

## Totals

- Images in manifest: 0 (on disk: 0)
- Target front / back / unknown side: 0 / 0 / 0

## Images per denomination

| value | count |
|---|---|
| (none) | 0 |

## Images per split

| value | count |
|---|---|
| (none) | 0 |

## Split x denomination

| split | 50_new | 50_old | 10 | 20 | 100 | 200 | 500 | object | background | mixed | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Capture sessions

| value | count |
|---|---|
| (none) | 0 |

## Image source

| value | count |
|---|---|
| (none) | 0 |

## Lighting condition

| value | count |
|---|---|
| (none) | 0 |

## Resolution (long side, px)

| value | count |
|---|---|
| (none) | 0 |

## Boxes per image

| value | count |
|---|---|
| (none) | 0 |

## Integrity

- Near-duplicate pairs inside one split: 0
- Cross-split near-duplicate pairs (leakage): 0
- Very blurry images (Laplacian variance < 20): 0

## Errors (0)


## Warnings (1)

- unsupported file in images/: .gitkeep

_Generated from the files on disk. Recommended minimums are guidelines, not statistical guarantees._
