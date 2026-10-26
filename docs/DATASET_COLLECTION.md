# Dataset collection guide

The quality of this detector is decided here, not in the training script.

## Rules (from the assignment; not negotiable)

1. You photograph **genuine notes yourself, with your phone**. Optional webcam captures (`capture_dataset.py webcam`)
   are recorded as `source=webcam` and reported separately; the training set must be primarily phone photos.
2. Photograph the **new ₹50 note on both sides**.
3. Photograph **₹10, ₹20, ₹100, ₹200, ₹500** as negatives (both sides).
4. **Never** download banknote images, use generated/edited banknote images, print, photocopy or draw notes.
5. Do not label anything with a denomination you are unsure of. Use `unknown` for metadata you do not know.

## Capture protocol: sessions matter more than image count

A **session** is one sitting: one place, one lighting setup, one day (`s01`, `s02`, ...). The whole session goes to
exactly one of train/val/test, so near-identical frames can never leak across splits and the held-out test measures
generalisation to a *new* setting. Therefore:

* Do **at least 6 sessions** (more is better): different rooms/desks/tables, different times of day, different phones
  or distances if you can.
* **Every session must contain everything**: new ₹50 front + back, all five negative denominations, and some
  non-note objects/backgrounds. Otherwise a split ends up without negatives or without the target.
* Take 10–25 shots per (denomination, side) per session rather than 200 near-identical burst frames.

### Variation checklist for the new ₹50 (both sides)

| axis | cover |
|---|---|
| distance | note fills ~80 %, ~40 %, ~15 % of the frame |
| angle | flat-on, tilted ~30°, steep ~60°, rotated in-plane (portrait/landscape) |
| lighting | bright, normal, dim, uneven/side light, shadow across the note |
| background | plain table, cluttered desk, held in hand, on a patterned surface |
| state | flat, slightly curled/folded, partly covered by fingers (occlusion), slightly blurry |

### Negatives (what keeps false positives down)

* **₹10, ₹20, ₹100, ₹200, ₹500**: same variation checklist, front and back; ≥30 images each (more for the ones
  most similar in colour/size to the ₹50 in your hands).
* **Other things that look like a note**: ID cards, receipts, coloured paper, a phone screen, books, wallets,
  envelopes (`--denomination object`), and **empty backgrounds** where you test (`--denomination background`).
* **Old ₹50 note**, if you already own one (`--denomination 50_old`). Do not obtain one just for this.
* **A hand holding a ₹50 *and* another note** in the same frame: `--denomination mixed` (needs a box around the ₹50 only).

Recommended minimums (the audit warns below these; they are guidelines, not statistical guarantees):
≥100 front + ≥100 back target images, ≥30 per negative denomination, ≥50 object/background images, ≥6 sessions.

## Importing and labelling

```bash
python scripts/capture_dataset.py ingest <folder> --denomination 50_new --side front \
    --lighting dim --angle tilted --background cluttered --session s03
python scripts/annotate.py --only-unlabeled
```

* `denomination`: `50_new | 50_old | 10 | 20 | 100 | 200 | 500 | object | background | mixed | unknown`
* `side`: `front | back | na | unknown` · `lighting`: `bright | normal | dim | uneven | shadow | unknown`
* `angle`: `frontal | tilted | steep | unknown` · `background`: `plain | cluttered | hand_held | unknown`
* Import re-encodes to JPEG (long side ≤ 1600 px), which **removes EXIF including GPS**. Photos are still
  yours: look through them for faces, addresses or documents before publishing the repository.
* **Annotation:** drag a tight box around each ₹50 note (right-click removes). Nothing is auto-annotated. Negatives
  get an empty label file on import (a genuine "no ₹50 here"). A target image must have ≥ 1 box; a negative
  image must have none (the auditor enforces both).
* The metadata describes what *you* know; the tools never infer a denomination from a file name.

## Verify before training

```bash
python scripts/split_dataset.py --seed 42
python scripts/validate_dataset.py       # reports/dataset_audit.md
```

The audit lists blockers (missing side/negative denomination/session split), errors (bad labels, corrupt or duplicate
images, cross-split near-duplicates) and warnings (below recommended counts, all-webcam data, blur, missing old-₹50
hard negatives). Training is refused while blockers or errors exist.

## Publishing the data

`dataset/images` and `dataset/labels` are tracked by git by default so reviewers can retrain. If the data is too
large (> ~100 MB) attach a zip to a GitHub Release and say so in the README; keep `metadata/image_manifest.csv`
in the repository.
