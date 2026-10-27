# Evaluation methodology and results

**Read this first.** The data come from one person, one physical copy of each note and a few rooms. The numbers below are an
honest sample for this setup; they are *not* a promise about a reviewer's webcam, notes or lighting.

## Data actually used (738 images, `reports/dataset_audit.md`)

| source | images | what it is |
|---|---|---|
| scanner-app PDF #1 | 72 | one genuine note per denomination (₹10/20/50/100/200/500), tight crops, ~15 clusters of near-identical shots |
| scanner-app PDFs #2–#4 | 45 | hand-held frames of the new ₹50 (40) + 5 frames with no note |
| **webcam sessions 1–3** (`scripts/collect_webcam.py`) | 621 | frames from the *deployment webcam*: ₹50 front/back/sideways (105 hand-boxed), ₹10/20/100/200/500, room/people, blue-green things (window, clothes) |

Total ₹50 targets: 11 + 40 + 105 = 156 boxed images from **one physical ₹50 note**. Lighting/angle metadata is "unknown", so
"dim lighting" and "angled" scenarios are **NOT TESTED** as separate groups. Webcam frames show the developer and bystanders.
They are supplementary data from the same camera the demo uses; the first 117 images are the phone/scanner captures.

## Why webcam data were added

The first model (trained only on phone/scanner captures) failed live: a window scored 0.60–0.69 as "₹50" while a real note held
to the webcam scored 0.33. See the commit history for the screenshots-driven fixes (shape sanity filter, webcam collection).

## Protocol

* Whole sessions / near-duplicate clusters go to one split; `validate_dataset.py` (256-bit dHash) checks for cross-split leakage.
  - train 335 images · **val 200** (29 target) · **test 203** (36 target); webcam session 1 → train, **session 2 → val**,
    **session 3 → test**, plus the earlier scan splits (`dataset/metadata/*.yaml`).
* **Evaluation model** `rs50_split`: YOLOv8n, COCO-pretrained, trained 40 epochs on the train split only.
* **Threshold chosen on val only**: 0.46. Rule: among thresholds with 0 false-alarm negative images and recall ≥ 0.80, take the
  middle of those within 0.05 recall of the best. (First pick, 0.41, was a knife edge — 0.40 already had a false alarm — so the
  rule was changed *before* looking at the test split.)
* **Test split evaluated once** with the frozen threshold (`reports/eval_test.md`).
* **Shipped model** `rs50_final`: the evaluation model fine-tuned for 12 epochs on **all 738 images** (`train.py --final-fit --init`).
  It has seen val/test images, so **no held-out score exists for it**; threshold 0.46 is carried over.

## Held-out results (evaluation model, threshold 0.46, IoU 0.5)

| split | ₹50 images found | negative images labelled ₹50 |
|---|---|---|
| val (29 target / 171 negative) | 26/29 at 0.41; 25/29 at 0.5 (recall 86–90 %) | **0 / 171** (95 % CI upper 2.2 %) |
| **test** (36 target / 167 negative) | **23 / 36 (63.9 %)**; AP@0.5 = 0.92 | **0 / 167** (95 % CI upper 2.2 %) |

Test negatives, all 0 false ₹50: ₹10 0/23, ₹20 0/31, ₹100 0/23, ₹200 0/29, ₹500 0/23, no-note frames 0/38 (including the window
and blue-green clothing). Test targets by side: back 6/6, front 4/8, side-unknown 13/22. Session `webcam_s03` alone: 13/22 found,
0/152 false alarms.

Interpretation: the model almost never calls something else a ₹50 (its previous main problem), but it **misses roughly a third of
₹50 frames in a new, bright, backlit session**. AP 0.92 means the real notes mostly rank above everything else; the 0.46
threshold is conservative for that session (some misses score 0.3–0.45). The threshold was **not** retuned on test.
Misses are mostly motion-blurred, strongly backlit or partially hidden frames.

## Shipped model: what was measured

* ONNX export validated through the OpenCV path (`reports/onnx_validation.md`: PASS; PyTorch parity ≤ 0.001 px).
* **Live, no note shown, 200 webcam frames: 0 accepted ₹50** (`reports/webcam_smoke.json`), median 83 ms inference / 87 ms pipeline
  on this 2-core CPU (≈ 11 FPS).
* On its own training images: 55/55 targets, 0/280 false ₹50 (`reports/final_on_train/`): an optimistic upper bound only.
* A live *positive* test with real notes held in many poses was done by the developer by hand, not recorded as a metric.

## Definitions

* **Accepted detection:** target-class candidate that passes the shape check (not a hairline/speck; aspect ≤ 4.5; short side ≥ 5 % of
  the short image side), has score ≥ threshold, after NMS (IoU 0.45).
* **Detection level:** an accepted detection is **TP** if it matches an unmatched ground-truth box at IoU ≥ 0.5 (greedy by score);
  otherwise **FP**; unmatched ground-truth boxes are **FN**.
* **Image level:** a *negative image* is a **false-positive image** if it has ≥ 1 accepted detection. Negative-image FP rate = FP
  images ÷ negative images, always printed with its denominator and a 95 % Wilson interval (0 of 167 allows a true rate up to
  ≈ 2 %).
* **AP@0.5:** all-point interpolated over all scores. Scores are detector confidences, **not calibrated probabilities**.
* **Frames from one capture are strongly correlated** (consecutive webcam frames), so the effective sample is smaller than the
  image counts suggest.

## Temporal confirmation and second stage

Not credited in any metric: evaluation uses raw per-frame decisions. `--confirm-frames N` only affects the live display. No
second-stage verifier exists.
