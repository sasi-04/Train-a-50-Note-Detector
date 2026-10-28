# Evaluation methodology and results

**Read this first.** The data come from one person, one physical copy of each note and a few rooms. These are honest numbers for
that setup, not a promise about a reviewer's webcam, notes or lighting.

## What the model does (two classes)

* class 0 **₹50** (the new note) → green box + `₹50 0.xx`, green banner "₹50 DETECTED".
* class 1 **other note** (₹10/₹20/₹100/₹200/₹500) → red box `Other note 0.xx`, red banner "OTHER NOTE - not a ₹50". It does **not** say
  which denomination.
* If both fire on one object, one box is kept (class-agnostic NMS); green wins in the app, so a ₹50 never gets a red box on top.

## Data actually used (689 images, `reports/dataset_audit.md`)

| source | images | what it is |
|---|---|---|
| scanner-app PDFs (phone captures) | 117 | one genuine note per denomination (tight crops, near-duplicate bursts) + hand-held ₹50 frames + 5 no-note frames |
| webcam sessions 1–3 (`scripts/collect_webcam.py`) | 572 | frames from the deployment webcam: ₹50 (front/back/sideways), ₹10/20/100/200/500 held in hand, no-note scenes (room, people, paper, phone, a window, blue-green clothing) |

Per class: **₹50 targets 120** (51 phone/scanner + 69 webcam), **other-note boxes 430**, **no-note images 139**. Lighting/angle
metadata is "unknown", so those scenarios are **NOT TESTED** as groups. Webcam frames show the developer and bystanders.
Boxes: 51 ₹50 boxes from colour proposals checked by eye; 69 ₹50 and 162 other-note boxes drawn by hand from grid views and
verified by overlay; 268 other-note boxes proposed by a helper model and **every one reviewed on contact sheets** (2 rejected,
49 frames with no usable box dropped).

## Protocol

* Whole sessions / near-duplicate clusters go to one split; `validate_dataset.py` (256-bit dHash) checks leakage.
  train 308 · **val 188** (29 ₹50) · **test 193** (36 ₹50). Webcam session 1 → train, 2 → val, 3 → test.
* **Evaluation model** `rs50_split2`: YOLOv8n, COCO start, 40 epochs on the train split only.
* **Thresholds chosen on val only** (rule: no false-alarm negative image and recall ≥ 0.80, take the middle of the plateau within
  0.05 recall of the best): **₹50 = 0.62**, **other note = 0.82**. The test split was evaluated **once** with them frozen.
* **Shipped model** `rs50_final2`: the evaluation model fine-tuned 12 epochs on **all 689 images**. It has seen val/test, so it has
  **no held-out score**; thresholds carried over.

## Held-out results (evaluation model, IoU 0.5)

What the app shows per frame (green ₹50 beats red), thresholds 0.62 / 0.82:

| held-out test (193 images) | frames | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|---|
| **new ₹50** | 36 | **33 (92 %)** | 0 | 3 |
| ₹10 | 22 | **0** | 14 | 8 |
| ₹20 | 29 | **0** | 23 | 6 |
| ₹100 | 19 | **0** | 11 | 8 |
| ₹200 | 28 | **0** | 23 | 5 |
| ₹500 | 21 | **0** | 13 | 8 |
| no note | 38 | **0** | 0 | 38 |

* **Non-₹50 images labelled ₹50: 0 of 157** (95 % CI upper bound 2.4 %). Every other-denomination image: 0 shown as ₹50.
* **₹50 found: 33 of 36 (92 %)**; detection-level (IoU ≥ 0.5): TP 32, FP 1, FN 4, precision 97 %, recall 89 %, AP@0.5 0.94.
  By side: front 8/8, back 5/6, side-unknown 19/22.
* **Other notes get a red box on 84 of 119 (71 %)**; the other 35 show nothing (mostly blurred or half-hidden notes). Class-1
  detection level: precision 100 %, recall 71 %, AP@0.5 0.92. A red box is never wrong-class on a ₹50 or on a no-note frame
  in this test.
* **Validation split** (used to choose thresholds): ₹50 27/29 found (93 %); 0/159 non-₹50 images labelled ₹50; other notes red on
  108/123 (88 %).
* Earlier single-class model on a comparable test: 23/36 ₹50 found (64 %), 0/167 false. The two-class model found more ₹50s; the
  likely reason is that other notes are now explicit training signal, but the runs also differ in data, so treat the comparison as
  indicative.

## Shipped model: what was measured

* ONNX export validated through the OpenCV path (`reports/onnx_validation.md`: PASS; PyTorch parity ≤ 0.001 px).
* **Live webcam, no note shown, 200 frames: 0 ₹50, 0 red boxes** (`reports/webcam_smoke.json`); median 76 ms inference / 80 ms
  pipeline per frame with the camera (≈ 12 FPS); the offline evaluation pipeline runs at ≈ 66 ms (15 FPS) on this 2-core CPU.
* On its own training images: 55/55 ₹50 found, 0/253 negatives labelled ₹50, other notes red on 88 %
  (`reports/final_on_train/`): an optimistic upper bound only.

## Definitions

* **Accepted ₹50 detection:** class-0 candidate with a plausible banknote shape (short side ≥ 5 % of the short image side, aspect
  ≤ 4.5), score ≥ threshold, after NMS (IoU 0.45). **Other-note box:** the same for class 1 with the other threshold.
* **Detection level:** TP if it matches an unmatched ground-truth box of that class at IoU ≥ 0.5 (greedy by score); otherwise FP;
  unmatched ground truth is FN.
* **Image level:** a *negative image* for ₹50 is any image without a ₹50 box (including other notes). It is a false-positive image
  if it has ≥ 1 accepted ₹50. Rates always carry their denominator and a 95 % Wilson interval.
* **AP@0.5:** all-point interpolated. Scores are detector confidences, **not calibrated probabilities**.
* Consecutive webcam frames are strongly correlated, so the effective sample is smaller than the image counts.

## Temporal confirmation and any second stage

Not credited in any metric: evaluation uses raw per-frame decisions. `--confirm-frames N` only affects the live display. No
second-stage verifier exists.
