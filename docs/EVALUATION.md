# Evaluation methodology and results

**Read this first.** The data come from one person, one physical copy of each note and a few rooms. These are honest numbers for
that setup, not a promise about a reviewer's webcam, notes or lighting.

## What the model does (two classes)

* class 0 **₹50** (the new note) → green box + `₹50 0.xx`, green banner "₹50 DETECTED".
* class 1 **other note** (₹10/₹20/₹100/₹200/₹500) → red box `Other note 0.xx`, red banner "OTHER NOTE - not a ₹50". It does **not** say
  which denomination.
* If both fire on one object, one box is kept (class-agnostic NMS); green wins in the app, so a ₹50 never gets a red box on top.

## Data actually used (801 images, `reports/dataset_audit.md`)

| source | images | what it is |
|---|---|---|
| scanner-app PDFs (phone captures) | 117 | one genuine note per denomination (tight crops, near-duplicate bursts) + hand-held ₹50 frames + 5 no-note frames |
| webcam sessions 1–5 (`scripts/collect_webcam.py`) | 684 | frames from the deployment webcam: ₹50 (front/back/sideways/**upright**), ₹10/20/100/200/500 held in hand (₹20 also upright), no-note scenes (room, people, paper, phone, a window, blue-green clothing) |

Per class: **₹50 targets 198** (51 phone/scanner + 147 webcam, of which 46 + 32 are the upright sessions 4 and 5), **other-note
boxes 464**, **no-note images 139**. Lighting/angle metadata is "unknown", so those scenarios are **NOT TESTED** as groups. Webcam
frames show the developer and bystanders. Boxes: 51 ₹50 boxes from colour proposals checked by eye; 147 ₹50 and 196 other-note
boxes drawn by hand from grid views and verified by overlay; 268 other-note boxes proposed by a helper model and **every one
reviewed on contact sheets** (2 rejected, 49 frames with no usable box dropped). In sessions 4 and 5, 65 + 8 frames with no usable
note (motion blur, edge-on, hidden by hands) were removed instead of boxed, which makes those two sessions slightly easier than the
raw recording.

## Protocol

* Whole sessions / near-duplicate clusters go to one split; `validate_dataset.py` (256-bit dHash) checks leakage.
  train 354 · **val 188** (29 ₹50) · **test 259** (68 ₹50). Webcam session 1 → train, 2 → val, 3 → test, 4 (upright) → train,
  5 (upright, recorded last) → test.
* **Evaluation model** `rs50_split3`: YOLOv8n, started from the earlier `rs50_split2` (itself COCO-start, 40 epochs on the train
  split) and fine-tuned 20 epochs on the train split only, with `--rot90`: a 90° rotated copy of every labelled-note *training*
  image is added (kept in the run folder, never from val/test).
* **Thresholds chosen on val only** (rule: no false-alarm negative image and recall ≥ 0.80, take the middle of the plateau within
  0.05 recall of the best): **₹50 = 0.27** (plateau 0.13–0.40), **other note = 0.37**. They are much lower than the earlier
  0.62 / 0.82 because this model scores more confidently. Validation is now saturated (29/29, 0 false), so it cannot tell a good
  threshold from a lucky one; the test set shows the ₹50 score does not matter between 0.13 and 0.43 (see below).
* **Shipped model** `rs50_final3`: the evaluation model fine-tuned 12 epochs on **all 801 images** (`--final-fit --rot90`). It
  has seen val and test, so it has **no held-out score**; thresholds carried over.

## Held-out results (evaluation model rs50_split3, IoU 0.5, test run with the thresholds frozen)

What the app shows per frame (green ₹50 beats red), thresholds 0.27 / 0.37:

| held-out test (259 images) | frames | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|---|
| **new ₹50** | 68 | **67 (99 %)** | 0 | 1 |
| ₹10 | 22 | **0** | 19 | 3 |
| ₹20 | 63 | **0** | 62 | 1 |
| ₹100 | 19 | **0** | 17 | 2 |
| ₹200 | 28 | **6 (false)** | 21 | 1 |
| ₹500 | 21 | **0** | 19 | 2 |
| no note | 38 | **2 (false)** | 0 | 36 |

* **Upright ₹50 (webcam session 5, recorded after training, never used for any choice): 32 of 32 found, 0 of 34 upright ₹20
  frames shown as ₹50.** The previous model (`rs50_split2`, thresholds 0.62 / 0.82) found **4 of 32** on the same frames
  (`reports/split3_model/baseline_split2_same_test/`).
* **₹50 found at detection level (IoU ≥ 0.5): TP 63, FN 5 of 68** (recall 92.6 %), precision 84 %, AP@0.5 0.91. The app *shows* a
  green box on 67 of 68 images, but on 4 of those the box overlaps the note by less than IoU 0.5 (hand/blur). By side: front
  8/8, back 3/6 (too few to say more), side-unknown 52/54. By session: hand_b_close 9/12, webcam_s03 20/22, webcam_s05 32/32.
* **Non-₹50 images labelled ₹50: 8 of 191 (4.2 %, 95 % CI upper bound 8.0 %)**, up from 0 of 191 for the previous model on
  this test set. This is the cost of the change, and it is real:
  * **6 are ₹200 close-up scan crops (`scan_test_crop`)** in which the note fills the whole image; the model draws an
    almost image-sized box at 0.73–0.85. A webcam frame rarely looks like this, but nothing in the model prevents it.
  * **2 are webcam frames with no note** (`webcam_s03_background_na_0016/0017`: a box over the pale ceiling, scored 0.65 and
    0.67; I viewed one of the two). This is the failure that would show up live.
  * The false scores are high (0.65–0.85), so no threshold between 0.13 and 0.43 changes the count (test sweep,
    `reports/threshold_sweep_test.csv`). I did **not** retune the threshold on the test set.
* **Other notes get a red box on 138 of 153 (90 %)** at detection level (precision 99 %, AP@0.5 0.93); I did not
  review the 15 misses one by one this time.
* **Validation split** (used to choose thresholds): ₹50 29/29 found; 0/159 non-₹50 images labelled ₹50; other notes red on 119/123
  (96.7 %); highest false ₹50 score on a val negative 0.13 (`reports/stress_val.md`).
* The test split was first used for `rs50_split2` (193 frames, then). It was used again for `rs50_split3` after session 5 was
  added. Neither model was tuned on it, but it is no longer a one-shot set.

## Shipped model: what was measured

* ONNX export validated through the OpenCV path (`reports/onnx_validation.md`: PASS; PyTorch parity ≤ 0.002 px).
* **Live webcam, no note shown, 200 frames: 0 ₹50, 0 red boxes** (`reports/webcam_demo_run.json`); median 64 ms inference /
  68 ms pipeline per frame with the camera (≈ 15 FPS) on this 2-core CPU. This was one short run in one room.
* On its own training split (354 images): 101/101 ₹50 found, 0/253 negatives labelled ₹50
  (`reports/final_on_train/`): an optimistic upper bound only. The 8 false alarms above are in its training data as negatives, so
  fixing them is not evidence of generalisation.
* **The shipped model's behaviour on upright ₹50 has not been measured on unseen frames**, and neither have its false alarms on
  frames like the ceiling above. Both are expected to be close to the evaluation model's numbers, not equal to them.

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
