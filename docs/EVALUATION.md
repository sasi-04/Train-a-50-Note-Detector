# Evaluation methodology and results

**Read this first:** the dataset is tiny and unusually structured (details below). The numbers are an honest
*stress sample*, not a statistically reliable estimate of how the model behaves on a reviewer's webcam.

## Data actually used (117 images, `reports/dataset_audit.md`)

| source | images | what it is |
|---|---|---|
| scanner-app PDF #1 | 72 | one genuine note per denomination (₹10/20/50/100/200/500), tight crops, ≈15 clusters of near-identical shots (6 of them on a blue table) |
| scanner-app PDFs #2–#4 | 45 | hand-held frames of the new ₹50 (40 images: face-to-camera, close-ups, front and back) + 5 frames with no note |

Only **11 + 40 = 51 target images** exist, from one physical ₹50 note. Lighting/angle metadata is "unknown" for all of them, so
"dim lighting" and "angled" scenarios are **NOT TESTED**. There are no photos of other denominations held in hand, and only
5 non-note frames, all of one person/room.

## Protocol

* Whole near-duplicate clusters / capture sessions are assigned to one split (`dataset/metadata/*.yaml` documents each choice);
  `validate_dataset.py` found and forced the removal of one real cross-split leak (hand-held frames A098/A099 vs B003…).
  - train: 45 images · val: 35 (13 target) · test: 29 (14 target, 15 negative) *for the evaluation model*.
* Evaluation model `rs50_split` = YOLOv8n trained only on the train split. **Threshold chosen on val only**:
  0.54 (rule 1, FP-free and 100 % recall plateau 0.45–0.62; it is a narrow plateau).
* Test split evaluated **once** with that frozen threshold (`reports/eval_test.md`, `reports/stress_val.md`).
* **Shipped model `rs50_final`** = same recipe trained on *all* 117 images (`train.py --final-fit`). Because it has seen the
  val/test images, **no held-out metric exists for it**; the threshold 0.54 was carried over from the evaluation model.

## Results of the evaluation model (held-out)

Definitions are in the section below. IoU 0.5, threshold 0.54, CPU i5-7300U, OpenCV 5.0.0.

| split | target images found | missed | negative images with a false ₹50 | detection-level FP |
|---|---|---|---|---|
| val (selection split) | 13 / 13 | 0 | 0 / 22 (95 % CI upper 14.9 %) | 0 |
| **test (final)** | **7 / 14** (50 %) | 7 | **1 / 15** (6.7 %, CI upper 29.8 %) | 8 |

Test, by side: **back 6 / 6 found**, **front 1 / 8 found** (the 7 misses are dark, rotated, frame-filling front close-ups of
session `hand_b_close`, a view the evaluation model never saw in training). Test negatives: ₹10 0/1, ₹20 **1/8** (a ₹20 on the
blue table scored above 0.54), ₹200 0/6; ₹100, ₹500, objects/backgrounds **NOT TESTED on the test split** (the held-out
sides of ₹100/₹500 are in val: 0/8 and 0/5, max false score 0.45 and 0.34).

Interpretation: front-side recall on an unseen *kind of view* is poor, and one ₹20 was mistaken for a ₹50. Both results
point to the same fix: more varied photos (lighting, tilt, distance, other denominations in hand, cluttered scenes). The shipped
model has seen those views, which is why its quality on *new* conditions is unknown.

## Shipped model: what was measured

* ONNX export validated through the OpenCV path (`reports/onnx_validation.md`, PASS; PyTorch parity ≤ 0.001 px).
* Live webcam smoke test results: see `docs/ACCEPTANCE_REPORT.md` (false-alarm rate on whatever the camera saw, no notes shown).
* No claim is made about recall on a reviewer's notes.

## Definitions

* **Accepted detection:** target-class candidate with score ≥ threshold after NMS (IoU 0.45).
* **Detection level:** accepted detection is **TP** if it matches an unmatched ground-truth box at IoU ≥ 0.5 (greedy by score);
  otherwise **FP**; unmatched ground-truth boxes are **FN**. Precision, recall, F1 follow.
* **Image level:** *positive image* = has a target box; *negative image* = none. A negative image is a **false-positive image**
  if it has ≥ 1 accepted detection. Negative-image FP rate = FP images ÷ negative images, always printed with its denominator and a
  95 % Wilson interval (0 of 10 is compatible with a true rate up to ≈ 28 %).
* **AP@0.5:** all-point interpolated over all scores.
* **Latency:** per image on the evaluating machine; forward pass and full pipeline (camera/display excluded).
* Scores are detector confidences, **not calibrated probabilities**.

## Threshold rule

Rule 1: among thresholds with negative-image FP rate ≤ 0 and recall ≥ 0.80, take the highest recall; if several tie, use the
**middle** of the tied range (not its edge, which would leave no margin). Fallback rule 2 (F0.5) is reported loudly when rule 1 is
unreachable. `evaluate.py` refuses `--select-threshold` on test/train.

## Temporal confirmation and any second stage

Not credited in any metric: evaluation uses raw per-frame decisions. `--confirm-frames N` only affects the live display. No
second-stage verifier exists, so no ablation is claimed.
