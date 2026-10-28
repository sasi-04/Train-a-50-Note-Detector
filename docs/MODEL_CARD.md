# Model card: ₹50 Vision detector

## Summary

| | |
|---|---|
| Task | detect the **new Indian ₹50 note** (front or back) and flag **other banknotes** separately |
| Architecture | Ultralytics YOLOv8n (3.0 M params), COCO-pretrained start, fine-tuned on CPU |
| Classes | `0: fifty_rupee_note` (green box `₹50`), `1: other_note` (red box `Other note`; ₹10/20/100/200/500, denomination not named) |
| Input | 480×480 letterboxed RGB, float32 NCHW, /255 |
| Output | `(1, 6, 4725)`: `cx, cy, w, h, score₀, score₁`; one class-agnostic NMS outside the graph (IoU 0.45) |
| Export | ONNX opset 12, fixed shape, FP32, 12.7 MB, sha256 `93a630e8…`; metadata in `model.meta.json` |
| Runtime | OpenCV DNN, CPU (OpenCV 5.0.0 verified; 4.14.0 verified for the loader/tests) |
| Thresholds | ₹50 **0.27**, other note **0.37**, chosen on the validation split of the evaluation model (`rs50_split3`) and carried over |
| Post-filter | box must be banknote-shaped (short side ≥ 5 % of the frame's short side, aspect ≤ 4.5) |
| Versions | ultralytics 8.4.189, torch 2.14.1 (CPU) |
| Licence | Ultralytics is AGPL-3.0; review before commercial use |

## Intended use and non-use

Technical demonstration that a webcam application can find the specified ₹50 note and not label other notes or objects as ₹50.
**Not** a currency-authentication or counterfeit-detection system; a detection does not show a note is genuine. The red box says
"some other banknote", not which one.

## Data (801 images, all the developer's own; nothing downloaded or generated)

* 117 phone/scanner-app images; 684 webcam frames from five sittings (`scripts/collect_webcam.py`); sessions 4 and 5 are the
  upright (portrait) ₹50 and ₹20.
* ₹50 targets 198 boxed images, other-note boxes 464, no-note images 139 (room, people, paper, phone, window, blue-green clothes).
* ₹50 boxes: 51 colour-proposed + checked, 147 hand-drawn + overlay-verified. Other-note boxes: 196 hand-drawn + overlay-verified,
  268 helper-model proposals each reviewed on contact sheets (49 unusable frames dropped).
* Splits by session/cluster; webcam session 1 → train, 2 → val, 3 → test, 4 → train, 5 → test. Webcam frames show the developer
  and bystanders.
* **Narrowness:** one ₹50 note and one copy of each other note, one person, a few rooms, no lighting/angle metadata; frames inside
  a capture are highly correlated.

## Training

`configs/train.yaml`: 480 px, batch 8, seed 0, deterministic, CPU; augmentation: hue ±0.01, saturation 0.5, value 0.4, rotation
±15°, translate 0.1, scale 0.6, shear 2°, perspective 5e-4, mosaic 1.0, no flips/mixup. Training also adds, with `--rot90`, a 90° rotated copy of every labelled-note
training image (kept in the run folder). Runs: **`rs50_split2`** (train split, 40 epochs), **`rs50_split3`** (that model +
sessions 4 and rotated copies, 20 epochs on the train split; used for every held-out number) and **`rs50_final3`** (shipped:
`rs50_split3` fine-tuned 12 epochs on all 801 images).

## Evaluation (held-out, evaluation model rs50_split3; details in EVALUATION.md)

| held-out test (259 frames) | result |
|---|---|
| new ₹50 (68) | green ₹50 on **67 (99 %)**; upright session 5: **32/32** (previous model 4/32) |
| other notes (153 images) | 6 ₹200 images **wrongly shown as ₹50**; red box on 138 boxes (90 %) |
| no note (38) | nothing shown on 36; **2 shown as ₹50** (ceiling) |

Non-₹50 images labelled ₹50: **8 of 191 (4.2 %)**, 95 % CI upper bound 8.0 % (previous model 0 of 191 on the same test). The shipped
model has no held-out score. Live: 0 ₹50 and 0 red boxes in 200 webcam frames with no note shown. Latency (i5-7300U, 2 cores):
≈ 64 ms inference / 68 ms per live frame (≈ 15 FPS).

## Known failure modes

* **Observed:** full-frame close-ups of a ₹200 scored 0.73–0.85 as ₹50 (6 images); a ceiling region scored 0.65–0.67 (2 frames);
  back-side ₹50 3 of 6 found; 15 other-note boxes without a red box; the first phone-image-only model scored a window at 0.69 as
  ₹50 (fixed by webcam negatives).
* **Likely:** another copy of the ₹50 or of other notes in poor condition, dim light, glare, a different camera's colour balance,
  notes far from the camera, several notes in one frame, a note-sized coloured object that resembles one of the classes.

## Hardware

Windows 10, Intel Core i5-7300U, no GPU.
