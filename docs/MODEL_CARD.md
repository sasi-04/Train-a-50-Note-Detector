# Model card: ₹50 Vision detector

## Summary

| | |
|---|---|
| Task | detect the **new Indian ₹50 note** (front or back) and flag **other banknotes** separately |
| Architecture | Ultralytics YOLOv8n (3.0 M params), COCO-pretrained start, fine-tuned on CPU |
| Classes | `0: fifty_rupee_note` (green box `₹50`), `1: other_note` (red box `Other note`; ₹10/20/100/200/500, denomination not named) |
| Input | 480×480 letterboxed RGB, float32 NCHW, /255 |
| Output | `(1, 6, 4725)`: `cx, cy, w, h, score₀, score₁`; one class-agnostic NMS outside the graph (IoU 0.45) |
| Export | ONNX opset 12, fixed shape, FP32, 12.7 MB, sha256 `ee2a0455…f7b04b`; metadata in `model.meta.json` |
| Runtime | OpenCV DNN, CPU (OpenCV 5.0.0 verified; 4.14.0 verified for the loader/tests) |
| Thresholds | ₹50 **0.62**, other note **0.82**, chosen on a validation webcam session of the evaluation model |
| Post-filter | box must be banknote-shaped (short side ≥ 5 % of the frame's short side, aspect ≤ 4.5) |
| Versions | ultralytics 8.4.189, torch 2.14.1 (CPU) |
| Licence | Ultralytics is AGPL-3.0; review before commercial use |

## Intended use and non-use

Technical demonstration that a webcam application can find the specified ₹50 note and not label other notes or objects as ₹50.
**Not** a currency-authentication or counterfeit-detection system; a detection does not show a note is genuine. The red box says
"some other banknote", not which one.

## Data (689 images, all the developer's own; nothing downloaded or generated)

* 117 phone/scanner-app images; 572 webcam frames from three sittings (`scripts/collect_webcam.py`).
* ₹50 targets 120 boxed images, other-note boxes 430, no-note images 139 (room, people, paper, phone, window, blue-green clothes).
* ₹50 boxes: 51 colour-proposed + checked, 69 hand-drawn + overlay-verified. Other-note boxes: 162 hand-drawn + overlay-verified,
  268 helper-model proposals each reviewed on contact sheets (49 unusable frames dropped).
* Splits by session/cluster; webcam session 1 → train, 2 → val, 3 → test. Webcam frames show the developer and bystanders.
* **Narrowness:** one ₹50 note and one copy of each other note, one person, a few rooms, no lighting/angle metadata; frames inside
  a capture are highly correlated.

## Training

`configs/train.yaml`: 480 px, batch 8, seed 0, deterministic, CPU; augmentation: hue ±0.01, saturation 0.5, value 0.4, rotation
±15°, translate 0.1, scale 0.6, shear 2°, perspective 5e-4, mosaic 1.0, no flips/mixup. Two runs: **`rs50_split2`** (train split,
40 epochs; used for every held-out number) and **`rs50_final2`** (shipped: that model fine-tuned 12 epochs on all 689 images).

## Evaluation (held-out, evaluation model; details in EVALUATION.md)

| held-out test (193 frames) | result |
|---|---|
| new ₹50 (36) | green ₹50 on **33 (92 %)** |
| other notes (119) | **0 shown as ₹50**; red box on 84 (71 %); nothing on 35 |
| no note (38) | nothing shown on all 38 |

Non-₹50 images labelled ₹50: **0 of 157** (95 % CI upper bound 2.4 %). The shipped model has no held-out score. Live: 0 ₹50 and 0
red boxes in 200 webcam frames with no note shown. Latency (i5-7300U, 2 cores): ≈ 76 ms inference / 80 ms per live frame (≈ 12 FPS).

## Known failure modes

* **Observed:** ~29 % of other-note frames (blurred, half-hidden, in motion) get no red box; 3 of 36 ₹50 frames were missed; the
  first phone-image-only model scored a window at 0.69 as ₹50 (fixed by webcam negatives).
* **Likely:** another copy of the ₹50 or of other notes in poor condition, dim light, glare, a different camera's colour balance,
  notes far from the camera, several notes in one frame, a note-sized coloured object that resembles one of the classes.

## Hardware

Windows 10, Intel Core i5-7300U, no GPU.
