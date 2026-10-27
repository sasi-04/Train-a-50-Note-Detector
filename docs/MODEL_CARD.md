# Model card: ₹50 Vision detector

## Summary

| | |
|---|---|
| Task | single-class detection of the **new Indian ₹50 note** (front or back) |
| Architecture | Ultralytics YOLOv8n (3.0 M params), COCO-pretrained start, fine-tuned on CPU |
| Class | `0: fifty_rupee_note` (shown as `₹50`) |
| Input | 480×480 letterboxed RGB, float32 NCHW, /255 |
| Output | `(1, 5, 4725)`: `cx, cy, w, h, score`; NMS applied outside the graph (IoU 0.45) |
| Export | ONNX opset 12, fixed shape, FP32, 12.7 MB, sha256 `ce5a6d1c…f7e99`; metadata in `model.meta.json` |
| Runtime | OpenCV DNN, CPU (OpenCV 5.0.0 verified; 4.14.0 verified for the loader/tests) |
| Default threshold | 0.46, chosen on a validation webcam session of the evaluation model |
| Post-filter | box must be banknote-shaped (short side ≥ 5 % of the frame's short side, aspect ≤ 4.5) |
| Versions | ultralytics 8.4.189, torch 2.14.1 (CPU) |
| Licence | Ultralytics is AGPL-3.0; review before commercial use |

## Intended use and non-use

Technical demonstration that a webcam application can find the specified ₹50 note and not label other notes/objects as ₹50.
**Not** a currency-authentication or counterfeit-detection system; a detection does not show a note is genuine.

## Data (738 images, all the developer's own; nothing downloaded or generated)

* 117 phone/scanner-app images: one physical copy of each of ₹10/20/50/100/200/500, 5 no-note frames.
* 621 frames from the developer's webcam in three sittings (`scripts/collect_webcam.py`): ₹50 front/back/sideways (105 boxed),
  ₹10/20/100/200/500, room/people, and blue-green things (a window, clothing).
* ₹50 boxes: 51 from colour-segmentation proposals that were visually checked, 105 drawn by hand from grid views and verified by
  overlay. Negatives have empty labels.
* Splits by session/cluster; webcam session 1 → train, 2 → val, 3 → test.
* **Narrowness:** one ₹50 note, one person, a few rooms, no controlled lighting/angle metadata; frames within a capture are
  highly correlated.

## Training

`configs/train.yaml`: 480 px, batch 8, seed 0, deterministic, CPU. Augmentation: hue ±0.01, saturation 0.5, value 0.4, rotation
±15°, translate 0.1, scale 0.6, shear 2°, perspective 5e-4, mosaic 1.0, no flips/mixup. Two runs: **`rs50_split`** (train
split only, 40 epochs; used for every held-out number) and **`rs50_final`** (shipped: `rs50_split` fine-tuned 12 epochs on all
738 images).

## Evaluation (held-out, evaluation model; details in EVALUATION.md)

| split | ₹50 found | non-₹50 images labelled ₹50 |
|---|---|---|
| val (webcam session 2 + earlier) | 25–26 / 29 | 0 / 171 |
| **test** (webcam session 3 + earlier) | **23 / 36 (64 %)**, back 6/6, front 4/8 | **0 / 167** (CI upper ≈ 2 %) |

The shipped model has no held-out score. Live: 0 accepted ₹50 in 200 webcam frames with no note shown.
Latency on the development CPU (i5-7300U): forward ≈ 83 ms, pipeline ≈ 87 ms median (≈ 11 FPS).

## Known failure modes (observed or likely)

* **Observed:** ₹50 frames that are motion-blurred, strongly backlit, or partly hidden by fingers are often below the threshold
  (some score 0.3–0.45). The very first model (phone images only) scored a window at 0.69 as ₹50; fixed by webcam negatives.
* **Likely:** other copies of the ₹50 in poor condition, dim light, glare, a different camera's colour balance, notes held far from
  the camera (small), another light-coloured note held in a pose not seen in training, several notes in one frame.

## Hardware

Windows 10, Intel Core i5-7300U, no GPU.
