# Model card: ₹50 Vision detector

## Summary

| | |
|---|---|
| Task | single-class detection of the **new Indian ₹50 note** (front or back) |
| Architecture | Ultralytics YOLOv8n (3.0 M params), COCO-pretrained start, fine-tuned for 120 epochs on CPU |
| Class | `0: fifty_rupee_note` (shown as `₹50`) |
| Input | 480×480 letterboxed RGB, float32 NCHW, /255 |
| Output | `(1, 5, 4725)`: `cx, cy, w, h, score`; NMS applied outside the graph (IoU 0.45) |
| Export | ONNX opset 12, fixed shape, FP32, 12.7 MB, sha256 `c4cc2902…ba6c4`; metadata in `model.meta.json` |
| Runtime | OpenCV DNN, CPU (OpenCV 5.0.0 verified; 4.14.0 verified for the loader/tests) |
| Default threshold | 0.54, chosen on the validation split of the *evaluation* model |
| Versions | ultralytics 8.4.189, torch 2.14.1 (CPU) |
| Licence | Ultralytics is AGPL-3.0; review before commercial use |

## Intended use and non-use

Technical demonstration that a webcam application can find the specified ₹50 note and not label other notes/objects as ₹50.
**Not** a currency-authentication or counterfeit-detection system; a detection does not show a note is genuine.

## Data

* 117 images from the developer's own captures (scanner-app/phone), one physical note per denomination:
  new ₹50 ×51 (front, back and unknown-side, including 40 hand-held frames), ₹10 ×12, ₹20 ×12, ₹100 ×15, ₹200 ×12, ₹500 ×10,
  no-note frames ×5.
* Nothing downloaded or generated. Boxes for the 51 targets were proposed by colour/segmentation and **visually reviewed**
  (not hand-drawn). Negatives have empty labels.
* Splits by whole cluster/session (`dataset/metadata/*.yaml` documents each choice).
* **Narrowness:** one ₹50 note, essentially one room/person, no lighting or angle metadata, no other note held in hand, no
  clutter or objects beyond 5 frames.

## Training

`configs/train.yaml`: 480 px, ≤ 120 epochs, patience 50, batch 8, seed 0, deterministic, CPU. Augmentation: hue ±0.01,
saturation 0.5, value 0.4, rotation ±15°, translate 0.1, scale 0.6, shear 2°, perspective 5e-4, mosaic 1.0, no flips/mixup.
Two runs: `rs50_split` (train split only; used for all held-out numbers) and **`rs50_final` (all 117 images; shipped)**.

## Evaluation (evaluation model, held-out; details in EVALUATION.md)

| split | targets found | false ₹50 on negative images |
|---|---|---|
| val | 13 / 13 | 0 / 22 |
| test | 7 / 14 (back 6/6, **front 1/8**) | 1 / 15 (a ₹20) |

The shipped model has no held-out score. Live: 0 accepted ₹50 in 150 webcam frames with no note shown.
Latency on the development CPU (i5-7300U): forward ≈ 87 ms median, pipeline ≈ 91 ms (≈ 11 FPS).

## Known failure modes (observed or likely)

* **Observed:** front-side close-ups that are dark, rotated and fill the frame were missed by the evaluation model; a ₹20 on a
  blue table scored above the threshold.
* **Likely:** other light-coloured notes held in hand, tilted/sideways notes, dim light, glare, clutter, different camera white
  balance, the same-design ₹50 in poor condition, a second note in view.

## Hardware

Windows 10, Intel Core i5-7300U, no GPU.
