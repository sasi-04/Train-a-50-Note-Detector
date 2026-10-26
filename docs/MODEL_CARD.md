# Model card: ₹50 Vision detector

**Status: no trained model has been produced yet.** Fields marked *pending* are filled after the README workflow is run on
real photographs. Nothing below is a measured result unless it says "measured".

## Summary

| | |
|---|---|
| Task | single-class object detection of the **new Indian ₹50 note** (front or back) |
| Architecture | Ultralytics YOLOv8n (3.0 M parameters), COCO-pretrained initialisation, fine-tuned |
| Class | `0: fifty_rupee_note` (displayed as `₹50`) |
| Input | 480×480 letterboxed RGB (configurable at export), float32 NCHW, /255 |
| Output | `(1, 5, 4725)` at 480: `cx, cy, w, h, score`; NMS done outside the graph |
| Export | ONNX opset 12, fixed shape, FP32, ≈ 12 MB (measured on a random-init export; actual file *pending*) |
| Runtime | OpenCV DNN on CPU (verified with OpenCV 5.0.0 and 4.14.0) |
| Licence | Ultralytics is AGPL-3.0 — review before commercial use |

## Intended use and non-use

Intended: technical demonstration that a webcam application can find the specified ₹50 note and not label other notes or
objects as ₹50. **Not** a currency-authentication or counterfeit-detection system. A detection does **not** show the note is
genuine, and a non-detection does not show it is fake. Not validated for any other denomination or banknote series.

## Data (pending)

* Source: photographs taken by the developer with their own phone, of genuine notes; nothing downloaded or generated.
* Positives: new ₹50, front and back. Negatives: ₹10, ₹20, ₹100, ₹200, ₹500, other objects, empty backgrounds
  (optionally old ₹50). Counts: *pending* (`reports/dataset_audit.md`).
* Annotation: boxes drawn by hand with `scripts/annotate.py`; negatives are empty label files.
* Splits: whole capture sessions → train/val/test; audited for near-duplicate leakage.

## Training (configuration is real; run is pending)

`configs/train.yaml`: YOLOv8n from COCO weights, 480 px, ≤ 100 epochs with patience 25, batch 8, CPU, seed 0,
deterministic; augmentation: hue ±0.01, saturation 0.5, value 0.4, rotation ±15°, translate 0.1, scale 0.4, shear 2°,
perspective 0.0005, mosaic 0.8, no flips, no mixup. Best checkpoint chosen by Ultralytics on the validation split only.

## Evaluation

*Pending.* See [EVALUATION.md](EVALUATION.md) for definitions and protocol. Measured so far, on **random-init/toy models only**
(this machine: i5-7300U, 2 cores, OpenCV 5.0.0): forward-pass median ≈ 115 ms at 640, ≈ 62 ms at 480, ≈ 47 ms at 416,
≈ 24 ms at 320. These are architecture timings, independent of weights.

## Known failure modes to check (hypotheses, not findings)

* ₹10/₹20/₹200 notes under warm indoor light may look similar in colour to the blue-green ₹50.
* Small, distant or heavily tilted notes; strong glare on the polymer thread; notes in a fan or partly folded.
* Other blue-green rectangles (cards, packaging), phone screens showing a note.
* Webcams with strong auto white balance (colour is a class cue and `hsv_h` augmentation is deliberately tiny).
* Over-fitting to the developer's sessions/backgrounds if sessions are few.

## Hardware

Training and all measurements so far: Windows 10, Intel Core i5-7300U @ 2.60 GHz (2 cores / 4 threads), no GPU.
