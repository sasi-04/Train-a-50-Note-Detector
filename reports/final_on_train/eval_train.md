# Evaluation: `train` split

- Generated: 2026-10-28T13:41:37+00:00
- Model: `model.onnx` sha256 `ee2a0455e060c746` (imgsz 480)
- Dataset manifest hash: `11c573e30d49235d`; images in this split: 308
- Threshold: **0.62** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

> **Training split: metrics are optimistic (the model was trained on these images).**

## Overall

- Detection level: TP 55, FP 0, FN 0; precision 100.0%, recall 100.0%, F1 1.000; AP@0.5 1.000 (all scores)
- Image level: 55 positive / 253 negative images; missed positives 0; **negative images with a false ₹50: 0/253** (95% CI upper bound 1.5%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 55 | 0 |
| actual none | 0 | 253 |

## What the app shows, per denomination (green ₹50 wins over red 'Other note')

Thresholds: ₹50 0.62, other note 0.82 (from configs/inference.yaml)

| denomination | images | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|---|
| 10 | 36 | 0 | 36 | 0 |
| 100 | 38 | 0 | 31 | 7 |
| 20 | 38 | 0 | 34 | 4 |
| 200 | 39 | 0 | 31 | 8 |
| 500 | 37 | 0 | 34 | 3 |
| 50_new | 55 | 55 | 0 | 0 |
| background | 65 | 0 | 0 | 65 |

- 'Other note' class at threshold 0.82: TP 166, FP 0, FN 22; precision 100.0%, recall 88.3%; AP@0.5 1.000

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 36 | 0 | 36 | 0 | 0 | 0 | n/a | 0/36 (0.0%, <= 9.6%) |
| 100 | 38 | 0 | 38 | 0 | 0 | 0 | n/a | 0/38 (0.0%, <= 9.2%) |
| 20 | 38 | 0 | 38 | 0 | 0 | 0 | n/a | 0/38 (0.0%, <= 9.2%) |
| 200 | 39 | 0 | 39 | 0 | 0 | 0 | n/a | 0/39 (0.0%, <= 9.0%) |
| 500 | 37 | 0 | 37 | 0 | 0 | 0 | n/a | 0/37 (0.0%, <= 9.4%) |
| 50_new | 55 | 55 | 0 | 55 | 0 | 0 | 100.0% | n/a |
| background | 65 | 0 | 65 | 0 | 0 | 0 | n/a | 0/65 (0.0%, <= 5.6%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 4 | 4 | 0 | 4 | 0 | 0 | 100.0% | n/a |
| 50_new/front | 20 | 20 | 0 | 20 | 0 | 0 | 100.0% | n/a |
| 50_new/unknown | 31 | 31 | 0 | 31 | 0 | 0 | 100.0% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 308 | 55 | 253 | 55 | 0 | 0 | 100.0% | 0/253 (0.0%, <= 1.5%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 308 | 55 | 253 | 55 | 0 | 0 | 100.0% | 0/253 (0.0%, <= 1.5%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_a | 20 | 16 | 4 | 16 | 0 | 0 | 100.0% | 0/4 (0.0%, <= 49.0%) |
| scan_train | 33 | 8 | 25 | 8 | 0 | 0 | 100.0% | 0/25 (0.0%, <= 13.3%) |
| webcam_s01 | 255 | 31 | 224 | 31 | 0 | 0 | 100.0% | 0/224 (0.0%, <= 1.7%) |

### Required scenarios

- front side: 20/20 found, 0 missed
- back side: 4/4 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 68.8 ms, p95 78.7 ms
- Full pipeline (pre+forward+post) median 72.8 ms, p95 82.9 ms (~13.7 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
