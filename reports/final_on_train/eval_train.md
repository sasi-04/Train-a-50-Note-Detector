# Evaluation: `train` split

- Generated: 2026-10-27T15:55:23+00:00
- Model: `model.onnx` sha256 `ce5a6d1ce6373f26` (imgsz 480)
- Dataset manifest hash: `c61ece36d59f192b`; images in this split: 335
- Threshold: **0.46** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

> **Training split: metrics are optimistic (the model was trained on these images).**

## Overall

- Detection level: TP 55, FP 0, FN 0; precision 100.0%, recall 100.0%, F1 1.000; AP@0.5 1.000 (all scores)
- Image level: 55 positive / 280 negative images; missed positives 0; **negative images with a false ₹50: 0/280** (95% CI upper bound 1.4%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 55 | 0 |
| actual none | 0 | 280 |

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 42 | 0 | 42 | 0 | 0 | 0 | n/a | 0/42 (0.0%, <= 8.4%) |
| 100 | 45 | 0 | 45 | 0 | 0 | 0 | n/a | 0/45 (0.0%, <= 7.9%) |
| 20 | 42 | 0 | 42 | 0 | 0 | 0 | n/a | 0/42 (0.0%, <= 8.4%) |
| 200 | 43 | 0 | 43 | 0 | 0 | 0 | n/a | 0/43 (0.0%, <= 8.2%) |
| 500 | 43 | 0 | 43 | 0 | 0 | 0 | n/a | 0/43 (0.0%, <= 8.2%) |
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
| unknown | 335 | 55 | 280 | 55 | 0 | 0 | 100.0% | 0/280 (0.0%, <= 1.4%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 335 | 55 | 280 | 55 | 0 | 0 | 100.0% | 0/280 (0.0%, <= 1.4%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_a | 20 | 16 | 4 | 16 | 0 | 0 | 100.0% | 0/4 (0.0%, <= 49.0%) |
| scan_train | 33 | 8 | 25 | 8 | 0 | 0 | 100.0% | 0/25 (0.0%, <= 13.3%) |
| webcam_s01 | 282 | 31 | 251 | 31 | 0 | 0 | 100.0% | 0/251 (0.0%, <= 1.5%) |

### Required scenarios

- front side: 20/20 found, 0 missed
- back side: 4/4 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 70.0 ms, p95 131.4 ms
- Full pipeline (pre+forward+post) median 73.9 ms, p95 135.5 ms (~13.5 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
