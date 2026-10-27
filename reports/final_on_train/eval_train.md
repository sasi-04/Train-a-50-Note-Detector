# Evaluation: `train` split

- Generated: 2026-10-27T04:21:13+00:00
- Model: `model.onnx` sha256 `c4cc2902700818cb` (imgsz 480)
- Dataset manifest hash: `258f672f30b057e4`; images in this split: 53
- Threshold: **0.54** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

> **Training split: metrics are optimistic (the model was trained on these images).**

## Overall

- Detection level: TP 24, FP 0, FN 0; precision 100.0%, recall 100.0%, F1 1.000; AP@0.5 1.000 (all scores)
- Image level: 24 positive / 29 negative images; missed positives 0; **negative images with a false ₹50: 0/29** (95% CI upper bound 11.7%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 24 | 0 |
| actual none | 0 | 29 |

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 4 | 0 | 4 | 0 | 0 | 0 | n/a | 0/4 (0.0%, <= 49.0%) |
| 100 | 7 | 0 | 7 | 0 | 0 | 0 | n/a | 0/7 (0.0%, <= 35.4%) |
| 20 | 4 | 0 | 4 | 0 | 0 | 0 | n/a | 0/4 (0.0%, <= 49.0%) |
| 200 | 5 | 0 | 5 | 0 | 0 | 0 | n/a | 0/5 (0.0%, <= 43.4%) |
| 500 | 5 | 0 | 5 | 0 | 0 | 0 | n/a | 0/5 (0.0%, <= 43.4%) |
| 50_new | 24 | 24 | 0 | 24 | 0 | 0 | 100.0% | n/a |
| background | 4 | 0 | 4 | 0 | 0 | 0 | n/a | 0/4 (0.0%, <= 49.0%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 4 | 4 | 0 | 4 | 0 | 0 | 100.0% | n/a |
| 50_new/front | 20 | 20 | 0 | 20 | 0 | 0 | 100.0% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 53 | 24 | 29 | 24 | 0 | 0 | 100.0% | 0/29 (0.0%, <= 11.7%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 53 | 24 | 29 | 24 | 0 | 0 | 100.0% | 0/29 (0.0%, <= 11.7%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_a | 20 | 16 | 4 | 16 | 0 | 0 | 100.0% | 0/4 (0.0%, <= 49.0%) |
| scan_train | 33 | 8 | 25 | 8 | 0 | 0 | 100.0% | 0/25 (0.0%, <= 13.3%) |

### Required scenarios

- front side: 20/20 found, 0 missed
- back side: 4/4 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 55.7 ms, p95 60.5 ms
- Full pipeline (pre+forward+post) median 59.1 ms, p95 64.9 ms (~16.9 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
