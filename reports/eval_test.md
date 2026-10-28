# Evaluation: `test` split

- Generated: 2026-10-28T12:42:23+00:00
- Model: `runs\rs50_split2\model_split2.onnx` sha256 `46100956e47e3d19` (imgsz 480)
- Dataset manifest hash: `11c573e30d49235d`; images in this split: 193
- Threshold: **0.62** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

## Overall

- Detection level: TP 32, FP 1, FN 4; precision 97.0%, recall 88.9%, F1 0.928; AP@0.5 0.938 (all scores)
- Image level: 36 positive / 157 negative images; missed positives 4; **negative images with a false ₹50: 0/157** (95% CI upper bound 2.4%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 32 | 4 |
| actual none | 0 | 157 |

## What the app shows, per denomination (green ₹50 wins over red 'Other note')

Thresholds: ₹50 0.62, other note 0.82 (from configs/inference.yaml)

| denomination | images | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|---|
| 10 | 22 | 0 | 14 | 8 |
| 100 | 19 | 0 | 11 | 8 |
| 20 | 29 | 0 | 23 | 6 |
| 200 | 28 | 0 | 23 | 5 |
| 500 | 21 | 0 | 13 | 8 |
| 50_new | 36 | 33 | 0 | 3 |
| background | 38 | 0 | 0 | 38 |

- 'Other note' class at threshold 0.82: TP 84, FP 0, FN 35; precision 100.0%, recall 70.6%; AP@0.5 0.922

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 22 | 0 | 22 | 0 | 0 | 0 | n/a | 0/22 (0.0%, <= 14.9%) |
| 100 | 19 | 0 | 19 | 0 | 0 | 0 | n/a | 0/19 (0.0%, <= 16.8%) |
| 20 | 29 | 0 | 29 | 0 | 0 | 0 | n/a | 0/29 (0.0%, <= 11.7%) |
| 200 | 28 | 0 | 28 | 0 | 0 | 0 | n/a | 0/28 (0.0%, <= 12.1%) |
| 500 | 21 | 0 | 21 | 0 | 0 | 0 | n/a | 0/21 (0.0%, <= 15.5%) |
| 50_new | 36 | 36 | 0 | 32 | 4 | 0 | 88.9% | n/a |
| background | 38 | 0 | 38 | 0 | 0 | 0 | n/a | 0/38 (0.0%, <= 9.2%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 6 | 6 | 0 | 5 | 1 | 0 | 83.3% | n/a |
| 50_new/front | 8 | 8 | 0 | 8 | 0 | 0 | 100.0% | n/a |
| 50_new/unknown | 22 | 22 | 0 | 19 | 3 | 0 | 86.4% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 193 | 36 | 157 | 32 | 4 | 0 | 88.9% | 0/157 (0.0%, <= 2.4%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 193 | 36 | 157 | 32 | 4 | 0 | 88.9% | 0/157 (0.0%, <= 2.4%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_b_close | 12 | 12 | 0 | 11 | 1 | 0 | 91.7% | n/a |
| scan_test_blue | 6 | 2 | 4 | 2 | 0 | 0 | 100.0% | 0/4 (0.0%, <= 49.0%) |
| scan_test_crop | 11 | 0 | 11 | 0 | 0 | 0 | n/a | 0/11 (0.0%, <= 25.9%) |
| webcam_s03 | 164 | 22 | 142 | 19 | 3 | 0 | 86.4% | 0/142 (0.0%, <= 2.6%) |

### Required scenarios

- front side: 8/8 found, 0 missed
- back side: 5/6 found, 1 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 62.7 ms, p95 83.7 ms
- Full pipeline (pre+forward+post) median 66.5 ms, p95 88.4 ms (~15.0 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
