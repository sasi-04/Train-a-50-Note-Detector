# Evaluation: `test` split

- Generated: 2026-10-27T14:47:13+00:00
- Model: `runs\rs50_split\model_split.onnx` sha256 `0d44fb5904fc6109` (imgsz 480)
- Dataset manifest hash: `c61ece36d59f192b`; images in this split: 203
- Threshold: **0.46** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

## Overall

- Detection level: TP 23, FP 0, FN 13; precision 100.0%, recall 63.9%, F1 0.780; AP@0.5 0.922 (all scores)
- Image level: 36 positive / 167 negative images; missed positives 13; **negative images with a false ₹50: 0/167** (95% CI upper bound 2.2%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 23 | 13 |
| actual none | 0 | 167 |

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 23 | 0 | 23 | 0 | 0 | 0 | n/a | 0/23 (0.0%, <= 14.3%) |
| 100 | 23 | 0 | 23 | 0 | 0 | 0 | n/a | 0/23 (0.0%, <= 14.3%) |
| 20 | 31 | 0 | 31 | 0 | 0 | 0 | n/a | 0/31 (0.0%, <= 11.0%) |
| 200 | 29 | 0 | 29 | 0 | 0 | 0 | n/a | 0/29 (0.0%, <= 11.7%) |
| 500 | 23 | 0 | 23 | 0 | 0 | 0 | n/a | 0/23 (0.0%, <= 14.3%) |
| 50_new | 36 | 36 | 0 | 23 | 13 | 0 | 63.9% | n/a |
| background | 38 | 0 | 38 | 0 | 0 | 0 | n/a | 0/38 (0.0%, <= 9.2%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 6 | 6 | 0 | 6 | 0 | 0 | 100.0% | n/a |
| 50_new/front | 8 | 8 | 0 | 4 | 4 | 0 | 50.0% | n/a |
| 50_new/unknown | 22 | 22 | 0 | 13 | 9 | 0 | 59.1% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 203 | 36 | 167 | 23 | 13 | 0 | 63.9% | 0/167 (0.0%, <= 2.2%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 203 | 36 | 167 | 23 | 13 | 0 | 63.9% | 0/167 (0.0%, <= 2.2%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_b_close | 12 | 12 | 0 | 8 | 4 | 0 | 66.7% | n/a |
| scan_test_blue | 6 | 2 | 4 | 2 | 0 | 0 | 100.0% | 0/4 (0.0%, <= 49.0%) |
| scan_test_crop | 11 | 0 | 11 | 0 | 0 | 0 | n/a | 0/11 (0.0%, <= 25.9%) |
| webcam_s03 | 174 | 22 | 152 | 13 | 9 | 0 | 59.1% | 0/152 (0.0%, <= 2.5%) |

### Required scenarios

- front side: 4/8 found, 4 missed
- back side: 6/6 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 131.5 ms, p95 145.3 ms
- Full pipeline (pre+forward+post) median 137.4 ms, p95 152.5 ms (~7.3 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
