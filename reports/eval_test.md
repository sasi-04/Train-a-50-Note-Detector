# Evaluation: `test` split

- Generated: 2026-10-26T09:44:34+00:00
- Model: `runs\rs50_split\model_split.onnx` sha256 `983d1e3221d76252` (imgsz 480)
- Dataset manifest hash: `258f672f30b057e4`; images in this split: 29
- Threshold: **0.54** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

## Overall

- Detection level: TP 7, FP 8, FN 7; precision 46.7%, recall 50.0%, F1 0.483; AP@0.5 0.657 (all scores)
- Image level: 14 positive / 15 negative images; missed positives 7; **negative images with a false ₹50: 1/15** (95% CI upper bound 29.8%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 7 | 7 |
| actual none | 1 | 14 |

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 1 | 0 | 1 | 0 | 0 | 0 | n/a | 0/1 (0.0%, <= 79.3%) |
| 20 | 8 | 0 | 8 | 0 | 0 | 1 | n/a | 1/8 (12.5%, <= 47.1%) |
| 200 | 6 | 0 | 6 | 0 | 0 | 0 | n/a | 0/6 (0.0%, <= 39.0%) |
| 50_new | 14 | 14 | 0 | 7 | 7 | 0 | 50.0% | n/a |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 6 | 6 | 0 | 6 | 0 | 0 | 100.0% | n/a |
| 50_new/front | 8 | 8 | 0 | 1 | 7 | 0 | 12.5% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 29 | 14 | 15 | 7 | 7 | 1 | 50.0% | 1/15 (6.7%, <= 29.8%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 29 | 14 | 15 | 7 | 7 | 1 | 50.0% | 1/15 (6.7%, <= 29.8%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_b_close | 12 | 12 | 0 | 5 | 7 | 0 | 41.7% | n/a |
| scan_test_blue | 6 | 2 | 4 | 2 | 0 | 1 | 100.0% | 1/4 (25.0%, <= 69.9%) |
| scan_test_crop | 11 | 0 | 11 | 0 | 0 | 0 | n/a | 0/11 (0.0%, <= 25.9%) |

### Required scenarios

- front side: 1/8 found, 7 missed
- back side: 6/6 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 104.8 ms, p95 130.4 ms
- Full pipeline (pre+forward+post) median 111.4 ms, p95 142.6 ms (~9.0 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
