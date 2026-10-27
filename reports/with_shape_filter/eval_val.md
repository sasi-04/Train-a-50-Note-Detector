# Evaluation: `val` split

- Generated: 2026-10-27T04:21:01+00:00
- Model: `runs\rs50_split\model_split.onnx` sha256 `983d1e3221d76252` (imgsz 480)
- Dataset manifest hash: `258f672f30b057e4`; images in this split: 35
- Threshold: **0.54** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

## Overall

- Detection level: TP 13, FP 0, FN 0; precision 100.0%, recall 100.0%, F1 1.000; AP@0.5 1.000 (all scores)
- Image level: 13 positive / 22 negative images; missed positives 0; **negative images with a false ₹50: 0/22** (95% CI upper bound 14.9%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 13 | 0 |
| actual none | 0 | 22 |

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 7 | 0 | 7 | 0 | 0 | 0 | n/a | 0/7 (0.0%, <= 35.4%) |
| 100 | 8 | 0 | 8 | 0 | 0 | 0 | n/a | 0/8 (0.0%, <= 32.4%) |
| 200 | 1 | 0 | 1 | 0 | 0 | 0 | n/a | 0/1 (0.0%, <= 79.3%) |
| 500 | 5 | 0 | 5 | 0 | 0 | 0 | n/a | 0/5 (0.0%, <= 43.4%) |
| 50_new | 13 | 13 | 0 | 13 | 0 | 0 | 100.0% | n/a |
| background | 1 | 0 | 1 | 0 | 0 | 0 | n/a | 0/1 (0.0%, <= 79.3%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 7 | 7 | 0 | 7 | 0 | 0 | 100.0% | n/a |
| 50_new/unknown | 6 | 6 | 0 | 6 | 0 | 0 | 100.0% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 35 | 13 | 22 | 13 | 0 | 0 | 100.0% | 0/22 (0.0%, <= 14.9%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 35 | 13 | 22 | 13 | 0 | 0 | 100.0% | 0/22 (0.0%, <= 14.9%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_b_face | 13 | 12 | 1 | 12 | 0 | 0 | 100.0% | 0/1 (0.0%, <= 79.3%) |
| scan_val | 22 | 1 | 21 | 1 | 0 | 0 | 100.0% | 0/21 (0.0%, <= 15.5%) |

### Required scenarios

- front side: **NOT TESTED** (no images)
- back side: 7/7 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 55.9 ms, p95 64.0 ms
- Full pipeline (pre+forward+post) median 59.6 ms, p95 70.0 ms (~16.8 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
