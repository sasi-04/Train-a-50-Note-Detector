# Evaluation: `test` split

- Generated: 2026-10-28T16:23:28+00:00
- Model: `runs\rs50_split3\model_split3.onnx` sha256 `5b74b15af7d2f797` (imgsz 480)
- Dataset manifest hash: `7db1d0e102c3300e`; images in this split: 259
- Threshold: **0.27** (from configs/inference.yaml); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

## Overall

- Detection level: TP 63, FP 12, FN 5; precision 84.0%, recall 92.6%, F1 0.881; AP@0.5 0.911 (all scores)
- Image level: 68 positive / 191 negative images; missed positives 5; **negative images with a false ₹50: 8/191** (95% CI upper bound 8.0%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 63 | 5 |
| actual none | 8 | 183 |

## What the app shows, per denomination (green ₹50 wins over red 'Other note')

Thresholds: ₹50 0.27, other note 0.37 (from configs/inference.yaml)

| denomination | images | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|---|
| 10 | 22 | 0 | 19 | 3 |
| 100 | 19 | 0 | 17 | 2 |
| 20 | 63 | 0 | 62 | 1 |
| 200 | 28 | 6 | 21 | 1 |
| 500 | 21 | 0 | 19 | 2 |
| 50_new | 68 | 67 | 0 | 1 |
| background | 38 | 2 | 0 | 36 |

- 'Other note' class at threshold 0.37: TP 138, FP 1, FN 15; precision 99.3%, recall 90.2%; AP@0.5 0.928

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 22 | 0 | 22 | 0 | 0 | 0 | n/a | 0/22 (0.0%, <= 14.9%) |
| 100 | 19 | 0 | 19 | 0 | 0 | 0 | n/a | 0/19 (0.0%, <= 16.8%) |
| 20 | 63 | 0 | 63 | 0 | 0 | 0 | n/a | 0/63 (0.0%, <= 5.7%) |
| 200 | 28 | 0 | 28 | 0 | 0 | 6 | n/a | 6/28 (21.4%, <= 39.5%) |
| 500 | 21 | 0 | 21 | 0 | 0 | 0 | n/a | 0/21 (0.0%, <= 15.5%) |
| 50_new | 68 | 68 | 0 | 63 | 5 | 0 | 92.6% | n/a |
| background | 38 | 0 | 38 | 0 | 0 | 2 | n/a | 2/38 (5.3%, <= 17.3%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 6 | 6 | 0 | 3 | 3 | 0 | 50.0% | n/a |
| 50_new/front | 8 | 8 | 0 | 8 | 0 | 0 | 100.0% | n/a |
| 50_new/unknown | 54 | 54 | 0 | 52 | 2 | 0 | 96.3% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 259 | 68 | 191 | 63 | 5 | 8 | 92.6% | 8/191 (4.2%, <= 8.0%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 259 | 68 | 191 | 63 | 5 | 8 | 92.6% | 8/191 (4.2%, <= 8.0%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_b_close | 12 | 12 | 0 | 9 | 3 | 0 | 75.0% | n/a |
| scan_test_blue | 6 | 2 | 4 | 2 | 0 | 0 | 100.0% | 0/4 (0.0%, <= 49.0%) |
| scan_test_crop | 11 | 0 | 11 | 0 | 0 | 6 | n/a | 6/11 (54.5%, <= 78.7%) |
| webcam_s03 | 164 | 22 | 142 | 20 | 2 | 2 | 90.9% | 2/142 (1.4%, <= 5.0%) |
| webcam_s05 | 66 | 32 | 34 | 32 | 0 | 0 | 100.0% | 0/34 (0.0%, <= 10.2%) |

### Required scenarios

- front side: 8/8 found, 0 missed
- back side: 3/6 found, 3 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 63.8 ms, p95 74.1 ms
- Full pipeline (pre+forward+post) median 67.7 ms, p95 78.6 ms (~14.8 FPS excluding camera/display)

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
