# Evaluation: `val` split

- Generated: 2026-10-27T14:46:10+00:00
- Model: `runs\rs50_split\model_split.onnx` sha256 `0d44fb5904fc6109` (imgsz 480)
- Dataset manifest hash: `c61ece36d59f192b`; images in this split: 200
- Threshold: **0.46** (selected on validation split); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

- Selection: rule 1: negative-image FP rate <= 0.000 and recall >= 0.80; best recall 0.897; 10 thresholds within 0.05 recall of it span 0.41-0.50; the middle one is used, not the edge, so a small shift in conditions does not flip it into false alarms

## Overall

- Detection level: TP 25, FP 0, FN 4; precision 100.0%, recall 86.2%, F1 0.926; AP@0.5 0.939 (all scores)
- Image level: 29 positive / 171 negative images; missed positives 4; **negative images with a false ₹50: 0/171** (95% CI upper bound 2.2%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 25 | 4 |
| actual none | 0 | 171 |

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 30 | 0 | 30 | 0 | 0 | 0 | n/a | 0/30 (0.0%, <= 11.4%) |
| 100 | 31 | 0 | 31 | 0 | 0 | 0 | n/a | 0/31 (0.0%, <= 11.0%) |
| 20 | 23 | 0 | 23 | 0 | 0 | 0 | n/a | 0/23 (0.0%, <= 14.3%) |
| 200 | 23 | 0 | 23 | 0 | 0 | 0 | n/a | 0/23 (0.0%, <= 14.3%) |
| 500 | 28 | 0 | 28 | 0 | 0 | 0 | n/a | 0/28 (0.0%, <= 12.1%) |
| 50_new | 29 | 29 | 0 | 25 | 4 | 0 | 86.2% | n/a |
| background | 36 | 0 | 36 | 0 | 0 | 0 | n/a | 0/36 (0.0%, <= 9.6%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 7 | 7 | 0 | 7 | 0 | 0 | 100.0% | n/a |
| 50_new/unknown | 22 | 22 | 0 | 18 | 4 | 0 | 81.8% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 200 | 29 | 171 | 25 | 4 | 0 | 86.2% | 0/171 (0.0%, <= 2.2%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 200 | 29 | 171 | 25 | 4 | 0 | 86.2% | 0/171 (0.0%, <= 2.2%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_b_face | 13 | 12 | 1 | 12 | 0 | 0 | 100.0% | 0/1 (0.0%, <= 79.3%) |
| scan_val | 22 | 1 | 21 | 1 | 0 | 0 | 100.0% | 0/21 (0.0%, <= 15.5%) |
| webcam_s02 | 165 | 16 | 149 | 12 | 4 | 0 | 75.0% | 0/149 (0.0%, <= 2.5%) |

### Required scenarios

- front side: **NOT TESTED** (no images)
- back side: 7/7 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 63.1 ms, p95 73.3 ms
- Full pipeline (pre+forward+post) median 66.8 ms, p95 78.3 ms (~15.0 FPS excluding camera/display)

## Threshold trade-off at 0.46 (val)

- precision 100.0%, recall 86.2%, negative-image FP 0/171
- Full sweep: `threshold_sweep_val.csv`, plot `threshold_analysis_val.png`

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
