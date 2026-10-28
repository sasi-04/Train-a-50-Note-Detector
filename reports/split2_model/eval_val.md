# Evaluation: `val` split

- Generated: 2026-10-28T12:41:31+00:00
- Model: `runs\rs50_split2\model_split2.onnx` sha256 `46100956e47e3d19` (imgsz 480)
- Dataset manifest hash: `11c573e30d49235d`; images in this split: 188
- Threshold: **0.62** (selected on validation split); IoU for matching: 0.5
- Hardware: Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 4 logical cores, OpenCV 5.0.0

- Selection: rule 1: negative-image FP rate <= 0.000 and recall >= 0.80; best recall 0.931; 5 thresholds within 0.05 recall of it span 0.60-0.64; the middle one is used, not the edge, so a small shift in conditions does not flip it into false alarms

## Overall

- Detection level: TP 27, FP 0, FN 2; precision 100.0%, recall 93.1%, F1 0.964; AP@0.5 0.998 (all scores)
- Image level: 29 positive / 159 negative images; missed positives 2; **negative images with a false ₹50: 0/159** (95% CI upper bound 2.4%)
- Image-level confusion matrix (rows actual, cols predicted):

| | predicted ₹50 | predicted none |
|---|---|---|
| actual ₹50 | 27 | 2 |
| actual none | 0 | 159 |

## What the app shows, per denomination (green ₹50 wins over red 'Other note')

Thresholds: ₹50 0.62, other note 0.82 (selected on validation split)

| denomination | images | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|---|
| 10 | 27 | 0 | 25 | 2 |
| 100 | 31 | 0 | 27 | 4 |
| 20 | 19 | 0 | 17 | 2 |
| 200 | 19 | 0 | 17 | 2 |
| 500 | 27 | 0 | 22 | 5 |
| 50_new | 29 | 27 | 0 | 2 |
| background | 36 | 0 | 0 | 36 |

- 'Other note' class at threshold 0.82: TP 108, FP 0, FN 15; precision 100.0%, recall 87.8%; AP@0.5 0.983

- Other-note threshold selection: rule 1: negative-image FP rate <= 0.000 and recall >= 0.80; best recall 0.902; 4 thresholds within 0.05 recall of it span 0.80-0.83; the middle one is used, not the edge, so a small shift in conditions does not flip it into false alarms

### By denomination

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 10 | 27 | 0 | 27 | 0 | 0 | 0 | n/a | 0/27 (0.0%, <= 12.5%) |
| 100 | 31 | 0 | 31 | 0 | 0 | 0 | n/a | 0/31 (0.0%, <= 11.0%) |
| 20 | 19 | 0 | 19 | 0 | 0 | 0 | n/a | 0/19 (0.0%, <= 16.8%) |
| 200 | 19 | 0 | 19 | 0 | 0 | 0 | n/a | 0/19 (0.0%, <= 16.8%) |
| 500 | 27 | 0 | 27 | 0 | 0 | 0 | n/a | 0/27 (0.0%, <= 12.5%) |
| 50_new | 29 | 29 | 0 | 27 | 2 | 0 | 93.1% | n/a |
| background | 36 | 0 | 36 | 0 | 0 | 0 | n/a | 0/36 (0.0%, <= 9.6%) |

### Target note by side

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| 50_new/back | 7 | 7 | 0 | 7 | 0 | 0 | 100.0% | n/a |
| 50_new/unknown | 22 | 22 | 0 | 20 | 2 | 0 | 90.9% | n/a |

### By lighting

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 188 | 29 | 159 | 27 | 2 | 0 | 93.1% | 0/159 (0.0%, <= 2.4%) |

### By angle

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| unknown | 188 | 29 | 159 | 27 | 2 | 0 | 93.1% | 0/159 (0.0%, <= 2.4%) |

### By capture session

| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |
|---|---|---|---|---|---|---|---|---|
| hand_b_face | 13 | 12 | 1 | 12 | 0 | 0 | 100.0% | 0/1 (0.0%, <= 79.3%) |
| scan_val | 22 | 1 | 21 | 1 | 0 | 0 | 100.0% | 0/21 (0.0%, <= 15.5%) |
| webcam_s02 | 153 | 16 | 137 | 14 | 2 | 0 | 87.5% | 0/137 (0.0%, <= 2.7%) |

### Required scenarios

- front side: **NOT TESTED** (no images)
- back side: 7/7 found, 0 missed
- dim lighting: **NOT TESTED** (no images)
- angled: **NOT TESTED** (no images)

## Latency (this machine, per image, includes decode)

- Inference (forward) median 62.9 ms, p95 88.2 ms
- Full pipeline (pre+forward+post) median 66.5 ms, p95 92.6 ms (~15.0 FPS excluding camera/display)

## Threshold trade-off at 0.62 (val)

- precision 100.0%, recall 93.1%, negative-image FP 0/159
- Full sweep: `threshold_sweep_val.csv`, plot `threshold_analysis_val.png`

Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.
