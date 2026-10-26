# Evaluation methodology and results

## Results

**NOT RUN.** No real photographs or trained model exist yet, so there are no banknote-detection results to report.
Nothing in this repository should be read as a measured accuracy, false-positive rate or FPS for the banknote model.
After running the README workflow, paste the contents of `reports/eval_test.md` and `reports/stress_*.md` here.

| scenario | status |
|---|---|
| new ₹50 front / back | NOT TESTED |
| different lighting / angles / distances | NOT TESTED |
| ₹10, ₹20, ₹100, ₹200, ₹500 | NOT TESTED |
| ordinary objects, empty background | NOT TESTED |
| partially obscured target | NOT TESTED |
| previously unseen capture sessions (test split) | NOT TESTED |
| ONNX ≡ PyTorch parity on the real model | NOT TESTED (parity machinery verified on random-init and synthetic-smoke models) |

## Definitions

* **Accepted detection:** target-class candidate with score ≥ threshold after NMS (IoU 0.45).
* **Detection level:** accepted detection is **TP** if it matches an unmatched ground-truth box at IoU ≥ 0.5 (greedy by
  score); otherwise **FP**; unmatched ground-truth boxes are **FN**. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1.
* **Image level:** *positive image* = has a target box; *negative image* = no target box. A negative image is a
  **false-positive image** if it contains ≥ 1 accepted detection. **Negative-image FP rate** = FP images ÷ negative
  images, always printed with its denominator and a 95 % Wilson interval (0 of 10 is compatible with a true rate up to ≈ 28 %).
* **False positive categories:** per `denomination` (₹10/20/100/200/500, old ₹50, object, background) plus lighting,
  angle, background and session breakdowns.
* **AP@0.5:** all-point interpolated, over all scores (threshold-free).
* **Latency:** measured on the evaluating machine per image: forward pass, and full pre+forward+post pipeline. Camera
  capture and window drawing are excluded from the evaluation script and included in the live HUD FPS.
* Scores are detector confidences, **not calibrated probabilities**; no calibration has been measured.

## Protocol

1. **Validation split** → `evaluate.py --split val --select-threshold --write-config`. Rule 1: among thresholds whose
   negative-image FP rate ≤ the cap (default 0) and recall ≥ 0.80, take the highest recall (ties → higher threshold).
   Rule 2 fallback (reported loudly): maximise F0.5. The sweep CSV/plot show the whole trade-off.
2. **Stress test** → `stress_test.py --split val`: negatives only, per-category counts at several thresholds, distribution
   of the strongest false score per image, annotated hardest negatives for manual review.
3. **Held-out test** → `evaluate.py --split test` once, with the frozen threshold. The script refuses to select a threshold
   on test/train. Report whatever comes out; if it is bad, collect more varied data rather than re-tuning on test.
4. The train split can be evaluated too, but its report is stamped "optimistic".

## Temporal confirmation and any second stage

Not credited in any metric: the evaluation scripts always use raw per-frame decisions. `--confirm-frames N` only affects
the live display. No second-stage verifier exists, so no ablation is claimed.

## Dry-run evidence (pipeline only)

The scripts were exercised end-to-end on a throwaway synthetic dataset (coloured rectangles, 186 images, 6 sessions,
3 training epochs). The numbers it produced (e.g. recall 33 %) describe that meaningless toy model and are intentionally
not reproduced here.
