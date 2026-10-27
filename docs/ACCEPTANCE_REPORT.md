# Acceptance report

Machine: Windows 10, Intel i5-7300U (2 cores / 4 threads), Python 3.14.4, OpenCV 5.0.0 (portability also checked on 4.14.0),
one webcam (index 0). Date: 2026-10-27.

**Verdict: complete, runnable and measured; reliability on a reviewer's webcam is still not guaranteed.** Every mandatory
deliverable exists and was verified. On a held-out webcam session the evaluation model labelled **0 of 167** non-₹50 frames as
₹50 and found **23 of 36 (64 %)** of the ₹50 frames. The shipped model was then fine-tuned on all data, so it has no held-out
score; the live positive check with real notes in many poses is the developer's to repeat on the reviewing machine.

Status: PASS · PARTIAL · FAIL · BLOCKED · NOT TESTED.

## Test evidence (actually run)

| run | result |
|---|---|
| `pytest -q` (OpenCV 5.0.0) | **115 passed** (includes loading the real `model.onnx`) |
| `pytest` minus parity file (separate venv, OpenCV 4.14.0) | 106 passed (earlier snapshot of the suite) |
| `validate_onnx.py` on the shipped `model.onnx` | **PASS**: OpenCV vs ONNX Runtime max score diff 7e-7; vs PyTorch max box diff 0.0009 px; decoded detections matched (`reports/onnx_validation.md`) |
| `python test.py --model model.onnx --no-display --max-frames 200`, real webcam, no note shown | 200 frames, **0 accepted ₹50**, 0 camera failures; inference median 83 ms, pipeline 87 ms (≈ 11 FPS) |
| shipped model on its own training images (`reports/final_on_train/`) | 55/55 targets, 0/280 false ₹50. **Optimistic (training data).** |
| `python test.py --model model.onnx` with the file missing | clean error, exit 3 |
| `train.py` before the dataset was ready | refused with the blockers (exit 1) |

## Held-out evaluation (evaluation model `rs50_split`, threshold 0.46 chosen on val; see EVALUATION.md)

| split | targets found | non-₹50 images labelled ₹50 |
|---|---|---|
| val (29 target, 171 negative) | 25–26/29 | 0/171 (CI upper 2.2 %) |
| **test** (36 target, 167 negative) | **23/36 (64 %)**; back 6/6, front 4/8 | **0/167** (CI upper 2.2 %): ₹10 0/23, ₹20 0/31, ₹100 0/23, ₹200 0/29, ₹500 0/23, no-note 0/38 |

NOT TESTED on held-out data: dim light and angle as separate groups, several notes in one frame, a second copy of the ₹50.

## Assignment compliance

| item | status | evidence |
|---|---|---|
| Correct target: new ₹50 | PASS | single class, label `₹50` |
| Front-side examples | PASS | front, back and unknown-side ₹50 frames in the audit (`reports/dataset_audit.md`) |
| Back-side examples | PASS | scans + hand-held + webcam backs |
| Required negative denominations | PASS | ₹10/20/100/200/500: 40–50 images each across train/val/test |
| Dataset from developer-collected photographs | PASS, with disclosure | 117 phone/scanner-app captures + 621 webcam frames, all the developer's own notes; **no** downloaded RBI images used (an RBI PDF folder was supplied earlier and is excluded/gitignored). Webcam data are supplementary and flagged `source=webcam`. Whether a scanner-app capture counts as a "phone photo" is for the assessor |
| Annotation files valid | PASS | audit: 0 errors; 51 boxes colour-proposed + visually checked, 105 hand-drawn + overlay-verified |
| Training source code | PASS | `scripts/train.py`, `configs/train.yaml`; run info in `runs/*/run_info.json` (git-ignored) |
| Reproducible training | PARTIAL | seed + deterministic flag + recorded versions; not re-run bit-for-bit |
| ONNX artifact exists | PASS | `model.onnx` 12.7 MB, sha256 `ce5a6d1c…f7e99`, `model.meta.json` |
| ONNX loads in OpenCV | PASS | validate PASS; OpenCV 5.0.0 |
| OpenCV inference implemented | PASS | `cv2.dnn` only |
| Root `test.py`, mandatory command | PASS | run on the webcam |
| Box + `₹50` label + confidence | PASS | rendering inspected; developer saw detections live on a phone-screen ₹50 |
| Webcam released correctly | PASS | `finally` blocks, tests, real runs exited cleanly |
| README explains install/run | PASS | |
| Public GitHub repository | PARTIAL | local git repo with ~25 descriptive commits authored by `sasi-04`; remote configured; latest history **not yet pushed** (needs the developer's go-ahead / force-push) |

## Model quality

| item | status | evidence |
|---|---|---|
| Validation separate from training | PASS | session-level splits; leakage audit (256-bit dHash) |
| Held-out evaluation exists | PASS | val 200 / test 203 images |
| Front-side performance measured | PASS: 4/8 on test | `reports/eval_test.md` |
| Back-side performance measured | PASS: 6/6 on test | same |
| Each required negative denomination evaluated | PASS | all five at 0 false ₹50 on val and test |
| Unrelated objects evaluated | PASS | no-note frames 0/38 (test), plus window/blue-green clothing; live 0/200 frames |
| FP / FN counts reported | PASS | EVALUATION.md |
| Threshold selection documented | PASS | 0.46 on val; rule and its revision documented |
| Latency measured | PASS | ≈ 83 ms forward / 87 ms pipeline median on this CPU |
| Limitations documented | PASS | LIMITATIONS.md, MODEL_CARD.md |

## Engineering quality

All PASS: dataset/annotation validators, 115 tests, model-load and camera error handling (simulated), Windows paths,
dependencies, no fabricated metrics, `test.py` independent of any dashboard. Not built (optional): second-stage verifier, model
comparison, web dashboard. Not verified: Linux/macOS; a physical camera unplug.

## What the developer should still do

1. Run `python test.py --model model.onnx` with your real notes in many poses; note misses/false alarms (`i` inspector, `+`/`-`).
2. If one pose keeps failing, record that pose with `scripts/collect_webcam.py` (stage `--stages 1` etc.), box, retrain
   (`train.py --init runs/rs50_final/weights/best.pt`), re-select the threshold on a *new* validation session.
3. Publish: review `dataset/images` (faces and bystanders in webcam frames), then push to `sasi-04/Train-a-50-Note-Detector`.

## Demonstrating to a reviewer

`python test.py --model model.onnx --inspector`; show the new ₹50 (both sides, held steady and well lit), then other notes and a
plain card; press `i` to show why something is rejected; open `docs/EVALUATION.md` and be upfront about the sample size and the
64 % recall on the held-out session.
