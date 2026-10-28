# Acceptance report

Machine: Windows 10, Intel i5-7300U (2 cores / 4 threads), Python 3.14.4, OpenCV 5.0.0 (portability also checked on 4.14.0),
one webcam (index 0). Date: 2026-10-28.

**Verdict: complete, runnable and measured; reliability on a reviewer's webcam is still not guaranteed.** Every mandatory
deliverable exists and was verified. On a held-out webcam session the evaluation model showed a green ₹50 on **33 of 36 (92 %)**
real ₹50 frames, labelled **0 of 157** non-₹50 frames as ₹50, and drew a red "Other note" box on 84 of 119 other-note frames.
The shipped model was then fine-tuned on all data, so it has no held-out score; the live check with real notes in many poses is
the developer's to repeat on the reviewing machine.

Status: PASS · PARTIAL · FAIL · BLOCKED · NOT TESTED.

## Test evidence (actually run)

| run | result |
|---|---|
| `pytest -q` (OpenCV 5.0.0) | **125 passed** (includes loading the real 2-class `model.onnx`) |
| `validate_onnx.py` on the shipped `model.onnx` | **PASS**: OpenCV vs ONNX Runtime max score diff 3.3e-6; vs PyTorch max box diff 0.001 px; 55/55 decoded detections matched (`reports/onnx_validation.md`) |
| `python test.py --model model.onnx --no-display --max-frames 200`, real webcam, no note shown | 200 frames, **0 ₹50, 0 red boxes**, 0 camera failures; inference median 76 ms, pipeline 80 ms (≈ 12 FPS) |
| shipped model on its own training images (`reports/final_on_train/`) | 55/55 ₹50 found, 0/253 negatives labelled ₹50, other notes red on 88 %. **Optimistic (training data).** |
| rendered frames with real model output (₹20, ₹500, ₹50) | red box + red banner for the two other notes, green box + banner for the ₹50 (frames from the training data; visual check only) |
| `python test.py --model model.onnx` with the file missing | clean error, exit 3 |
| `train.py` before the dataset was ready | refused with the blockers (exit 1) |

## Held-out evaluation (evaluation model `rs50_split2`; thresholds ₹50 0.62 / other 0.82 chosen on val; see EVALUATION.md)

| held-out **test** (193 frames) | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|
| new ₹50 (36) | **33 (92 %)** | 0 | 3 |
| ₹10 (22) / ₹20 (29) / ₹100 (19) / ₹200 (28) / ₹500 (21) | **0 / 0 / 0 / 0 / 0** | 14 / 23 / 11 / 23 / 13 (84 of 119) | 8 / 6 / 8 / 5 / 8 |
| no note (38) | 0 | 0 | 38 |

Non-₹50 images labelled ₹50: **0 of 157** (95 % CI upper bound 2.4 %). Validation: ₹50 27/29, 0/159 false, other notes red 108/123.
NOT TESTED on held-out data: dim light and angle as separate groups, several notes in one frame, a second copy of any note.

## Assignment compliance

| item | status | evidence |
|---|---|---|
| Correct target: new ₹50 | PASS | class 0, label `₹50` |
| Front-side examples | PASS | 28 front + 75 side-unknown ₹50 images (`reports/dataset_audit.md`) |
| Back-side examples | PASS | 17 back-labelled + unknown-side webcam frames |
| Required negative denominations | PASS | ₹10/20/100/200/500: 85–88 images each across train/val/test, as class 1 "other note" |
| Dataset from developer-collected photographs | PASS, with disclosure | 117 phone/scanner-app captures + 572 webcam frames, all the developer's own notes; **no** downloaded RBI images used (an RBI PDF folder was supplied earlier and is excluded/gitignored). Webcam data are supplementary and flagged `source=webcam`. Whether a scanner-app capture counts as a "phone photo" is for the assessor |
| Annotation files valid | PASS | audit: 0 errors; 120 ₹50 boxes (51 colour-proposed + checked, 69 hand-drawn + overlay-verified); 430 other-note boxes (162 hand-drawn, 268 helper-proposed and individually reviewed) |
| Training source code | PASS | `scripts/train.py`, `configs/train.yaml`; run info in `runs/*/run_info.json` (git-ignored) |
| Reproducible training | PARTIAL | seed + deterministic flag + recorded versions; not re-run bit-for-bit (the final run was resumed after an interruption) |
| ONNX artifact exists | PASS | `model.onnx` 12.7 MB, sha256 `ee2a0455…f7b04b`, `model.meta.json` |
| ONNX loads in OpenCV | PASS | validate PASS; OpenCV 5.0.0 |
| OpenCV inference implemented | PASS | `cv2.dnn` only |
| Root `test.py`, mandatory command | PASS | run on the webcam |
| Box + `₹50` label + confidence | PASS | developer saw `₹50 0.93` live; red `Other note` box added |
| Webcam released correctly | PASS | `finally` blocks, tests, real runs exited cleanly |
| README explains install/run | PASS | |
| Public GitHub repository | PARTIAL | local git repo with ~55 descriptive commits authored by `sasi-04`; remote configured; latest history **not yet pushed** (needs the developer's go-ahead / force-push) |

## Model quality

| item | status | evidence |
|---|---|---|
| Validation separate from training | PASS | session-level splits; leakage audit (256-bit dHash) |
| Held-out evaluation exists | PASS | val 188 / test 193 images |
| Front-side performance measured | PASS: 8/8 on test | `reports/eval_test.md` |
| Back-side performance measured | PASS: 5/6 on test | same |
| Each required negative denomination evaluated | PASS | all five at 0 shown as ₹50 on val and test |
| Unrelated objects evaluated | PASS | no-note frames 0/38 (test), window/blue-green clothing included; live 0/200 frames |
| FP / FN counts reported | PASS | EVALUATION.md |
| Threshold selection documented | PASS | 0.62 / 0.82 on val; rule documented |
| Latency measured | PASS | ≈ 76 ms forward / 80 ms pipeline median live; 66 ms offline (this CPU) |
| Limitations documented | PASS | LIMITATIONS.md, MODEL_CARD.md |

## Engineering quality

All PASS: dataset/annotation validators (two-class aware), 125 tests, model-load and camera error handling (simulated), Windows
paths, dependencies, no fabricated metrics, `test.py` independent of any dashboard. Not built (optional): second-stage verifier,
model comparison, web dashboard, naming the denomination in the red box. Not verified: Linux/macOS; a physical camera unplug.

**Corrections made during this work** (kept for honesty): an earlier version of these docs stated 156 boxed ₹50 images (105 webcam);
the audit shows 120 (69 webcam). Fixed everywhere. A helper training run had silently merged the two classes (checkpoint
`single_cls`); `train.py` now forces two classes and refuses a non-2-class result.

## What the developer should still do

1. Run `python test.py --model model.onnx` with your real notes in many poses; note misses/false alarms (`i` inspector, `+`/`-`).
2. If one pose keeps failing, record it with `scripts/collect_webcam.py`, box it, retrain
   (`train.py --init runs/rs50_final2/weights/best.pt`), re-select the thresholds on a *new* validation session.
3. Publish: review `dataset/images` (faces and bystanders in webcam frames), then push to `sasi-04/Train-a-50-Note-Detector`.

## Demonstrating to a reviewer

`python test.py --model model.onnx --inspector`; show the new ₹50 (both sides, held steady and well lit), then other notes (red
box) and a plain card (nothing); press `i` to show why something is rejected; open `docs/EVALUATION.md` and be upfront about the
sample size and what was held out.
