# Acceptance report

Machine: Windows 10, Intel i5-7300U (2 cores / 4 threads), Python 3.14.4, OpenCV 5.0.0 (portability also checked on 4.14.0),
one webcam (index 0). Date: 2026-10-28.

**Verdict: complete, runnable and measured; reliability on a reviewer's webcam is still not guaranteed.** Every mandatory
deliverable exists and was verified. On the held-out test (259 frames) the evaluation model showed a green ₹50 on **67 of 68**
real ₹50 frames (**32 of 32 upright**, up from 4 of 32 for the previous model), but it also labelled **8 of 191 (4.2 %)** non-₹50
frames as ₹50 (6 full-frame ₹200 scan crops, 2 ceiling frames; the previous model had 0 of 191), and it drew a red "Other note"
box on 138 of 153 other-note boxes. The shipped model was then fine-tuned on all data, so it has no held-out score; the live check
with real notes in many poses is the developer's to repeat on the reviewing machine.

Status: PASS · PARTIAL · FAIL · BLOCKED · NOT TESTED.

## Test evidence (actually run)

| run | result |
|---|---|
| `pytest -q` (OpenCV 5.0.0) | **127 passed** (includes loading the real 2-class `model.onnx`) |
| `validate_onnx.py` on the shipped `model.onnx` | **PASS**: OpenCV vs ONNX Runtime max score diff 1.7e-6; vs PyTorch max box diff 0.002 px; 66/66 decoded detections matched (`reports/onnx_validation.md`) |
| `python test.py --model model.onnx --no-display --max-frames 200`, real webcam, no note shown | 200 frames, **0 ₹50, 0 red boxes**, 0 camera failures; inference median 64 ms, pipeline 68 ms (≈ 15 FPS) |
| shipped model on its own training split (`reports/final_on_train/`) | 101/101 ₹50 found, 0/253 negatives labelled ₹50. **Optimistic (training data).** |
| rendered frames with real model output (₹20, ₹500, ₹50) | red box + red banner for the two other notes, green box + banner for the ₹50 (frames from the training data; visual check only) |
| `python test.py --model model.onnx` with the file missing | clean error, exit 3 |
| `train.py` before the dataset was ready | refused with the blockers (exit 1) |

## Held-out evaluation (evaluation model `rs50_split3`; thresholds ₹50 0.27 / other 0.37 chosen on val; see EVALUATION.md)

| held-out **test** (259 frames) | shown ₹50 (green) | shown Other note (red) | shown nothing |
|---|---|---|---|
| new ₹50 (68; 32 of them upright, session 5) | **67 (99 %)** (upright 32/32) | 0 | 1 |
| ₹10 (22) / ₹20 (63) / ₹100 (19) / ₹200 (28) / ₹500 (21) | 0 / 0 / 0 / **6** / 0 | 19 / 62 / 17 / 21 / 19 | 3 / 1 / 2 / 1 / 2 |
| no note (38) | **2** | 0 | 36 |

Non-₹50 images labelled ₹50: **8 of 191 (4.2 %)**, 95 % CI upper bound 8.0 % (previous model: 0 of 191 on the same test). The 8 are 6
close-up ₹200 scan crops filling the whole image and 2 webcam ceiling frames; scores 0.65–0.85, so no threshold fixes them and none
was retuned on the test set. Validation: ₹50 29/29, 0/159 false, other notes red 119/123.
NOT TESTED on held-out data: dim light and angle as separate groups, several notes in one frame, a second copy of any note, the
shipped (final-fit) model on any unseen frame.

## Assignment compliance

| item | status | evidence |
|---|---|---|
| Correct target: new ₹50 | PASS | class 0, label `₹50` |
| Front-side examples | PASS | 28 front + 75 side-unknown ₹50 images (`reports/dataset_audit.md`) |
| Back-side examples | PASS | 17 back-labelled + unknown-side webcam frames |
| Required negative denominations | PASS | ₹10/20/100/200/500: 85–88 images each across train/val/test, as class 1 "other note" |
| Dataset from developer-collected photographs | PASS, with disclosure | 117 phone/scanner-app captures + 684 webcam frames, all the developer's own notes; **no** downloaded RBI images used (an RBI PDF folder was supplied earlier and is excluded/gitignored). Webcam data are supplementary and flagged `source=webcam`. Whether a scanner-app capture counts as a "phone photo" is for the assessor |
| Annotation files valid | PASS | audit: 0 errors; 198 ₹50 boxes (51 colour-proposed + checked, 147 hand-drawn + overlay-verified); 464 other-note boxes (196 hand-drawn, 268 helper-proposed and individually reviewed) |
| Training source code | PASS | `scripts/train.py`, `configs/train.yaml`; run info in `runs/*/run_info.json` (git-ignored) |
| Reproducible training | PARTIAL | seed + deterministic flag + recorded versions; not re-run bit-for-bit (the final run was resumed after an interruption) |
| ONNX artifact exists | PASS | `model.onnx` 12.7 MB, sha256 `93a630e8…`, `model.meta.json` (final-fit model `rs50_final3`) |
| ONNX loads in OpenCV | PASS | validate PASS; OpenCV 5.0.0 |
| OpenCV inference implemented | PASS | `cv2.dnn` only |
| Root `test.py`, mandatory command | PASS | run on the webcam |
| Box + `₹50` label + confidence | PASS | developer saw `₹50 0.93` live; red `Other note` box added |
| Webcam released correctly | PASS | `finally` blocks, tests, real runs exited cleanly |
| README explains install/run | PASS | |
| Public GitHub repository | PARTIAL | local git repo with 60+ descriptive commits, all authored by `sasi-04`; remote configured; the last commits (session 4/5 data, the new model, docs) are **not yet pushed** (needs the developer's go-ahead; plain `git push origin main`) |

## Model quality

| item | status | evidence |
|---|---|---|
| Validation separate from training | PASS | session-level splits; leakage audit (256-bit dHash) |
| Held-out evaluation exists | PASS | val 188 / test 259 images |
| Front-side performance measured | PASS: 8/8 on test | `reports/eval_test.md` |
| Back-side performance measured | PARTIAL: 3/6 on test (too few) | same |
| Upright (portrait) ₹50 measured | PASS: 32/32 on a session recorded after training (previous model 4/32) | `reports/eval_test.md`, `reports/split3_model/baseline_split2_same_test/` |
| Each required negative denomination evaluated | PARTIAL | ₹10/20/100/500 at 0 shown as ₹50 on val and test; **₹200: 6 of 28 test images shown as ₹50** (full-frame scan crops) |
| Unrelated objects evaluated | PARTIAL | no-note frames: 2 of 38 test frames shown as ₹50 (ceiling); window/blue-green clothing 0; live 0/200 frames |
| FP / FN counts reported | PASS | EVALUATION.md |
| Threshold selection documented | PASS | 0.27 / 0.37 on val; rule documented; not changed after seeing test |
| Latency measured | PASS | ≈ 64 ms forward / 68 ms pipeline median live (this CPU) |
| Limitations documented | PASS | LIMITATIONS.md, MODEL_CARD.md |

## Engineering quality

All PASS: dataset/annotation validators (two-class aware), 127 tests, model-load and camera error handling (simulated), Windows
paths, dependencies, no fabricated metrics, `test.py` independent of any dashboard. Not built (optional): second-stage verifier,
model comparison, web dashboard, naming the denomination in the red box. Not verified: Linux/macOS; a physical camera unplug.

**Corrections made during this work** (kept for honesty): an earlier version of these docs stated 156 boxed ₹50 images (105 webcam);
the audit shows 120 (69 webcam). Fixed everywhere. A helper training run had silently merged the two classes (checkpoint
`single_cls`); `train.py` now forces two classes and refuses a non-2-class result.

## What the developer should still do

1. Run `python test.py --model model.onnx` with your real notes in many poses; note misses/false alarms (`i` inspector, `+`/`-`).
2. If one pose keeps failing, record it with `scripts/collect_webcam.py`, box it, retrain
   (`train.py --init runs/rs50_final3/weights/best.pt`), re-select the thresholds on a *new* validation session.
3. Publish: review `dataset/images` (faces and bystanders in webcam frames), then push to `sasi-04/Train-a-50-Note-Detector`.

## Demonstrating to a reviewer

`python test.py --model model.onnx --inspector`; show the new ₹50 (both sides, held steady and well lit), then other notes (red
box) and a plain card (nothing); press `i` to show why something is rejected; open `docs/EVALUATION.md` and be upfront about the
sample size and what was held out.
