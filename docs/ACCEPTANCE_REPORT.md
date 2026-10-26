# Acceptance report

Machine: Windows 10, Intel i5-7300U (2 cores / 4 threads), Python 3.14.4, OpenCV 5.0.0 (portability also checked on 4.14.0),
one webcam (index 0). Date: 2026-10-26.

**Verdict: complete and runnable; reliability on a reviewer's webcam is UNPROVEN and the dataset is small.** Every mandatory
deliverable exists and was verified mechanically. Detection quality was measured on held-out images with an *evaluation* model
(mixed results, below), and the shipped model was then retrained on all data, so it has no held-out score. A live positive
test (holding real notes up to the camera) was not possible from the tooling and is the remaining check for the developer.

Status: PASS · PARTIAL · FAIL · BLOCKED · NOT TESTED.

## Test evidence (actually run)

| run | result |
|---|---|
| `pytest -q` (OpenCV 5.0.0) | **112 passed** (includes loading the real `model.onnx`) |
| `pytest` minus parity file (separate venv, OpenCV 4.14.0) | 106 passed (before the real model was added) |
| `validate_onnx.py` on the shipped `model.onnx` | **PASS**: OpenCV vs ONNX Runtime max score diff 1.4e-6; vs PyTorch max box diff 0.0009 px; 107/107 decoded detections matched (`reports/onnx_validation.md`) |
| `python test.py --model model.onnx --no-display --max-frames 150` on the real webcam, no note shown | 150 frames, **0 accepted ₹50**, 0 camera failures; inference median 87 ms, full pipeline median 91 ms (≈ 11 FPS) on this CPU |
| shipped model on its own training images (`reports/final_on_train/`) | 24/24 targets, 0/29 false ₹50. **Optimistic (training data).** |
| `python test.py --model model.onnx` with the file missing | clean error, exit 3 |
| `train.py` before the dataset was ready | refused with the blockers (exit 1) |

## Held-out evaluation (evaluation model `rs50_split`, threshold 0.54 chosen on val; see EVALUATION.md)

| split | targets found | false-₹50 negative images |
|---|---|---|
| val (13 target, 22 negative) | 13/13 | 0/22 (CI upper 14.9 %) |
| **test** (14 target, 15 negative) | **7/14**: back 6/6, **front 1/8** | **1/15** (a ₹20), CI upper 29.8 % |

NOT TESTED on held-out data: dim light, angled notes, ₹100/₹500/objects on the test split, any other note held in hand.

## Assignment compliance

| item | status | evidence |
|---|---|---|
| Correct target: new ₹50 | PASS | single class, label `₹50` |
| Front-side examples | PASS | 28 front (below the recommended 100 per side), 6 side-unknown target images (`reports/dataset_audit.md`) |
| Back-side examples | PASS | 17 (scan backs + hand-held backs); below the recommended 100 per side |
| Required negative denominations | PASS (thin) | ₹10/20/100/200/500: one physical note each, both sides, 4–12 images per side |
| Dataset from developer-collected photographs | PASS, with disclosure | the developer supplied scanner-app/phone captures of their own notes; **no** downloaded RBI images were used (an RBI PDF folder was supplied earlier and excluded/gitignored). Whether a scanner-app capture counts as a "phone photo" is for the assessor |
| Annotation files valid | PASS | audit: 0 errors; 51 boxes colour/segmentation-proposed then visually reviewed (not hand-drawn) |
| Training source code | PASS | `scripts/train.py`, `configs/train.yaml`, run info in `runs/*/run_info.json` (ignored by git) |
| Reproducible training | PARTIAL | seed + deterministic flag + recorded versions; two full runs completed; not re-run to compare bit-for-bit |
| ONNX artifact exists | PASS | `model.onnx` 12.7 MB, sha256 `c4cc2902…ba6c4`, `model.meta.json` |
| ONNX loads in OpenCV | PASS | validate PASS; OpenCV 5.0.0 |
| OpenCV inference implemented | PASS | `cv2.dnn` only |
| Root `test.py`, mandatory command | PASS (mechanics) | run on the webcam |
| Box + `₹50` label + confidence | PASS (rendering) | frame render inspected; live detection of a real note NOT verified here |
| Webcam released correctly | PASS | `finally` blocks, tests, real runs exited cleanly |
| README explains install/run | PASS | |
| Public GitHub repository | PARTIAL | local git repo committed; remote configured; **not pushed** (needs the developer's OK) |

## Model quality

| item | status | evidence |
|---|---|---|
| Validation separate from training | PASS | whole-cluster/session splits; audit found and removed one leak |
| Held-out evaluation exists | PASS (small) | val 35 / test 29 images |
| Front-side performance measured | PASS: **poor on unseen views** (1/8) | `reports/eval_test.md` |
| Back-side performance measured | PASS (6/6) | same |
| Each required negative denomination evaluated | PARTIAL | ₹10, ₹20, ₹200 on test; ₹100 (0/8), ₹500 (0/5) on val only |
| Unrelated objects evaluated | PARTIAL | 1 background frame in val; 150 live webcam frames (0 false alarms) |
| FP / FN counts reported | PASS | EVALUATION.md |
| Threshold selection documented | PASS | 0.54, plateau 0.45–0.62 on val; narrow |
| Latency measured | PASS | ≈ 87 ms forward / 91 ms pipeline median on this CPU |
| Limitations documented | PASS | LIMITATIONS.md, MODEL_CARD.md |

## Engineering quality

All PASS: dataset/annotation validators, 112 tests, model-load and camera error handling (simulated), Windows paths,
dependencies, no fabricated metrics, no placeholder presented as complete, `test.py` independent of any dashboard.
Not built (optional): second-stage verifier, model comparison, web dashboard. Not verified: Linux/macOS; a physical camera
unplug.

## What the developer should still do

1. Run `python test.py --model model.onnx` with your real notes: new ₹50 front/back (flat, tilted, sideways, dim), and each
   other denomination. Note any misses/false alarms; use `i` for the inspector and `+`/`-` to nudge the threshold.
2. If results disappoint, add varied phone photos (dim light, tilt, cluttered desk, other notes in hand, objects), then:
   import → `split_dataset.py`/manual yaml → `validate_dataset.py` → `train.py` → `export_onnx.py` →
   `evaluate.py --split val --select-threshold --write-config`.
3. Publish: review `dataset/images` (faces; some hand-held frames show a lanyard, and wider shots may show an ID card),
   then `git push -u origin main` to `sasi-04/Train-a-50-Note-Detector`.

## Demonstrating to a reviewer

`python test.py --model model.onnx --inspector`; show the new ₹50 (both sides), then other notes and a plain card; press `i` to
show why something is rejected; open `docs/EVALUATION.md` and be upfront about the sample size and the front-side miss on unseen
views.
