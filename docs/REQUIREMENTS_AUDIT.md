# Requirements audit

**Baseline:** the project directory was empty (no code, data, model, README or tests) when work began, so every requirement
started as FAIL. The assignment PDF (`ZED_Digital_PS3_Rs50_Note_Detector.pdf`) matches the requirements below.
The table gives the **current** status with evidence; see `ACCEPTANCE_REPORT.md` for the full checklist.
Environment: Windows 10, Python 3.14.4, OpenCV 5.0.0 (and 4.14.0), i5-7300U, webcam index 0.

| ID | Requirement (PDF) | Implementation | Evidence | Status | Remaining |
|---|---|---|---|---|---|
| R1 | Own phone photos of the ₹50, front and back | 120 boxed ₹50 images (51 phone/scanner, 69 webcam; front, back, sideways) from the developer's own note | `reports/dataset_audit.md` | **PASS (one physical note)** | a second copy of the note and more lighting |
| R2 | Other notes ₹10/20/100/200/500 as negatives | one physical note each, both sides, plus held in hand on webcam (40–50 images each) | audit | **PASS** | – |
| R3 | No downloaded note images | none used; an RBI-ad PDF folder was supplied, declined and gitignored | `.gitignore`, review | **PASS** | – |
| R4 | No printed/photocopied/fake notes | none; no generated images in `dataset/` | review | **PASS** | – |
| R5 | Any model (YOLO recommended) | YOLOv8n | `runs/*/run_info.json` | **PASS** | – |
| R6 | Dataset + labels in `dataset/` | 801 images + YOLO labels + manifest | `dataset/` | **PASS** | review faces/bystanders before push |
| R7 | Source code (training + test script) | `scripts/train.py`, `test.py`, `src/detector` | repo | **PASS** | – |
| R8 | Model in ONNX, loads in OpenCV | `model.onnx` (12.7 MB), `cv2.dnn` only | `validate_onnx.py` PASS; 127 tests incl. real model | **PASS** | – |
| R9 | `test.py`: webcam, box labelled ₹50 + confidence | implemented | 200-frame webcam run; rendered frame; tests | **PASS** | repeat the live positive test on the reviewing machine |
| R10 | README: install and run | `README.md` | review | **PASS** | – |
| R11 | `python test.py --model model.onnx` | supported | run on the webcam | **PASS** | – |
| R12 | Detect ₹50 reliably | held-out test: 67/68 shown as ₹50 incl. upright 32/32; front 8/8, back 3/6; AP@0.5 0.91 | `docs/EVALUATION.md` | **PASS (small sample)** | back side; a reviewer's note and camera |
| R13 | Do not show ₹50 for other notes/objects | thresholds 0.27 / 0.37 on val; held-out test **8/191 non-₹50 images labelled ₹50** (CI ≤ 8.0 %: 6 full-frame ₹200 scan crops, 2 ceiling frames), other notes get a red box (138/153); live 0/200 frames | EVALUATION.md | **PARTIAL (measured; worse than the previous model's 0/191)** | the ceiling-type false alarm; a reviewer's other notes/rooms |
| R14 | One public GitHub repository | local git with 60+ descriptive commits (`sasi-04`); remote set; the last commits are not yet pushed | `git log` | **PARTIAL** | developer's go-ahead (`git push origin main`) |

Engineering extras (all verified by runs/tests): dataset auditor with leakage detection, threshold selection on val only,
stress test, ONNX validator with PyTorch/ONNX-Runtime parity, reproducibility records, camera failure handling, temporal
confirmation (off by default). Not built: second-stage verifier, model comparison, dashboard.
