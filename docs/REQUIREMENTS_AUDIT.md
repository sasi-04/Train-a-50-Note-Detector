# Requirements audit

**Baseline:** the project directory was empty (no code, data, model, README or tests) when work began, so every requirement
started as FAIL. The assignment PDF (`ZED_Digital_PS3_Rs50_Note_Detector.pdf`) matches the requirements below.
The table gives the **current** status with evidence; see `ACCEPTANCE_REPORT.md` for the full checklist.
Environment: Windows 10, Python 3.14.4, OpenCV 5.0.0 (and 4.14.0), i5-7300U, webcam index 0.

| ID | Requirement (PDF) | Implementation | Evidence | Status | Remaining |
|---|---|---|---|---|---|
| R1 | Own phone photos of the ₹50, front and back | 51 target images (28 front, 17 back, 6 unknown side) from the developer's scanner-app/phone captures | `reports/dataset_audit.md` | **PASS (small)** | more varied photos recommended |
| R2 | Other notes ₹10/20/100/200/500 as negatives | one physical note each, both sides | audit | **PASS (thin)** | held-in-hand negatives |
| R3 | No downloaded note images | none used; an RBI-ad PDF folder was supplied, declined and gitignored | `.gitignore`, review | **PASS** | – |
| R4 | No printed/photocopied/fake notes | none; no generated images in `dataset/` | review | **PASS** | – |
| R5 | Any model (YOLO recommended) | YOLOv8n | `runs/*/run_info.json` | **PASS** | – |
| R6 | Dataset + labels in `dataset/` | 117 images + YOLO labels + manifest | `dataset/` | **PASS** | review for privacy before push |
| R7 | Source code (training + test script) | `scripts/train.py`, `test.py`, `src/detector` | repo | **PASS** | – |
| R8 | Model in ONNX, loads in OpenCV | `model.onnx` (12.7 MB), `cv2.dnn` only | `validate_onnx.py` PASS; 112 tests incl. real model | **PASS** | – |
| R9 | `test.py`: webcam, box labelled ₹50 + confidence | implemented | 150-frame webcam run; rendered frame; tests | **PASS (mechanics)** | live positive test with real notes by the developer |
| R10 | README: install and run | `README.md` | review | **PASS** | – |
| R11 | `python test.py --model model.onnx` | supported | run on the webcam | **PASS** | – |
| R12 | Detect ₹50 reliably | evaluation: back 6/6, front 1/8 on unseen views; val 13/13 | `docs/EVALUATION.md` | **PARTIAL** | more data; live check |
| R13 | Do not show ₹50 for other notes/objects | threshold 0.54 on val; test 1/15 false (a ₹20); live 0/150 frames | EVALUATION.md | **PARTIAL** | more negatives |
| R14 | One public GitHub repository | local git; remote set; not pushed | `git log` | **PARTIAL** | developer's OK to push |

Engineering extras (all verified by runs/tests): dataset auditor with leakage detection, threshold selection on val only,
stress test, ONNX validator with PyTorch/ONNX-Runtime parity, reproducibility records, camera failure handling, temporal
confirmation (off by default). Not built: second-stage verifier, model comparison, dashboard.
