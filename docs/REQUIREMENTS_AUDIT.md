# Requirements audit

**Baseline (start of this work):** `D:\ps3` was an empty, non-git directory. No source, dataset, model, README, tests
or assignment PDF existed, so the original assignment text was taken from the brief supplied in the request. Every
requirement therefore started as **FAIL**. The table shows the status **now**, with evidence from commands actually run.
Environment: Windows 10, Python 3.14.4, OpenCV 5.0.0 (also OpenCV 4.14.0 for the portability check), Intel i5-7300U (2 cores,
4 threads), one working webcam (index 0, 640×480 default).

Status key: **PASS** verified by a run · **PARTIAL** implemented, not fully verifiable yet · **BLOCKED** needs the developer's real photographs / a trained model · **FAIL** not done.

| ID | Requirement | Current implementation | Evidence | Status | Missing work | Proposed fix | Verification |
|---|---|---|---|---|---|---|---|
| R1 | Training photos taken by the developer with their own phone and genuine notes | `capture_dataset.py ingest` (phone folders, `source=phone`), webcam mode flagged `source=webcam`, auditor warns if data is all-webcam | `tests/test_dataset_validation.py`; synthetic dry run | **BLOCKED** | No photographs exist | Developer follows `docs/DATASET_COLLECTION.md` | `validate_dataset.py` → READY |
| R2 | New ₹50 photographed on both sides | Manifest `side` field; auditor blocks training without ≥1 front and ≥1 back; warns < 100 each | `test_missing_back_side_and_negatives_block` | **BLOCKED** | Photos | Collect | audit "Target front / back" |
| R3 | ₹10/20/100/200/500 photographed as negatives | Required denominations enforced by auditor and `train.py` gate; each reported separately in evaluation | `test_missing_back_side_and_negatives_block` | **BLOCKED** | Photos | Collect | audit + `evaluate.py` per-denomination table |
| R4 | No downloaded banknote images | No code fetches images. Only network use: Ultralytics downloads COCO-pretrained `yolov8n.pt` (no banknote data) | code review; smoke training fetched only that file | **PASS** (by design) | – | Disclosed in README/model card | `grep` for downloads; manifest `source` |
| R5 | No generated banknote images | Synthetic images exist only in pytest temp dirs and a scratch smoke run (plain coloured rectangles); none under `dataset/` | `dataset/` is empty | **PASS** | – | – | `dataset/` contents |
| R6 | No printing/photocopying/counterfeits | Forbidden in the collection guide; nothing in code creates note images | docs | **PASS** | – | – | review |
| R7 | YOLO recommended | YOLOv8n (Ultralytics) | `scripts/train.py`, smoke run completed 3 epochs | **PASS** (code); trained banknote model **BLOCKED** | Data | – | – |
| R8 | Export to ONNX | `export_onnx.py` (opset 12, fixed size, sidecar metadata) | exported random-init and smoke models; `[1,5,N]` output | **PARTIAL** | A real `model.onnx` from real photos | run steps 4–5 of README | `validate_onnx.py` |
| R9 | ONNX loads in an OpenCV pipeline | `OnnxDetector` uses `cv2.dnn.readNetFromONNX` only | `validate_onnx.py` PASS under OpenCV 5.0.0 and 4.14.0; PyTorch and ONNX Runtime parity ≤ 0.003 px / 8e-6 score | **PASS** (mechanics) | Re-run on the real model | – | `validate_onnx.py` |
| R10 | Root `test.py` opens a webcam and detects | `test.py` → `detector.app` | 30-frame headless run and 40-frame windowed run on the real webcam with a throwaway model, exit 0 | **PARTIAL** | Needs the real model to demonstrate detection of a real note | – | manual demo |
| R11 | Public GitHub repository | Not a git repo yet; `.gitignore` prepared | – | **BLOCKED** | User action (create + push) | See acceptance report §publishing | repo URL opens logged-out |
| R12 | README explains install and execution | `README.md` | reviewed | **PASS** | Add real results after evaluation | – | read-through |
| R13 | `python test.py --model model.onnx` supported | `--model` default `model.onnx`; missing file → clean error exit 3 | `test_help_and_mandatory_command_via_subprocess` | **PARTIAL** | `model.onnx` absent | train + export | run command |
| R14 | Box, label `₹50`, confidence displayed | Pillow-rendered `₹`; box + `₹50 0.87` | rendered frame inspected visually | **PASS** (rendering) | Confirm on live detection | – | demo |
| R15 | Avoid other notes/objects | Single-class head, negatives as empty labels, validated threshold, stress test | mechanics tested only | **BLOCKED** | Needs real data + evaluation | – | `evaluate.py`, `stress_test.py` |
| R16 | Reject below threshold; no ₹50 for rejected/other class | `decide()` | `tests/test_threshold_logic.py` | **PASS** | – | – | – |
| R17 | Searching/no-target state | grey banner | `test_headless...` + visual | **PASS** | – | – | – |
| R18 | Handle camera failures, lighting changes, clean exit | `FrameSource` retry/reopen/give-up, `q`/Esc/window-close, `finally` releases camera and windows | `tests/test_webcam_error_handling.py` (fake cameras) | **PASS** (simulated) | Physical disconnect not tried | – | unplug test by reviewer |
| R19 | Report actual timing | HUD + exit summary (median/p95), per-stage timings | measured values in summary JSON | **PASS** | – | – | – |
| R20 | README run instructions, public repo | see R11/R12 | | **PARTIAL** | | | |
| R21 | Validation separate from training; held-out test | session-level splits, leakage checks, `evaluate.py` refuses threshold selection on test | `test_cross_split_near_duplicate_flagged_as_leakage`, CLI refusal verified | **PARTIAL** | Real data | – | – |
| R22 | Threshold chosen on validation, documented | `select_threshold` + `--write-config` | smoke dry run (garbage model) | **PARTIAL** | Real validation set | – | `configs/inference.yaml` flag |
| R23 | Metrics: P/R/F1/FP/FN/confusion/per-condition/latency | `evaluate.py`, `stress_test.py` | ran on synthetic smoke data | **PARTIAL** | Real data | – | reports/ |
| R24 | Automated tests | 111 pytest tests | see TEST_PLAN | **PASS** | – | – | `pytest -q` |
| R25 | Reproducibility records | `run_info.json`, `model.meta.json`, eval reports with hashes | smoke run produced them | **PASS** (mechanism) | – | – | – |
| R26 | Optional dashboard | not built | – | **NOT DONE (optional)** | – | Only if time allows after real results | – |
