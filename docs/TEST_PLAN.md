# Test plan

Run: `pytest -q` (≈30 s on the dev laptop). Install with `pip install -r requirements-dev.txt`; the parity tests also need
`requirements-train.txt` and are skipped otherwise.

## What each layer proves (and does not)

| layer | files | uses | proves | does **not** prove |
|---|---|---|---|---|
| Unit | `test_preprocessing`, `test_postprocessing`, `test_threshold_logic`, `test_metrics`, `test_annotations`, `test_visualization` | pure functions | geometry, decoding, NMS, thresholds, temporal rule, metric maths, label validation | model quality |
| Simulated hardware | `test_webcam_error_handling` | fake `VideoCapture` | retry/reopen/give-up, release on failure, unreadable sources | a real unplugged camera |
| Fixture ONNX | `test_model_loading`, `test_cli` | tiny ONNX graph emitting a known tensor, loaded by `cv2.dnn` | OpenCV loading, end-to-end coordinate mapping, exit codes, headless run, subprocess `test.py` | detection accuracy |
| Dataset tooling | `test_dataset_validation` | tiny synthetic images in temp dirs | missing/invalid labels, corrupt/blank/duplicate images, leakage, session splits, coverage blockers | real dataset adequacy |
| Export parity (slow) | `test_onnx_parity` | random-init YOLOv8n exported to ONNX | PyTorch ≡ OpenCV raw tensors (≤ 0.5 px / 2e-3) and decoded boxes (IoU ≥ 0.9), wrong-`imgsz` error, `validate_onnx.py` | accuracy |
| Real model | `test_onnx_parity::test_real_model_artifact_loads_and_runs` | repo-root `model.onnx` | loads + runs | **SKIPPED until a real model exists** |

## Manual / hardware checks (record results in ACCEPTANCE_REPORT)

* Webcam, windowed: `python test.py --model model.onnx` — box/label/confidence/banner, `q` exits, camera light turns off.
* Hold the new ₹50 front, back, tilted, dim; hold each other denomination; hold a non-note rectangle; leave the frame.
* Unplug the camera mid-run → "camera lost, reconnecting" then a clear error, no traceback.
* `--inspector`: rejected candidates show raw score + reason.
* Evaluation scripts on real data (README steps 6) with 0 `NOT TESTED` scenarios.
