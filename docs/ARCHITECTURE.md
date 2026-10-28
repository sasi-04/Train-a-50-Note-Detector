# Architecture

```
phone photos ─► capture_dataset.py ingest ─► annotate.py ─► split_dataset.py (by session) ─► validate_dataset.py
                                                                                              │ (gate)
                                   train.py (YOLOv8n, CPU) ◄──────────────────────────────────┘
                                        │ best.pt (val-selected)
                              export_onnx.py ─► model.onnx + model.meta.json
                                        │
                              validate_onnx.py  (OpenCV vs ONNX Runtime vs PyTorch)
                                        │
        evaluate.py --split val --select-threshold ─► configs/inference.yaml   stress_test.py
        evaluate.py --split test (frozen threshold) ─► reports/
                                        │
                      test.py ─► detector.app ─► FrameSource ─► OnnxDetector (cv2.dnn) ─► decide() ─► draw
```

`test.py` depends only on `src/detector` + OpenCV/NumPy/PyYAML/Pillow. No server, no dashboard, no PyTorch.

## Modules (`src/detector/`)

| module | responsibility |
|---|---|
| `preprocessing.py` | letterbox (pad 114), BGR→RGB, /255, NCHW float32; strict input validation |
| `postprocessing.py` | detect output layout from shape, decode cx,cy,w,h + class scores, score floor, class-aware NMS, map back to original-frame pixels (undo pad/scale), clip, drop degenerate boxes |
| `inference.py` | `OnnxDetector`: `cv2.dnn.readNetFromONNX`, warm-up forward (also validates input size), per-stage timings, explicit `ModelLoadError` / `InferenceError` |
| `thresholding.py` | `decide()` → accepted / below_threshold / wrong_class / invalid_box; `TemporalConfirmer` (k-of-m + hysteresis, off by default); `select_threshold()` |
| `webcam.py` | `FrameSource`: camera (DirectShow→MSMF→any on Windows), retry, reopen, give up with a clear error; also image/video files |
| `visualization.py` | box/label/HUD/inspector; Pillow renders `₹` because `cv2.putText` cannot; cached label chips |
| `app.py` | CLI, main loop, exit codes, timing statistics, clean shutdown |
| `dataset.py` | manifest I/O, YOLO label validation, auditor (dHash near-duplicates, session leakage, coverage) |
| `metrics.py` | IoU matching, threshold sweep, AP, per-group stats, Wilson intervals |

## Verified model-I/O contract (Ultralytics YOLOv8n export; the shipped model has 2 classes, 1-class models also load)

| item | value | where verified |
|---|---|---|
| input | `images` float32 `(1,3,S,S)`, RGB, 0–1, letterbox pad 114 | `validate_onnx.py` graph check + PyTorch parity |
| output | `output0` float32 `(1, 4+nc, N)`: rows `cx,cy,w,h,score₀[,score₁]` (nc = 2 shipped); N = 8400 @640, 4725 @480, 2100 @320 | exported shapes |
| scores | sigmoid already applied, in [0,1] | `validate_onnx.py` range check |
| NMS | **not** embedded; done in `postprocessing.nms` (IoU 0.45, class-agnostic: one box per note) | PyTorch decoded-detection parity |
| opset | 12, fixed size, FP32 | `export_onnx.py` |

Other layouts (`(1,N,4+nc)`, end-to-end `(1,N,6)`) are decoded too and unit-tested, but only the first has been
exercised with a real exported network.

## Key decisions

* **YOLOv8n, not YOLO11 / larger.** Plain conv/SiLU/C2f graph that OpenCV's DNN importer handles at opset 12
  (verified on OpenCV 5.0.0; 4.12 check recorded in the acceptance report); 3 M parameters, 12 MB ONNX.
* **Two classes.** Front and back are both class 0 `fifty_rupee_note`. All other denominations are class 1 `other_note` (red
  box): the detector learns "a banknote that is not a ₹50" explicitly instead of treating other notes as empty background, and the
  app can show it. Green ₹50 wins over red when both fire; the red box never names the denomination. No-note images keep empty
  labels. The evaluation reports each denomination separately.
* **Input size 480 by default.** Forward-pass time on the 2-core dev laptop (random-init yolov8n, median of 20, OpenCV 5.0):
  640 → ~115 ms, 480 → ~62 ms, 416 → ~47 ms, 320 → ~24 ms. 480 was chosen for a usable frame rate; `train.py --imgsz 640` is
  supported and the choice should be revisited with real validation numbers.
* **Sessions are the unit of splitting**, enforced by the auditor (a session spanning splits is an error; cross-split
  near-duplicates are an error).
* **Threshold chosen on validation only**; `evaluate.py` refuses `--select-threshold` on test/train.
* **Temporal confirmation exists but is off by default** and never counted in metrics (evaluation uses raw per-frame decisions).
* **Not built:** second-stage verifier (only justified if the baseline fails the stress test on real data), model
  comparison (needs real data), web dashboard (optional; not needed for any requirement).
* **Augmentation:** Ultralytics built-ins only, tuned for plausibility (tiny hue shift because colour is a class cue,
  no flips because mirrored notes never occur). Blur/noise/shadow via Albumentations is deliberately not used: it pulls in
  `opencv-python-headless`, which conflicts with the GUI `opencv-python` that `test.py` needs.
