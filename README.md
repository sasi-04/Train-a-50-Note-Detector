# ₹50 Vision — Indian ₹50 banknote detection

Real-time webcam detector for the **new light blue-green Indian ₹50 note** (Mahatma Gandhi portrait on the
front, Hampi stone chariot on the back). Built for the ZED Digital assignment "Train a ₹50 Note Detector".
YOLOv8n is trained on the developer's own phone photographs, exported to **ONNX**, and run live through
**OpenCV's DNN module** (no PyTorch needed at run time).

```bash
python test.py --model model.onnx
```

> **Status of this repository (read first).** All code, tooling and tests are in place and have been run, but
> **no trained `model.onnx` exists yet** and **no banknote photographs have been collected**: those require the
> developer's real phone photos of genuine notes, which cannot be fabricated. Until they are collected, trained
> and evaluated, this README contains **no accuracy, false-positive or FPS claims for the banknote model**. The
> pipeline itself was verified with throwaway models trained on synthetic rectangles; see
> [docs/ACCEPTANCE_REPORT.md](docs/ACCEPTANCE_REPORT.md) for exactly what was and was not verified.

## What you should see when you run it

A window titled *Rs50 Vision* opens on your webcam.

* **Target note in view (front or back):** a green box with the label `₹50 0.87`, where the number is the
  detector's **confidence score** (not a calibrated probability), and a green banner *Rs.50 DETECTED*.
* **No target / another note / an object:** no box, grey banner *Searching for Rs.50 note...*.
* Top-left HUD: the active threshold (marked **NOT VALIDATED** until it was chosen on validation data),
  inference time and FPS measured live.
* Keys: `q` or `Esc` quit · `i` decision inspector (shows rejected candidates, their raw score and why they were
  rejected) · `+` / `-` change the threshold live.

Only the **new ₹50** note is the target class. Front and back are both instances of that single class. Older ₹50
notes and every other denomination are negatives. How well they are actually rejected is **only known after
you run the evaluation below on your own held-out photos** and is reported in `reports/` and
[docs/EVALUATION.md](docs/EVALUATION.md); no denomination is guaranteed to be rejected.

## Requirements

* Python 3.10+ (developed and tested on 3.14.4, Windows 10). Linux/macOS should work (code is portable) but only
  Windows has been tested.
* A webcam (or use `--source image_or_video` to run without one).
* CPU only. No CUDA, no server, no internet at run time.

## Installation (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt            # enough to run test.py
pip install -r requirements-train.txt      # only for training / export / evaluation (installs PyTorch + Ultralytics)
pip install -r requirements-dev.txt        # only for running the test suite
```

Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip` lines.

## Running the detector

```bash
python test.py --model model.onnx                 # mandatory form
python test.py --model model.onnx --threshold 0.75 --camera 1
python test.py --model model.onnx --source clip.mp4 --inspector
python test.py --model model.onnx --no-display --max-frames 100 --summary-json run.json   # headless timing run
python test.py --help
```

| flag | meaning |
|---|---|
| `--model` | ONNX file (default `model.onnx`) |
| `--threshold` | score threshold in (0,1); default from `configs/inference.yaml` |
| `--camera` / `--source` | webcam index (default 0) / image or video file instead of a webcam |
| `--imgsz` | model input size; normally read from `model.meta.json` next to the model |
| `--confirm-frames N` | require N hits in the last 5 frames before showing a box (default 1 = off) |
| `--inspector` | start with the decision inspector on |
| `--no-display`, `--max-frames`, `--summary-json`, `--log-file`, `--verbose` | headless/diagnostics |

Exit codes: 0 ok · 2 bad arguments · 3 model missing/unloadable · 4 camera problem · 5 repeated inference failure.
A failed inference is shown as an error banner and never as "no detection" or a ₹50.

**Operating threshold.** `configs/inference.yaml` ships with a *provisional* `0.50` and
`threshold_validated: false`. After training, `scripts/evaluate.py --select-threshold --write-config` chooses the
threshold on the **validation** split (lowest false-positive rate first, then highest recall) and flips that flag.
A higher threshold reduces false positives but misses more genuine notes; the trade-off curve is written to
`reports/threshold_analysis_val.png` and `reports/threshold_sweep_val.csv`.

## Complete workflow: photos → model → evaluation

Full collection protocol: [docs/DATASET_COLLECTION.md](docs/DATASET_COLLECTION.md).

**Hard rules (from the assignment):** photograph genuine notes yourself with your phone; photograph the new ₹50
on **both sides** and ₹10, ₹20, ₹100, ₹200, ₹500 as negatives; never download banknote images, never use generated
images, never print or photocopy notes. Everything below only ever reads your local dataset.

```bash
# 1. import each batch of phone photos (EXIF/GPS is stripped on import). One call per denomination/side/condition.
python scripts/capture_dataset.py ingest D:\photos\s01_front --denomination 50_new --side front --lighting bright --background plain --session s01
python scripts/capture_dataset.py ingest D:\photos\s01_back  --denomination 50_new --side back  --session s01
python scripts/capture_dataset.py ingest D:\photos\s01_n20   --denomination 20 --side front --session s01
#    ...repeat for 10, 20, 100, 200, 500 (and optionally 50_old, object, background), for 6+ different sessions

# 2. draw the box around every target note yourself (negatives get empty label files automatically)
python scripts/annotate.py --only-unlabeled

# 3. split whole sessions into train/val/test (no leakage) and audit the dataset
python scripts/split_dataset.py --seed 42
python scripts/validate_dataset.py              # -> reports/dataset_audit.md ; must say READY before training

# 4. train (CPU, YOLOv8n, COCO-pretrained start; configs/train.yaml)
python scripts/train.py

# 5. export to ONNX and validate it through the exact OpenCV path
python scripts/export_onnx.py --weights runs/rs50/weights/best.pt --out model.onnx
python scripts/validate_onnx.py --model model.onnx --weights runs/rs50/weights/best.pt --images dataset/images/pool

# 6. choose the threshold on VAL, stress-test negatives, then the one-time held-out TEST report
python scripts/evaluate.py --model model.onnx --split val --select-threshold --write-config
python scripts/stress_test.py --model model.onnx --split val
python scripts/evaluate.py --model model.onnx --split test

# 7. demo
python test.py --model model.onnx
```

`train.py` refuses to start until the audit passes (both sides of the target, every required negative
denomination, valid labels, leak-free session splits). Pretrained weights (`yolov8n.pt`, trained on the COCO
dataset, no banknotes) are downloaded automatically by Ultralytics on first use; this is the only download.

### Dataset layout and annotation format

```
dataset/
  images/pool/*.jpg          all images (JPEG, long side <= 1600 px)
  labels/pool/*.txt          YOLO labels, same basename; EMPTY file = genuine negative
  metadata/image_manifest.csv  denomination, side, lighting, angle, background, source, session, split, label_status, sha256
```

Each label line: `0 cx cy w h`, normalised to [0,1]; class `0 = fifty_rupee_note`. The split lives in the manifest
(whole capture *sessions* go to train/val/test), and `train.py` generates the Ultralytics file lists from it.

## Tests

```bash
pytest -q          # 111 tests (110 pass, 1 skipped until a real model.onnx exists), ~25 s; see docs/TEST_PLAN.md
```

Tests that use fixture or random-weight ONNX models verify *mechanics* (loading, decoding, coordinate mapping,
export parity, error handling), not detection quality. The one test that exercises the real `model.onnx` is
skipped until that file exists.

## Project structure

```
test.py                     required entry point (thin; logic in src/detector/app.py)
src/detector/               preprocessing, postprocessing, inference (cv2.dnn), thresholding, webcam, visualization, dataset, metrics
scripts/                    capture_dataset, annotate, split_dataset, validate_dataset, train, export_onnx, validate_onnx, evaluate, stress_test
configs/                    train.yaml, inference.yaml
tests/                      pytest suite
docs/                       audit, architecture, plan, test plan, collection guide, evaluation, model card, limitations, acceptance report
reports/                    generated (audit, ONNX validation, evaluation, threshold sweeps)
```

## Troubleshooting

| symptom | fix |
|---|---|
| `Model file not found` | train + export first, or pass `--model path\to\model.onnx` |
| `Could not open camera 0` | close Zoom/Teams/browser tabs using the webcam, check Windows *Privacy → Camera*, try `--camera 1`, or use `--source` |
| `Model loaded but a test forward pass failed ... --imgsz` | pass the size used at export, e.g. `--imgsz 480` (normally auto-read from `model.meta.json`, so keep that file next to the model) |
| label shows `Rs50` instead of `₹50` | no font with the ₹ glyph found; set env var `RUPEE_FONT` to a `.ttf` that has it (Segoe UI on Windows, DejaVu Sans on Linux) |
| `cv2.imshow` error | an OpenCV *headless* build is installed: `pip uninstall opencv-python-headless` then `pip install opencv-python`, or use `--no-display` |
| low FPS | CPU only; use a smaller `--imgsz` model (see Latency in the model card), close other apps |

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). In short: this is a technical demonstration, **not** a currency
authentication or counterfeit-detection system; a detection does not mean a note is genuine; the confidence score is
not a calibrated probability; reliability depends entirely on how varied your photographs are; performance has only
been measured on the development laptop (2-core i5-7300U).

## Licence note

Training and export use [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Check its licence terms
before any commercial use. Run-time inference (`test.py`) uses only OpenCV, NumPy, PyYAML and Pillow.

## Reproducibility

Each training run writes `runs/<name>/run_info.json` (config, seed, library versions, dataset manifest hash,
checkpoint SHA-256); `export_onnx.py` writes `model.meta.json` (input size, hashes, versions); each evaluation
report records the model hash, dataset manifest hash, threshold, hardware and timestamp.
