# ₹50 Vision — Indian ₹50 banknote detection

Real-time webcam detector for the **new light blue-green Indian ₹50 note** (Mahatma Gandhi portrait on the
front, Hampi stone chariot on the back). Built for the ZED Digital assignment "Train a ₹50 Note Detector".
YOLOv8n is trained on the developer's own phone photographs, exported to **ONNX**, and run live through
**OpenCV's DNN module** (no PyTorch needed at run time).

```bash
python test.py --model model.onnx
```

> **Status (read first).** `model.onnx` (YOLOv8n, 480 px, 12.7 MB) is trained on **738 of the developer's own captures**:
> 117 phone/scanner images plus **621 frames from the developer's webcam** (the same kind of camera the demo uses): 156 boxed
> images of the new ₹50 (**one physical note**, front/back/sideways), ₹10/₹20/₹100/₹200/₹500 and no-note scenes (room, people,
> a blue-green window) as negatives. The data are still **narrow** (one person, one copy of each note, a few rooms).
>
> Measured honestly on a **held-out webcam session** (details and caveats in [docs/EVALUATION.md](docs/EVALUATION.md)): the
> evaluation model labelled **0 of 167** non-₹50 frames as ₹50 (₹10/20/100/200/500, faces, paper, phone, window; true rate
> could still be up to ~2 %) and found **23 of 36 (64 %)** of the ₹50 frames (blurred, backlit and partly hidden ones are
> missed). The *shipped* model was then fine-tuned on all images, so no held-out number exists for it. A 200-frame live webcam
> run with no note shown raised 0 false ₹50s. Expect some misses; hold the note steady and well lit.
> Full audit: [docs/ACCEPTANCE_REPORT.md](docs/ACCEPTANCE_REPORT.md).

## What you should see when you run it

A window titled *Rs50 Vision* opens on your webcam.

* **Target note in view (front or back):** a green box with the label `₹50 0.87`, where the number is the
  detector's **confidence score** (not a calibrated probability), and a green banner *Rs.50 DETECTED*.
* **No target / another note / an object:** no box, grey banner *Searching for Rs.50 note...*.
* Header bar: the active threshold (marked *unvalidated* if it was not chosen on validation data), inference time and
  FPS measured live. A detection also has to have a plausible banknote shape (not a hairline strip or a speck), a simple rule
  from the note's physical proportions. It does not stop large note-shaped false alarms (see the webcam section below).
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

**Operating threshold.** `configs/inference.yaml` holds `0.46`, selected by `scripts/evaluate.py --select-threshold` on the
**validation** webcam session of the evaluation model (0 false ₹50 on 171 negative images, recall 86–90 %; the false-alarm-free
range starts at 0.41, so 0.46 sits inside it with margin). The shipped model reuses that value. A higher threshold reduces false
positives but misses more genuine notes: `--threshold 0.40` is more sensitive, `--threshold 0.60` stricter. Trade-off curve:
`reports/threshold_analysis_val.png`, `reports/threshold_sweep_val.csv`. Press `+`/`-` while running to adjust live.

## Complete workflow: photos → model → evaluation

Full collection protocol: [docs/DATASET_COLLECTION.md](docs/DATASET_COLLECTION.md).

**Hard rules (from the assignment):** photograph genuine notes yourself with your phone; photograph the new ₹50
on **both sides** and ₹10, ₹20, ₹100, ₹200, ₹500 as negatives; never download banknote images, never use generated
images, never print or photocopy notes. Everything below only ever reads your local dataset.

The current dataset was imported from four scanner-app PDFs with
`scripts/import_scan_pdf.py` and `scripts/import_handheld.py` (page→label/split maps in `dataset/metadata/*.yaml`, which also
explain every split decision). To add more photos use the generic flow below, then re-run steps 3–6.

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

### Making it work on *your* webcam (recommended)

A webcam looks different from a phone (softer, colour cast, other lighting): the first model, trained on phone/scanner images
only, scored a real note at 0.33 and a window at 0.69. Webcam frames fixed that, so they are included (sessions 1–3). To add
more, record guided sessions (about 4 minutes each):

```bash
python scripts/collect_webcam.py --session webcam_s01 --split train     # prompts: ₹50 front, back, sideways, ₹10 ... ₹500, no note
python scripts/collect_webcam.py --session webcam_s02 --split val       # a second sitting, used to choose the threshold
python scripts/collect_webcam.py --session webcam_s03 --split test      # a third, kept for an honest final check
python scripts/propose_boxes.py                                         # boxes for the ₹50 frames; open reports/box_review/sheet_*.jpg
python scripts/propose_boxes.py --accept --exclude 4,17 --drop-rejected # list the tiles with a wrong box
python scripts/validate_dataset.py                                      # then train / export / evaluate as above
```

Hold **only** the note named on screen: frames of other notes and of "no note" are saved as negatives.

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
pytest -q          # 115 tests, ~30 s; see docs/TEST_PLAN.md (the real-model test is skipped if model.onnx is absent)
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
