# ₹50 Vision: live detector for the new Indian ₹50 note

A webcam program that draws a green box labelled **`₹50`** and a confidence score when the **new light blue-green ₹50 note**
(Gandhi portrait on the front, Hampi stone chariot on the back) is held up to the camera. Other banknotes get a separate
**red** box ("Other note"). It is the submission for the ZED Digital assignment *Train a ₹50 Note Detector*.

```bash
python test.py --model model.onnx
```

## For reviewers: the short version

| Question | Answer |
|---|---|
| Does it run with the required command? | **Yes.** `python test.py --model model.onnx` opens the webcam with OpenCV and draws the box, the label `₹50` and the score. |
| What model is it? | YOLOv8n, trained by the developer, exported to **ONNX** (`model.onnx`, 12.7 MB). At run time only OpenCV's DNN module is used (no PyTorch, no GPU, no internet). |
| Whose images? | Only the developer's own captures of their own notes: 801 images, **nothing downloaded, printed, photocopied or generated**. |
| Does it find the ₹50? | On a held-out test the evaluation model showed ₹50 on **67 of 68** real ₹50 frames, front and back, held flat or upright. |
| Does it avoid false ₹50s? | **Mostly, not perfectly.** 8 of 191 non-₹50 test images (4.2 %) were also labelled ₹50: six close-up ₹200 scans and two ceiling frames. Held-up ₹10/₹20/₹100/₹500 notes were never called ₹50 in the test. See [the numbers](#measured-results). |
| Is it finished? | Yes: data, training code, ONNX model, `test.py`, evaluation, 127 passing tests, documentation. The limits are listed honestly below. |

## Run it (3 steps)

Needs Python 3.10+ and a webcam. Tested on Windows 10 with Python 3.14 and OpenCV 5.0; nothing else is needed at run time.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python test.py --model model.onnx
```

Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip` and `python` lines (code is portable, but only Windows was tested).

Press **`q`** or **Esc** to quit. Press **`i`** for the decision inspector (shows why a candidate was rejected) and **`+` / `-`** to change the threshold live.

## What you will see

| In front of the camera | On screen |
|---|---|
| The new ₹50 (front or back) | Green box `₹50 0.87` (the number is the model's confidence score, not a calibrated probability) and a green banner *Rs.50 DETECTED* |
| Another note (₹10, ₹20, ₹100, ₹200, ₹500) | Red box `Other note 0.xx` and a red banner *OTHER NOTE - not a ₹50*. It does not say which denomination. |
| No note, or any other object | No box, grey banner *Searching for a ₹50 note...* |

The top bar shows the threshold in use, the inference time and live FPS. Hold the note fairly steady, well lit, filling a good part of the frame.

## Assignment checklist

| Requirement from the PDF | Status | Where |
|---|---|---|
| Only the new ₹50 note (front: Gandhi, back: Hampi chariot) | Met | class 0 `fifty_rupee_note`; both sides are examples of it; older ₹50 not used |
| Own photos of the ₹50, front and back, varied conditions | Met, one physical note | 198 boxed ₹50 images: front, back, sideways, upright; different backgrounds and sittings. [dataset/](dataset/), [audit](reports/dataset_audit.md) |
| Photos of ₹10, ₹20, ₹100, ₹200, ₹500 as "not ₹50" | Met | 85–88 images of each (one physical copy of each note), class 1 `other_note` |
| Do not download note images | Met | none used; an offered RBI PDF folder was declined and is git-ignored |
| Do not print, photocopy or create fake notes | Met | only real notes were photographed; no generated images anywhere in `dataset/` |
| Any model (YOLO recommended) | Met | YOLOv8n |
| Training images and label files in `dataset/` | Met | `dataset/images/pool`, `dataset/labels/pool` (YOLO format), `dataset/metadata/image_manifest.csv` |
| Source code: training code and test script | Met | [scripts/train.py](scripts/train.py), [configs/train.yaml](configs/train.yaml), [test.py](test.py), [src/detector/](src/detector/) |
| Trained model in ONNX that loads in OpenCV | Met | [model.onnx](model.onnx) loaded with `cv2.dnn`; [reports/onnx_validation.md](reports/onnx_validation.md) shows it matches the PyTorch model |
| `test.py` opens the webcam with OpenCV, draws a box labelled "₹50" with the confidence | Met | 200-frame run on a real webcam, rendered frames, tests |
| README: how to install and run | Met | this file |
| Run command `python test.py --model model.onnx` | Met | works as written |
| One public GitHub repository | Met | https://github.com/sasi-04/Train-a-50-Note-Detector |
| "Detects the ₹50 reliably and does not show ₹50 for anything else" | **Partly**, measured | see below. Reliable on the developer's data; not perfect, and not yet proven on a different camera and different notes |

One thing to judge: the assignment says "photos with your phone". 117 of the images are phone/scanner-app captures; the other 684 are
frames from the **laptop webcam** (the same kind of camera the demo uses), because the phone-only first model scored a real note
poorly and a window as a ₹50 on a webcam. They are the developer's own captures of their own notes and are marked `source=webcam` in the manifest.

## Measured results

Everything here comes from files in [reports/](reports/); methods and caveats are in [docs/EVALUATION.md](docs/EVALUATION.md).
The test frames come from recording sessions that were **not used for training or for choosing thresholds**.

| Held-out test (259 images) | Result |
|---|---|
| New ₹50 (68 images) | green ₹50 shown on **67** (99 %); the upright (portrait) ones **32 of 32**; front 8 of 8; back 3 of 6 (too few to be sure) |
| ₹10 / ₹20 / ₹100 / ₹500 held in hand | never labelled ₹50; a red "Other note" box on most of them |
| ₹200 (28 images) | **6 labelled ₹50 by mistake** (full-frame close-up scans) |
| No note (38 frames) | nothing shown on 36; **2 labelled ₹50** (ceiling) |
| All non-₹50 images | **8 of 191 (4.2 %) labelled ₹50**; statistically the true rate could be up to about 8 % |
| Red box on other notes | 138 of 153 |
| Speed | about 64 ms per frame on a 2-core laptop CPU (about 15 FPS) |
| Live check, no note shown | 200 webcam frames: 0 ₹50, 0 red boxes |

Why the shipped model has no held-out score: after the evaluation, the model was fine-tuned once more on **all** 801 images
(so the weakest cases are learned too). A model cannot be scored on data it trained on, so the numbers above are for the
evaluation model that has the same design; the shipped file is verified only by the ONNX check, the tests and the live run.

## Limitations in plain words

* **One physical copy of each note, one person, a few rooms.** A different ₹50 note, a different webcam or different light may behave differently.
* **Some false ₹50 labels remain** (4.2 % of non-₹50 images in the test), mostly large flat note-like images and one ceiling area. No score threshold can remove them, because they score high.
* **Back side and upright are covered by one recording session each**; the back side had only 6 test frames.
* **Not an authenticity checker.** It recognises how the note looks; it cannot tell a genuine note from a good fake. The score is not a probability.
* **Privacy:** the webcam frames show the developer and people in the background (other people, lanyards). Faces are not labelled or used for anything but the background; if this is a concern, remove those frames and retrain.
* Full list: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Command options

```bash
python test.py --model model.onnx --threshold 0.5 --camera 1     # stricter score, other webcam
python test.py --model model.onnx --source clip.mp4 --inspector  # video or image instead of a webcam
python test.py --model model.onnx --no-display --max-frames 100 --summary-json run.json   # headless timing run
python test.py --help
```

| flag | meaning |
|---|---|
| `--model` | ONNX file (default `model.onnx`); keep `model.meta.json` next to it (holds the input size) |
| `--threshold` | ₹50 score threshold in (0,1); default **0.27** from `configs/inference.yaml` (other-note threshold 0.37) |
| `--camera` / `--source` | webcam index (default 0) / image or video file |
| `--imgsz` | model input size; normally read from `model.meta.json` (480) |
| `--confirm-frames N` | require N hits in the last 5 frames before showing a box (default 1 = off) |
| `--inspector`, `--no-display`, `--max-frames`, `--summary-json`, `--log-file`, `--verbose` | diagnostics |

Exit codes: 0 ok · 2 bad arguments · 3 model missing/unloadable · 4 camera problem · 5 repeated inference failure.
An inference failure is shown as an error banner, never as "no detection" or a ₹50.

The two thresholds were chosen on the validation sessions only (rule: no false ₹50 on any validation negative, then the middle
of the safe range). Curves: `reports/threshold_analysis_val.png`.

## Troubleshooting

| symptom | fix |
|---|---|
| `Model file not found` | run from the repository folder, or pass `--model path\to\model.onnx` |
| `Could not open camera 0` | close Zoom/Teams/browser tabs using the webcam, check Windows *Privacy → Camera*, try `--camera 1` |
| `... test forward pass failed ... --imgsz` | keep `model.meta.json` beside the model, or pass `--imgsz 480` |
| label shows `Rs50` instead of `₹50` | no font with the ₹ glyph found; set env var `RUPEE_FONT` to a `.ttf` that has it (Segoe UI on Windows) |
| `cv2.imshow` error | a headless OpenCV is installed: `pip uninstall opencv-python-headless`, `pip install opencv-python`, or use `--no-display` |
| low FPS | CPU only; close other apps |

## For developers: retrain and re-evaluate

Install the extra packages: `pip install -r requirements-train.txt` (PyTorch + Ultralytics) and `pip install -r requirements-dev.txt` (tests).

```bash
pytest -q                                   # 127 tests, ~30 s (see docs/TEST_PLAN.md)
python scripts/validate_dataset.py          # audit -> reports/dataset_audit.md ; must say READY
python scripts/train.py                     # YOLOv8n, CPU, configs/train.yaml (refuses to start if the audit fails)
python scripts/export_onnx.py --weights runs/<name>/weights/best.pt --out model.onnx --imgsz 480 --force
python scripts/validate_onnx.py --model model.onnx --weights runs/<name>/weights/best.pt --images dataset/images/pool --imgsz 480
python scripts/evaluate.py --model model.onnx --split val --select-threshold --write-config   # thresholds from VAL only
python scripts/stress_test.py --model model.onnx --split val
python scripts/evaluate.py --model model.onnx --split test                                    # held-out report
```

Collecting more data (rules: your own genuine notes only; never download, generate, print or photocopy notes):

```bash
python scripts/collect_webcam.py --session webcam_s06 --split train   # guided ~4 min webcam session, 12 prompts incl. upright ₹50/₹20
python scripts/capture_dataset.py ingest D:\photos\s01_front --denomination 50_new --side front --session s01   # phone photos
python scripts/annotate.py --only-unlabeled                           # draw the boxes yourself
python scripts/split_dataset.py --seed 42                             # whole sessions go to train/val/test
```

Dataset format: `dataset/images/pool/*.jpg`, `dataset/labels/pool/*.txt` (YOLO: `class cx cy w h`, normalised; `0` = ₹50, `1` = other note;
an empty file = background image), and `dataset/metadata/image_manifest.csv` with denomination, side, source, session, split and SHA-256.
Training options include `--rot90` (adds 90°-rotated copies of training images so the upright note is learned) and `--final-fit`.
Pretrained COCO weights (`yolov8n.pt`, no banknotes) are downloaded automatically by Ultralytics on first training; this is the only download.

## Project structure

```
test.py                  required entry point (thin; logic in src/detector/app.py)
model.onnx, model.meta.json   the trained model and its input size / hashes
src/detector/            preprocessing, postprocessing, inference (cv2.dnn), thresholding, webcam, visualisation, metrics
scripts/                 data collection, annotation, split, audit, train, export, ONNX validation, evaluation, stress test
configs/                 train.yaml, inference.yaml
dataset/                 801 images + YOLO labels + manifest
tests/                   pytest suite (127 tests)
reports/                 audit, ONNX validation, evaluation, threshold sweeps
docs/                    requirements audit, architecture, test plan, evaluation, model card, limitations, acceptance report
```

Further reading: [ACCEPTANCE_REPORT](docs/ACCEPTANCE_REPORT.md) (full checklist with evidence), [EVALUATION](docs/EVALUATION.md),
[MODEL_CARD](docs/MODEL_CARD.md), [LIMITATIONS](docs/LIMITATIONS.md), [REQUIREMENTS_AUDIT](docs/REQUIREMENTS_AUDIT.md).

## Licence note

Training and export use [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0); check its terms before commercial use.
Run-time inference (`test.py`) uses only OpenCV, NumPy, PyYAML and Pillow.

## Reproducibility

Each training run writes `runs/<name>/run_info.json` (config, seed, library versions, dataset hash, checkpoint hash);
`export_onnx.py` writes `model.meta.json`; each evaluation report records the model hash, dataset hash, thresholds, hardware and time.
