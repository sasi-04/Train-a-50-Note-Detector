# Implementation plan and status

| phase | scope | status |
|---|---|---|
| A Audit | inspect repo (found empty), environment, webcam, dependency availability | done |
| B Dataset readiness | ingest, annotate, split-by-session, auditor, collection guide | done (117 images imported from the developer's scans) |
| C Training | `train.py` + `configs/train.yaml`, dataset gate, run info | done: evaluation run `rs50_split` + shipped run `rs50_final` (all images) |
| D ONNX deployment | `export_onnx.py`, `validate_onnx.py`, OpenCV decode path, sidecar metadata | done; OpenCV 5.0 and 4.14 verified |
| E Webcam app | `test.py`/`app.py`, camera recovery, inspector, timing | done |
| F Evaluation | `evaluate.py`, `stress_test.py`, threshold sweep, Wilson CIs | done; results in `docs/EVALUATION.md` |
| G Innovations | A dataset auditor ✔ · B stress test ✔ · C inspector ✔ · D ONNX validator ✔ · E reproducible reports ✔ · F graceful degradation ✔ · temporal confirmation ✔ (off by default) · second-stage verifier ✘ (not justified without data) · model comparison ✘ (needs data) | done as listed |
| H Dashboard | optional | **not built** (no requirement depends on it) |
| I Final verification | full pytest, mandatory command, docs, acceptance report | done; live positive test with real notes left to the developer |

## Remaining sequence (to improve reliability)

1. Collect + import more varied photos (`docs/DATASET_COLLECTION.md`): dim light, tilt, clutter, other notes in hand, objects.
2. `annotate.py`, `split_dataset.py`, `validate_dataset.py` until READY.
3. `train.py`, `export_onnx.py`, `validate_onnx.py`.
4. `evaluate.py --split val --select-threshold --write-config`, `stress_test.py --split val`; if negatives fail, add
   more varied negatives for the confusing categories and retrain (preferred over a higher threshold alone).
5. Only then `evaluate.py --split test`; paste real numbers into README, `docs/EVALUATION.md`, `docs/MODEL_CARD.md`.
6. If the stress test still fails, consider (in this order) more hard negatives, `--imgsz 640`, temporal confirmation
   (`--confirm-frames 3`, measure its latency), a crop-classifier verifier with an ablation.
7. Commit `model.onnx` + `model.meta.json` (12 MB is fine for git), push publicly.
