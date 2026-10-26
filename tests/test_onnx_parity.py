"""Exports a RANDOM-INIT single-class YOLOv8n and checks the OpenCV deployment path against PyTorch.

This validates export + preprocessing + decoding numerics (box coordinates, scores, NMS). It says
nothing about banknote accuracy. Skipped when the training stack is not installed.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("ultralytics")
pytest.importorskip("onnx")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    d = tmp_path_factory.mktemp("parity")
    m = YOLO("yolov8n.yaml")
    m.model = DetectionModel("yolov8n.yaml", nc=1, verbose=False)
    m.model.names = {0: "fifty_rupee_note"}
    pt = d / "rand.pt"
    m.save(str(pt))
    onnx_path = d / "rand.onnx"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_onnx.py"), "--weights", str(pt), "--out", str(onnx_path),
                        "--imgsz", "320", "--no-simplify"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    return pt, onnx_path


@pytest.mark.slow
def test_export_creates_sidecar_and_expected_shape(exported):
    import json
    _, onnx_path = exported
    meta = json.loads(onnx_path.with_suffix(".meta.json").read_text())
    assert meta["imgsz"] == 320 and meta["output_shape"] == [1, 5, 2100] and meta["class_names"] == {"0": "fifty_rupee_note"}


@pytest.mark.slow
def test_opencv_matches_pytorch_raw_and_decoded(exported, frame):
    import torch
    from ultralytics import YOLO

    from detector.inference import OnnxDetector
    from detector.postprocessing import decode_output, iou_one_to_many
    from detector.preprocessing import preprocess

    pt, onnx_path = exported
    det = OnnxDetector(onnx_path, score_floor=0.001)
    assert det.imgsz == 320  # picked up from the sidecar
    net = YOLO(str(pt)).model.eval().float()
    blob, meta = preprocess(frame, 320)
    det.net.setInput(blob)
    cv_out = np.asarray(det.net.forward(det.net.getUnconnectedOutLayersNames())[0])
    with torch.no_grad():
        ref = net(torch.from_numpy(blob))
    ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
    assert np.abs(ref[:, :4] - cv_out[:, :4]).max() < 0.5       # px in model space
    assert np.abs(ref[:, 4:] - cv_out[:, 4:]).max() < 2e-3      # raw scores
    a = decode_output(cv_out, meta, 0.001)[:15]
    b = decode_output(ref, meta, 0.001)[:15]
    assert len(a) == len(b) > 0
    for x in b:
        ious = iou_one_to_many(np.array(x.box), np.array([y.box for y in a]))
        assert ious.max() > 0.9


@pytest.mark.slow
def test_wrong_imgsz_gives_actionable_error(exported):
    from detector.inference import ModelLoadError, OnnxDetector
    _, onnx_path = exported
    with pytest.raises(ModelLoadError, match="--imgsz"):
        OnnxDetector(onnx_path, imgsz=640)


@pytest.mark.slow
def test_validate_onnx_script_passes(exported, tmp_path):
    pt, onnx_path = exported
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_onnx.py"), "--model", str(onnx_path),
                        "--weights", str(pt), "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:]
    assert "RESULT: PASS" in r.stdout


@pytest.mark.real_model
def test_real_model_artifact_loads_and_runs():
    model = ROOT / "model.onnx"
    if not model.is_file():
        pytest.skip("model.onnx not present: train and export first (no real model has been produced yet)")
    from detector.inference import OnnxDetector
    det = OnnxDetector(model)
    res = det.infer(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    assert res.timings.inference_ms > 0
