"""Loads a fixture ONNX through cv2.dnn. Proves loading/decoding mechanics, not detection quality."""

import numpy as np
import pytest

from detector.inference import InferenceError, ModelLoadError, OnnxDetector
from detector.preprocessing import InvalidImageError


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        OnnxDetector(tmp_path / "nope.onnx")


def test_directory_is_not_a_model(tmp_path):
    with pytest.raises(ModelLoadError, match="not a file"):
        OnnxDetector(tmp_path)


def test_garbage_file_reports_clean_error(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"this is not an onnx model")
    with pytest.raises(ModelLoadError):
        OnnxDetector(bad)


def test_fixture_loads_and_reports_geometry(fixed_onnx):
    d = OnnxDetector(fixed_onnx, imgsz=480)
    assert d.imgsz == 480 and d.layout_shape == (1, 5, 20)


def test_explicit_imgsz_is_respected(fixed_onnx):
    # the fixture graph is shape-agnostic; a real size mismatch is covered by test_onnx_parity
    assert OnnxDetector(fixed_onnx, imgsz=320).imgsz == 320


def test_end_to_end_boxes_in_original_coordinates(fixed_onnx, frame):
    d = OnnxDetector(fixed_onnx, imgsz=480, score_floor=0.05)
    res = d.infer(frame)  # frame is 1280x720 -> ratio 0.375, pad_y 105
    scores = [round(c.score, 2) for c in res.candidates]
    assert scores == [0.9, 0.3]  # duplicate removed by NMS
    best = res.candidates[0]
    assert best.box == pytest.approx(((240 - 50) / 0.375, (240 - 30 - 105) / 0.375, (240 + 50) / 0.375, (240 + 30 - 105) / 0.375), abs=0.5)
    assert res.timings.inference_ms > 0 and res.timings.total_ms >= res.timings.inference_ms


def test_infer_rejects_invalid_frame(fixed_onnx):
    d = OnnxDetector(fixed_onnx, imgsz=480)
    with pytest.raises(InvalidImageError):
        d.infer(np.zeros((10, 10), np.uint8))


def test_infer_error_is_not_swallowed(fixed_onnx, frame, monkeypatch):
    d = OnnxDetector(fixed_onnx, imgsz=480)
    monkeypatch.setattr(d, "_forward_blob", lambda blob: np.full((1, 5, 20), np.nan, np.float32))
    with pytest.raises(InferenceError):
        d.infer(frame)
