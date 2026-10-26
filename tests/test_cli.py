import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from detector import app

ROOT = Path(__file__).resolve().parents[1]


def run(argv):
    return app.run(argv)


def test_missing_model_returns_model_error(tmp_path, capsys):
    assert run(["--model", str(tmp_path / "missing.onnx"), "--no-display"]) == app.EXIT_MODEL
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize("thr", ["0", "1", "-0.2", "1.5", "nan"])
def test_invalid_threshold_is_usage_error(thr, tmp_path, capsys):
    assert run(["--model", str(tmp_path / "m.onnx"), "--threshold", thr]) == app.EXIT_USAGE
    assert "threshold" in capsys.readouterr().err


def test_invalid_imgsz(tmp_path):
    assert run(["--model", str(tmp_path / "m.onnx"), "--imgsz", "100"]) == app.EXIT_USAGE


def test_non_numeric_threshold_exits_via_argparse():
    with pytest.raises(SystemExit) as e:
        run(["--threshold", "abc"])
    assert e.value.code == 2


def test_confirm_frames_range(tmp_path):
    assert run(["--model", str(tmp_path / "m.onnx"), "--confirm-frames", "99"]) == app.EXIT_USAGE


def test_camera_failure_exit_code(fixed_onnx, monkeypatch, capsys):
    class Dead:
        def __init__(self, *a, **k): pass
        def isOpened(self): return False
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoCapture", Dead)
    assert run(["--model", str(fixed_onnx), "--imgsz", "480", "--no-display"]) == app.EXIT_CAMERA
    assert "Could not open camera" in capsys.readouterr().err


def test_headless_still_image_run_reports_detection_and_timing(fixed_onnx, tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8))
    out = tmp_path / "s.json"
    rc = run(["--model", str(fixed_onnx), "--imgsz", "480", "--source", str(img), "--no-display",
              "--threshold", "0.5", "--summary-json", str(out)])
    s = json.loads(out.read_text())
    assert rc == 0 and s["frames_processed"] == 1 and s["frames_with_accepted_detection"] == 1
    assert s["inference_ms"]["median"] > 0


def test_high_threshold_rejects_everything(fixed_onnx, tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((300, 400, 3), np.uint8) + 90)
    out = tmp_path / "s.json"
    run(["--model", str(fixed_onnx), "--imgsz", "480", "--source", str(img), "--no-display",
         "--threshold", "0.95", "--summary-json", str(out)])
    assert json.loads(out.read_text())["frames_with_accepted_detection"] == 0


def test_temporal_confirmation_delays_detection_on_still(fixed_onnx, tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((300, 400, 3), np.uint8) + 90)
    out = tmp_path / "s.json"
    run(["--model", str(fixed_onnx), "--imgsz", "480", "--source", str(img), "--no-display", "--max-frames", "5",
         "--threshold", "0.5", "--confirm-frames", "3", "--summary-json", str(out)])
    s = json.loads(out.read_text())
    assert s["frames_processed"] == 5 and s["frames_with_accepted_detection"] == 3  # frames 3,4,5


def test_unreadable_source_file(fixed_onnx, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert run(["--model", str(fixed_onnx), "--imgsz", "480", "--source", str(bad), "--no-display"]) == app.EXIT_CAMERA


def test_help_and_mandatory_command_via_subprocess(tmp_path):
    h = subprocess.run([sys.executable, str(ROOT / "test.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert h.returncode == 0 and "--model" in h.stdout and "--threshold" in h.stdout
    # exact mandatory form, with a model that does not exist -> clean error, no traceback
    r = subprocess.run([sys.executable, str(ROOT / "test.py"), "--model", str(tmp_path / "model.onnx")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == app.EXIT_MODEL
    assert "Traceback" not in r.stderr and "not found" in r.stderr
