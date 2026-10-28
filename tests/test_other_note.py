"""Two-class behaviour: 'other note' candidates become red boxes and never count as a Rs.50."""

import json

import cv2
import numpy as np
import pytest

from detector import app
from detector.postprocessing import Candidate, LetterboxMeta, decode_output
from detector.thresholding import OTHER_NOTE, WRONG_CLASS, decide

META = LetterboxMeta(ratio=0.5, pad_x=0, pad_y=140, orig_w=1280, orig_h=720, size=640)


def two_class_head(dets, n=40):
    out = np.zeros((1, 6, n), np.float32)
    for i, d in enumerate(dets):
        out[0, :, i] = d
    return out


def test_confident_other_note_is_marked_but_not_accepted():
    ds = decide([Candidate(100, 100, 500, 300, 0.8, 1)], 0.5, 0, 1280, 720, other_class_id=1, other_threshold=0.5)
    assert ds[0].reason == OTHER_NOTE and not ds[0].accepted


def test_weak_other_note_is_ignored():
    ds = decide([Candidate(100, 100, 500, 300, 0.3, 1)], 0.5, 0, 1280, 720, other_class_id=1, other_threshold=0.5)
    assert ds[0].reason == WRONG_CLASS


def test_other_note_with_implausible_shape_is_ignored():
    ds = decide([Candidate(838, 0, 842, 700, 0.9, 1)], 0.5, 0, 1280, 720, other_class_id=1, other_threshold=0.5)
    assert ds[0].reason == WRONG_CLASS


def test_other_class_disabled_for_single_class_models():
    ds = decide([Candidate(100, 100, 500, 300, 0.9, 1)], 0.5, 0, 1280, 720)
    assert ds[0].reason == WRONG_CLASS


def test_class_agnostic_nms_gives_one_box_per_note():
    # same box claimed by both classes: Rs.50 0.55 vs other 0.70 -> only the stronger survives
    out = two_class_head([(320, 320, 200, 100, 0.55, 0.70)])
    c = decode_output(out, META)
    assert len(c) == 1 and c[0].class_id == 1 and c[0].score == pytest.approx(0.70)
    # (argmax over classes already picks one class per anchor; a second anchor on the same note must not add a 2nd box)
    out = two_class_head([(320, 320, 200, 100, 0.55, 0.10), (322, 321, 200, 100, 0.10, 0.70)])
    c = decode_output(out, META)
    assert len(c) == 1 and c[0].class_id == 1
    c2 = decode_output(out, META, class_agnostic=False)
    assert len(c2) == 2


def test_target_beats_other_on_same_object_when_stronger():
    out = two_class_head([(320, 320, 200, 100, 0.80, 0.10), (322, 321, 200, 100, 0.10, 0.60)])
    c = decode_output(out, META)
    assert len(c) == 1 and c[0].class_id == 0


def test_end_to_end_headless_summary_counts_other_frames(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    size = 480
    out = np.zeros((1, 6, 20), np.float32)
    out[0, :, 0] = (240, 240, 100, 60, 0.05, 0.9)  # confident OTHER note, no Rs.50
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])
    outp = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))
    nodes = [helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0), helper.make_node("Mul", ["mean", "zero"], ["z"]),
             helper.make_node("Add", ["z", "const"], ["output0"])]
    g = helper.make_graph(nodes, "two", [inp], [outp], initializer=[numpy_helper.from_array(np.zeros((), np.float32), "zero"),
                                                                  numpy_helper.from_array(out, "const")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    p = tmp_path / "two.onnx"; onnx.save(m, str(p))
    img = tmp_path / "a.jpg"; cv2.imwrite(str(img), np.full((300, 400, 3), 90, np.uint8))
    s = tmp_path / "s.json"
    cfg = tmp_path / "c.yaml"; cfg.write_text("other_class_id: 1\nother_threshold: 0.5\nthreshold: 0.5\nimgsz: 480\n")
    rc = app.run(["--model", str(p), "--imgsz", "480", "--source", str(img), "--no-display", "--config", str(cfg), "--summary-json", str(s)])
    j = json.loads(s.read_text())
    assert rc == 0 and j["frames_with_accepted_detection"] == 0 and j["frames_with_other_note_box"] == 1
