"""Shared fixtures.

`fixed_onnx` builds a tiny ONNX graph that ignores its input and emits a KNOWN YOLOv8-style output
tensor. It lets us test cv2.dnn loading + decoding + coordinate mapping end-to-end without a trained
model. It proves pipeline mechanics only, NOT detection quality.
"""

from __future__ import annotations

import numpy as np
import pytest

SIZE = 480


def make_output(dets, n=20):
    """dets: list of (cx, cy, w, h, score) in model-space pixels -> (1, 5, n) float32."""
    out = np.zeros((1, 5, n), np.float32)
    for i, d in enumerate(dets):
        out[0, :, i] = d
    return out


FIXED_DETS = [
    (240, 240, 100, 60, 0.90),   # strong detection
    (242, 241, 100, 60, 0.80),   # duplicate -> removed by NMS
    (100, 100, 50, 50, 0.30),    # weak candidate
]


@pytest.fixture(scope="session")
def fixed_onnx(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    out = make_output(FIXED_DETS)
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])
    outp = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zeroed"]),
        helper.make_node("Add", ["zeroed", "const"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "fixed", [inp], [outp],
        initializer=[numpy_helper.from_array(np.zeros((), np.float32), "zero"), numpy_helper.from_array(out, "const")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    p = tmp_path_factory.mktemp("onnx") / "fixed.onnx"
    onnx.save(model, str(p))
    return p


@pytest.fixture()
def frame():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
