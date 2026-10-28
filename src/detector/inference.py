"""ONNX model loading and inference through OpenCV's DNN module (the assignment's deployment path)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .logging_utils import get_logger
from .postprocessing import Candidate, OutputFormatError, decode_output, describe_layout
from .preprocessing import InvalidImageError, LetterboxMeta, preprocess

FALLBACK_SIZES = (640, 480, 416, 320)


class ModelLoadError(RuntimeError):
    """The ONNX file is missing, unreadable, or not runnable by OpenCV."""


class InferenceError(RuntimeError):
    """Inference failed for a frame; callers must NOT treat this as 'no detection'."""


@dataclass
class Timings:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class FrameResult:
    candidates: list[Candidate]
    timings: Timings
    meta: LetterboxMeta
    raw_shape: tuple[int, ...] = field(default_factory=tuple)


def sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.json")


def read_sidecar(model_path: Path) -> dict:
    p = sidecar_path(model_path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        get_logger().warning("Ignoring unreadable metadata file %s", p)
        return {}


class OnnxDetector:
    """Runs an exported YOLO ONNX model with cv2.dnn on CPU. No PyTorch / ONNX Runtime involved."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int | None = None,
        score_floor: float = 0.05,
        nms_iou: float = 0.45,
    ):
        log = get_logger()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        if not self.model_path.is_file():
            raise ModelLoadError(f"Model path is not a file: {self.model_path}")
        if self.model_path.suffix.lower() != ".onnx":
            log.warning("Model file %s does not have an .onnx extension", self.model_path.name)

        meta = read_sidecar(self.model_path)
        self.imgsz = int(imgsz or meta.get("imgsz") or 640)
        self.score_floor = score_floor
        self.nms_iou = nms_iou
        self.init_ms = 0.0
        self.layout_shape: tuple[int, ...] = ()

        t0 = time.perf_counter()
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except cv2.error as exc:
            raise ModelLoadError(f"OpenCV could not load {self.model_path.name}: {_short(exc)}") from None
        if self.net is None or self.net.empty():
            raise ModelLoadError(f"OpenCV loaded an empty network from {self.model_path.name}")
        if int(cv2.__version__.split(".")[0]) < 5:  # OpenCV 5's new engine is CPU-only and warns on these calls
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._out_names = list(self.net.getUnconnectedOutLayersNames())
        self._warmup(imgsz_explicit=imgsz is not None or "imgsz" in meta)
        self.init_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Loaded %s via OpenCV %s (imgsz=%d, output shape %s, init %.0f ms)",
            self.model_path.name, cv2.__version__, self.imgsz, self.layout_shape, self.init_ms,
        )

    @property
    def num_classes(self) -> int | None:
        """Number of class channels of a (1, 4+nc, N) head; None for layouts where it cannot be read from the shape."""
        s = self.layout_shape
        if len(s) == 3 and describe_layout(s) == "channels_first":
            return s[1] - 4
        if len(s) == 3 and describe_layout(s) == "channels_last":
            return s[2] - 4
        return None

    def _forward_blob(self, blob: np.ndarray) -> np.ndarray:
        self.net.setInput(blob)
        out = self.net.forward(self._out_names) if self._out_names else self.net.forward()
        if isinstance(out, (list, tuple)):
            out = out[0]
        return np.asarray(out)

    def _warmup(self, imgsz_explicit: bool) -> None:
        sizes = [self.imgsz] + ([] if imgsz_explicit else [s for s in FALLBACK_SIZES if s != self.imgsz])
        last: Exception | None = None
        for size in sizes:
            try:
                blob = np.zeros((1, 3, size, size), np.float32)
                out = self._forward_blob(blob)
                if out.ndim not in (2, 3) or out.size == 0:
                    raise OutputFormatError(f"unexpected output shape {out.shape}")
                self.imgsz, self.layout_shape = size, tuple(out.shape)
                return
            except (cv2.error, OutputFormatError) as exc:
                last = exc
        hint = " Pass --imgsz matching the size used at export." if imgsz_explicit else ""
        raise ModelLoadError(f"Model loaded but a test forward pass failed: {_short(last)}.{hint}")

    def infer(self, frame: np.ndarray) -> FrameResult:
        """Preprocess -> forward -> decode. Raises InvalidImageError / InferenceError, never swallows."""
        t0 = time.perf_counter()
        blob, meta = preprocess(frame, self.imgsz)
        t1 = time.perf_counter()
        try:
            out = self._forward_blob(blob)
        except cv2.error as exc:
            raise InferenceError(f"forward pass failed: {_short(exc)}") from None
        t2 = time.perf_counter()
        try:
            cands = decode_output(out, meta, self.score_floor, self.nms_iou)
        except OutputFormatError as exc:
            raise InferenceError(f"could not decode model output: {exc}") from None
        t3 = time.perf_counter()
        return FrameResult(
            cands, Timings((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000), meta, tuple(out.shape)
        )


def _short(exc: Exception | None) -> str:
    text = str(exc).strip().splitlines()
    return text[-1] if text else type(exc).__name__


__all__ = ["OnnxDetector", "ModelLoadError", "InferenceError", "InvalidImageError", "FrameResult", "Timings"]
