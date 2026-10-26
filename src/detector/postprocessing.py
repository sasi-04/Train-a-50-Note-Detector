"""Decode raw YOLO ONNX output into boxes in original-frame pixel coordinates.

Supported layouts (detected from the output shape, never assumed):
  * (1, 4+nc, N)  channels-first, Ultralytics YOLOv8 export: rows cx,cy,w,h,class scores
  * (1, N, 4+nc)  channels-last variant of the same head
  * (1, N, 6)     end-to-end / NMS-free export: x1,y1,x2,y2,score,class

Class scores in the YOLOv8 head are sigmoid outputs, already in [0,1]; they are
detector confidence scores, not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import LetterboxMeta


class OutputFormatError(ValueError):
    """The model output does not match any supported layout."""


@dataclass(frozen=True)
class Candidate:
    """One post-NMS detection candidate in original-frame pixels (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def describe_layout(shape: tuple[int, ...]) -> str:
    """Return 'channels_first' | 'channels_last' | 'end2end' for an output shape."""
    s = tuple(int(d) for d in shape)
    if len(s) == 2:
        s = (1,) + s
    if len(s) != 3 or s[0] != 1:
        raise OutputFormatError(f"unsupported output shape {shape}; expected (1, C, N) or (1, N, C)")
    _, a, b = s
    if b == 6 and a > 6:
        return "end2end"
    if a < b:
        return "channels_first"
    if b < a:
        return "channels_last"
    raise OutputFormatError(f"ambiguous square output shape {shape}")


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = a + b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS. Returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_one_to_many(boxes_xyxy[i], boxes_xyxy[rest])
        order = rest[ious <= iou_thr]
    return keep


def _to_original(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    out = boxes_xyxy.astype(np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, meta.orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, meta.orig_h)
    return out


def decode_output(
    output: np.ndarray,
    meta: LetterboxMeta,
    score_floor: float = 0.05,
    nms_iou: float = 0.45,
    max_det: int = 100,
    min_side_px: float = 2.0,
) -> list[Candidate]:
    """Raw network output -> NMS-filtered candidates (all classes) above ``score_floor``."""
    arr = np.asarray(output)
    if not np.issubdtype(arr.dtype, np.floating):
        raise OutputFormatError(f"model output must be floating point, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise OutputFormatError("model output contains NaN or Inf")
    layout = describe_layout(arr.shape)
    arr = arr.reshape((1,) + arr.shape[-2:]) if arr.ndim == 2 else arr
    mat = arr[0]

    if layout == "end2end":
        boxes, scores, cls = mat[:, :4], mat[:, 4], mat[:, 5].astype(int)
        keep = scores >= score_floor
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    else:
        if layout == "channels_first":
            mat = mat.T  # -> (N, 4+nc)
        if mat.shape[1] < 5:
            raise OutputFormatError(f"expected at least 5 channels (4 box + >=1 class), got {mat.shape[1]}")
        cls_scores = mat[:, 4:]
        cls = cls_scores.argmax(axis=1)
        scores = cls_scores.max(axis=1)
        keep = scores >= score_floor
        mat, scores, cls = mat[keep], scores[keep], cls[keep]
        cx, cy, w, h = mat[:, 0], mat[:, 1], mat[:, 2], mat[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    if boxes.shape[0] == 0:
        return []
    boxes = _to_original(boxes, meta)

    valid = ((boxes[:, 2] - boxes[:, 0]) >= min_side_px) & ((boxes[:, 3] - boxes[:, 1]) >= min_side_px)
    boxes, scores, cls = boxes[valid], scores[valid], cls[valid]
    if boxes.shape[0] == 0:
        return []

    result: list[Candidate] = []
    for c in np.unique(cls):  # class-aware NMS
        idx = np.nonzero(cls == c)[0]
        for k in nms(boxes[idx], scores[idx], nms_iou):
            j = idx[k]
            result.append(Candidate(*(float(v) for v in boxes[j]), float(scores[j]), int(c)))
    result.sort(key=lambda d: -d.score)
    return result[:max_det]
