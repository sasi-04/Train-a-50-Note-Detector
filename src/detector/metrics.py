"""Detection evaluation: matching, threshold sweeps, grouped error rates, AP and Wilson intervals.

Definitions (used verbatim in docs/EVALUATION.md):
  * A prediction is a candidate of the target class with score >= threshold ("accepted").
  * Detection level: an accepted prediction is a TP if it matches an unmatched ground-truth box with
    IoU >= iou_thr (greedy, highest score first); otherwise FP. Unmatched GT boxes are FN.
  * Image level: a *positive image* has >= 1 GT box; it is a TP image if any accepted detection matches
    a GT box, else FN image. A *negative image* has no GT box; it is a FP image if ANY accepted
    detection exists. "Negative-image FP rate" = FP images / negative images (denominator reported).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from .dataset import NUM_CLASSES, DatasetPaths, parse_label_file
from .postprocessing import iou_one_to_many
from .thresholding import SweepPoint, plausible_shape


@dataclass
class ImageRecord:
    image_id: str
    meta: dict
    gt: np.ndarray  # (n,4) xyxy pixels, all classes
    preds: list[tuple[tuple[float, float, float, float], float, int]]  # (box, score, class)
    inference_ms: float = 0.0
    total_ms: float = 0.0
    image_wh: tuple[int, int] = (0, 0)
    gt_cls: np.ndarray | None = None  # (n,) class per gt box; default all class 0

    def __post_init__(self) -> None:
        if self.gt_cls is None:
            self.gt_cls = np.zeros(len(self.gt), int)

    def boxes_of(self, cls: int = 0) -> np.ndarray:
        return self.gt[self.gt_cls == cls]

    def has(self, cls: int = 0) -> bool:
        return bool((self.gt_cls == cls).any())

    @property
    def positive(self) -> bool:
        return self.has(0)

    def target_scores(self, target_class: int = 0) -> list[float]:
        """Scores of target-class candidates that pass the same shape check as the deployed decision."""
        return [s for b, s, c in self.preds if c == target_class and self._ok(b)]

    def _ok(self, box) -> bool:
        w, h = self.image_wh
        return True if not (w and h) else plausible_shape(box, w, h)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (n=0 -> (0,1))."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_gt(paths: DatasetPaths, image_id: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth boxes (xyxy pixels) and their class ids."""
    lab = paths.labels / f"{image_id}.txt"
    boxes, classes = [], []
    if lab.is_file():
        for c, cx, cy, bw, bh in parse_label_file(lab, NUM_CLASSES).boxes:
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            classes.append(c)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(classes, dtype=int)


def collect_predictions(detector, rows: Sequence[dict], paths: DatasetPaths,
                        progress: Callable[[int, int], None] | None = None) -> list[ImageRecord]:
    """Run the deployed OpenCV/ONNX detector over manifest rows (low score floor; thresholded later)."""
    recs: list[ImageRecord] = []
    for i, r in enumerate(rows):
        img = cv2.imread(str(paths.images / r["filename"]))
        if img is None:
            raise RuntimeError(f"cannot read {r['filename']}")
        h, w = img.shape[:2]
        res = detector.infer(img)
        gt, gt_cls = load_gt(paths, r["image_id"], w, h)
        recs.append(ImageRecord(
            r["image_id"], r, gt,
            [(c.box, c.score, c.class_id) for c in res.candidates],
            res.timings.inference_ms, res.timings.total_ms, (w, h), gt_cls))
        if progress:
            progress(i + 1, len(rows))
    return recs


def match(rec: ImageRecord, threshold: float, iou_thr: float = 0.5, target_class: int = 0) -> tuple[int, int, int]:
    """Return (tp, fp, fn) detections for one image at ``threshold``."""
    preds = sorted((p for p in rec.preds if p[2] == target_class and p[1] >= threshold and rec._ok(p[0])),
                   key=lambda p: -p[1])
    gt = rec.boxes_of(target_class)
    used = np.zeros(len(gt), bool)
    tp = 0
    for box, _, _ in preds:
        if len(gt):
            ious = iou_one_to_many(np.asarray(box), gt)
            ious[used] = -1
            j = int(ious.argmax())
            if ious[j] >= iou_thr:
                used[j] = True
                tp += 1
    return tp, len(preds) - tp, int((~used).sum())


def sweep(records: Sequence[ImageRecord], thresholds: Iterable[float], iou_thr: float = 0.5,
          target_class: int = 0) -> list[SweepPoint]:
    pts = []
    neg = [r for r in records if not r.has(target_class)]
    for t in thresholds:
        tp = fp = fn = fp_imgs = 0
        for r in records:
            a, b, c = match(r, t, iou_thr, target_class)
            tp, fp, fn = tp + a, fp + b, fn + c
        fp_imgs = sum(1 for r in neg if any(s >= t for s in r.target_scores(target_class)))
        p = tp / (tp + fp) if tp + fp else 1.0
        rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * rc / (p + rc) if (p + rc) and rc == rc else 0.0
        pts.append(SweepPoint(round(float(t), 4), tp, fp, fn, fp_imgs, len(neg), p, rc, f1))
    return pts


def average_precision(records: Sequence[ImageRecord], iou_thr: float = 0.5, target_class: int = 0) -> float:
    """All-point interpolated AP@IoU for ``target_class`` (VOC style)."""
    n_gt = sum(len(r.boxes_of(target_class)) for r in records)
    if n_gt == 0:
        return float("nan")
    dets = []
    for r in records:
        gt = r.boxes_of(target_class)
        used = np.zeros(len(gt), bool)
        for box, s, c in sorted((p for p in r.preds if p[2] == target_class and r._ok(p[0])), key=lambda p: -p[1]):
            hit = 0
            if len(gt):
                ious = iou_one_to_many(np.asarray(box), gt)
                ious[used] = -1
                j = int(ious.argmax())
                if ious[j] >= iou_thr:
                    used[j], hit = True, 1
            dets.append((s, hit))
    dets.sort(key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in dets])
    fp = np.cumsum([1 - d[1] for d in dets])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


@dataclass
class GroupStats:
    n_images: int = 0
    n_pos: int = 0
    n_neg: int = 0
    tp_images: int = 0
    fn_images: int = 0
    fp_images: int = 0
    det_tp: int = 0
    det_fp: int = 0
    det_fn: int = 0

    def row(self) -> dict:
        lo, hi = wilson_interval(self.fp_images, self.n_neg) if self.n_neg else (float("nan"),) * 2
        rlo, rhi = wilson_interval(self.tp_images, self.n_pos) if self.n_pos else (float("nan"),) * 2
        return {
            "images": self.n_images, "positive_images": self.n_pos, "negative_images": self.n_neg,
            "tp_images": self.tp_images, "fn_images": self.fn_images, "fp_images": self.fp_images,
            "image_recall": self.tp_images / self.n_pos if self.n_pos else None,
            "recall_ci95": [rlo, rhi] if self.n_pos else None,
            "neg_image_fp_rate": self.fp_images / self.n_neg if self.n_neg else None,
            "neg_image_fp_rate_ci95_upper": hi if self.n_neg else None,
            "det_tp": self.det_tp, "det_fp": self.det_fp, "det_fn": self.det_fn,
        }


def group_stats(records: Sequence[ImageRecord], threshold: float, key: Callable[[ImageRecord], str],
                iou_thr: float = 0.5, target_class: int = 0) -> dict[str, dict]:
    groups: dict[str, GroupStats] = defaultdict(GroupStats)
    for r in records:
        g = groups[key(r)]
        tp, fp, fn = match(r, threshold, iou_thr, target_class)
        g.n_images += 1
        g.det_tp, g.det_fp, g.det_fn = g.det_tp + tp, g.det_fp + fp, g.det_fn + fn
        if r.has(target_class):
            g.n_pos += 1
            g.tp_images += tp > 0
            g.fn_images += tp == 0
        else:
            g.n_neg += 1
            g.fp_images += fp > 0
    return {k: v.row() for k, v in sorted(groups.items())}


def overall(records: Sequence[ImageRecord], threshold: float, iou_thr: float = 0.5, target_class: int = 0) -> dict:
    out = group_stats(records, threshold, lambda r: "all", iou_thr, target_class).get("all", GroupStats().row())
    tp, fp, fn = out["det_tp"], out["det_fp"], out["det_fn"]
    out["precision"] = tp / (tp + fp) if tp + fp else None
    out["recall"] = tp / (tp + fn) if tp + fn else None
    p, rc = out["precision"], out["recall"]
    out["f1"] = 2 * p * rc / (p + rc) if p and rc else (0.0 if p is not None and rc is not None else None)
    out["ap50"] = average_precision(records, iou_thr, target_class)
    # image-level confusion matrix: rows = actual, cols = predicted
    out["confusion_image_level"] = {
        "actual_target__pred_target": out["tp_images"], "actual_target__pred_none": out["fn_images"],
        "actual_none__pred_target": out["fp_images"], "actual_none__pred_none": out["negative_images"] - out["fp_images"],
    }
    return out


def shown_label(rec: ImageRecord, thr_target: float, thr_other: float | None) -> str:
    """What the app would show for this image: 'rs50' (green) wins, else 'other' (red), else 'none'."""
    if any(c == 0 and s >= thr_target and rec._ok(b) for b, s, c in rec.preds):
        return "rs50"
    if thr_other is not None and any(c == 1 and s >= thr_other and rec._ok(b) for b, s, c in rec.preds):
        return "other"
    return "none"


def note_type_table(records: Sequence[ImageRecord], thr_target: float, thr_other: float | None,
                    key: Callable[[ImageRecord], str]) -> dict[str, dict]:
    """Per group: how many images end up shown as Rs.50 / Other note / nothing, with counts (the app's real output)."""
    out: dict[str, dict] = {}
    for r in records:
        g = out.setdefault(key(r), {"images": 0, "shown_rs50": 0, "shown_other": 0, "shown_none": 0})
        g["images"] += 1
        g["shown_" + shown_label(r, thr_target, thr_other)] += 1
    return dict(sorted(out.items()))


def latency_summary(records: Sequence[ImageRecord]) -> dict:
    inf = np.array([r.inference_ms for r in records]) if records else np.array([])
    tot = np.array([r.total_ms for r in records]) if records else np.array([])
    if not len(inf):
        return {}
    return {"n": len(inf), "inference_ms_median": float(np.median(inf)), "inference_ms_p95": float(np.percentile(inf, 95)),
            "pipeline_ms_median": float(np.median(tot)), "pipeline_ms_p95": float(np.percentile(tot, 95)),
            "throughput_fps_pipeline": float(1000 / np.median(tot)) if np.median(tot) > 0 else None}


DEFAULT_THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.99, 0.01)]
