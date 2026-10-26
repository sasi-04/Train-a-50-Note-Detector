"""Letterbox preprocessing matching the Ultralytics YOLOv8 export convention.

Verified contract (see docs/ARCHITECTURE.md):
  * input tensor  : float32, NCHW, shape (1, 3, S, S), S = imgsz (default 640)
  * colour order  : RGB (OpenCV frames are BGR, so we swap)
  * normalisation : pixel / 255.0, no mean/std
  * resize        : aspect-preserving, padded with grey (114) to a square
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114


class InvalidImageError(ValueError):
    """Raised when a frame cannot be preprocessed."""


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry needed to map model-space boxes back to the original frame."""

    ratio: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int
    size: int


def check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise InvalidImageError(f"image must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"image must be HxWx3 (BGR), got shape {image.shape}")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise InvalidImageError(f"image too small: {image.shape[1]}x{image.shape[0]}")


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize keeping aspect ratio and pad to size x size. Returns (BGR image, meta)."""
    check_image(image)
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = image if (new_w, new_h) == (w, h) else cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    meta = LetterboxMeta(ratio=ratio, pad_x=float(left), pad_y=float(top), orig_w=w, orig_h=h, size=size)
    return out, meta


def preprocess(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    """BGR uint8 frame -> (1,3,S,S) float32 RGB blob in [0,1] plus letterbox meta."""
    boxed, meta = letterbox(image, size)
    blob = cv2.dnn.blobFromImage(boxed, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True, crop=False)
    return blob, meta
