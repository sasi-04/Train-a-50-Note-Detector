"""Drawing: detection box, rupee-sign label (cv2.putText cannot render U+20B9), header, banner and inspector."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import DISPLAY_LABEL
from .logging_utils import get_logger
from .postprocessing import Candidate
from .thresholding import Decision

# BGR colours
GREEN = (80, 200, 60)
RED = (60, 60, 230)
ACCENT = (45, 45, 215)  # red accent of the header mark
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
PANEL = (28, 28, 28)

_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]


@lru_cache(maxsize=2)
def _font_path(bold: bool = False) -> str | None:
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    extra = os.environ.get("RUPEE_FONT")
    for cand in ([extra] if extra else []) + (_BOLD if bold else []) + _REGULAR:
        if cand and Path(cand).is_file():
            try:
                f = ImageFont.truetype(cand, 28)
                # A font lacking the glyph renders the same .notdef box as any other missing char.
                rupee, missing = f.getmask("\u20b9"), f.getmask("\uffff")
                if rupee.size != missing.size or bytes(rupee) != bytes(missing):
                    return cand
            except OSError:
                continue
    if not bold:
        get_logger().warning("No font with the rupee sign found; falling back to 'Rs50'. Set RUPEE_FONT=<ttf path>.")
    return None


def rupee_supported() -> bool:
    return _font_path() is not None


def currency() -> str:
    """'₹' when a font can draw it, otherwise 'Rs.'."""
    return DISPLAY_LABEL[0] if rupee_supported() else "Rs."


@lru_cache(maxsize=512)
def _text_mask(text: str, height: int, bold: bool) -> np.ndarray:
    """Anti-aliased 8-bit coverage mask of ``text`` (Pillow). Bounded cache: few distinct strings per run."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(_font_path(bold) or _font_path(False), height)
    left, top, right, bottom = font.getbbox(text)
    img = Image.new("L", (right - left + 4, bottom - top + 4), 0)
    ImageDraw.Draw(img).text((2 - left, 2 - top), text, font=font, fill=255)
    m = np.asarray(img)
    m.setflags(write=False)
    return m


def _composite(frame: np.ndarray, mask: np.ndarray, x: int, y: int, color) -> None:
    H, W = frame.shape[:2]
    h, w = mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[..., None] / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - m) + np.asarray(color, np.float32) * m).astype(np.uint8)


def text_size(text: str, height: int, bold: bool = False) -> tuple[int, int]:
    if rupee_supported():
        h, w = _text_mask(text, height, bold).shape
        return w, h
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, height / 32, 2 if bold else 1)
    return tw + 4, th + base + 4


def put_text(frame: np.ndarray, text: str, x: int, y: int, height: int = 20, color=WHITE, bold: bool = False,
             shadow: bool = True) -> None:
    """Draw ``text`` with its top-left at (x, y). Falls back to Hershey (ASCII) without a usable font."""
    if not rupee_supported():
        scale = height / 32
        base_y = y + int(height * 0.85)
        if shadow:
            cv2.putText(frame, text, (x + 1, base_y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3 if bold else 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, base_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)
        return
    mask = _text_mask(text, height, bold)
    if shadow:
        _composite(frame, mask, x + 1, y + 1, BLACK)
    _composite(frame, mask, x, y, color)


def _fill_alpha(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color, alpha: float) -> None:
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def format_label(score: float, show_rupee: bool | None = None) -> str:
    """'₹50 0.87' (score is a detector confidence, not a calibrated probability)."""
    sym = DISPLAY_LABEL if (show_rupee if show_rupee is not None else rupee_supported()) else "Rs50"
    return f"{sym} {score:.2f}"


def draw_label(frame: np.ndarray, text: str, x: int, y: int, bg=GREEN, fg=WHITE, height: int = 26) -> None:
    """Filled chip whose bottom-left is (x, y): above the box if there is room, else inside it."""
    tw, th = text_size(text, height, bold=True)
    w, h = tw + 14, th + 8
    H, W = frame.shape[:2]
    top = y - h if y - h >= 0 else y
    left = max(0, min(x, W - w))
    cv2.rectangle(frame, (left, top), (min(W - 1, left + w), min(H - 1, top + h)), bg, -1)
    put_text(frame, text, left + 7, top + 4, height, fg, bold=True, shadow=False)


def draw_detection(frame: np.ndarray, det: Candidate) -> None:
    """Thin outline + thick corner brackets + label chip."""
    H, W = frame.shape[:2]
    x1, y1 = int(round(max(0, det.x1))), int(round(max(0, det.y1)))
    x2, y2 = int(round(min(W - 1, det.x2))), int(round(min(H - 1, det.y2)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2, cv2.LINE_AA)
    ln = max(14, min(x2 - x1, y2 - y1) // 5)
    t = 6
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * ln, cy), GREEN, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * ln), GREEN, t, cv2.LINE_AA)
    draw_label(frame, format_label(det.score), x1, y1)


def draw_other(frame: np.ndarray, det: Candidate) -> None:
    """Red box for a recognised banknote that is NOT a Rs.50 (label 'Other note')."""
    H, W = frame.shape[:2]
    x1, y1 = int(round(max(0, det.x1))), int(round(max(0, det.y1)))
    x2, y2 = int(round(min(W - 1, det.x2))), int(round(min(H - 1, det.y2)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 3, cv2.LINE_AA)
    draw_label(frame, f"Other note {det.score:.2f}", x1, y1, bg=RED)


def draw_header(frame: np.ndarray, threshold: float, inference_ms: float | None, fps: float, validated: bool) -> None:
    """Translucent dark top bar: red-accent title on the left, live stats on the right."""
    H, W = frame.shape[:2]
    bar = 46
    _fill_alpha(frame, 0, 0, W, bar, PANEL, 0.72)
    cv2.rectangle(frame, (0, 0), (8, bar), ACCENT, -1)
    put_text(frame, f"{currency()}50 VISION", 20, 8, 26, WHITE, bold=True, shadow=False)
    w1, _ = text_size(f"{currency()}50 VISION", 26, True)
    put_text(frame, "Banknote Detection", 20 + w1 + 14, 17, 15, GREY, shadow=False)
    ms = f"{inference_ms:.0f} ms" if inference_ms is not None else "-- ms"
    stats = f"threshold {threshold:.2f}{'' if validated else ' (unvalidated)'}   |   {ms}   |   {fps:.0f} FPS"
    sw, _ = text_size(stats, 17)
    put_text(frame, stats, W - sw - 16, 14, 17, WHITE, shadow=False)


def draw_banner(frame: np.ndarray, text: str, ok: bool, hint: str = "q quit  |  i inspector  |  + / - threshold",
                error: bool = False) -> None:
    """Large result banner along the bottom; green when a target is accepted, red on inference error."""
    H, W = frame.shape[:2]
    bh = 58
    color = RED if error else (GREEN if ok else (58, 58, 58))
    _fill_alpha(frame, 0, H - bh, W, H, color, 0.92)
    put_text(frame, text, 20, H - bh + 11, 30 if ok else 24, WHITE, bold=ok, shadow=False)
    hw, _ = text_size(hint, 15)
    put_text(frame, hint, W - hw - 16, H - bh + 21, 15, (225, 225, 225), shadow=False)
    if ok:  # green frame border: unmistakable from across the room
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), GREEN, 8)


def draw_hud(frame: np.ndarray, lines: list[str], color=WHITE, top: int = 56) -> None:
    """Small readable text block (used by the inspector) on a translucent panel below the header."""
    if not lines:
        return
    widths = [text_size(ln, 17)[0] for ln in lines]
    _fill_alpha(frame, 8, top - 4, 8 + max(widths) + 20, top + 8 + 26 * len(lines), PANEL, 0.7)
    y = top + 2
    for ln in lines:
        put_text(frame, ln, 18, y, 17, color, shadow=False)
        y += 26


def draw_status_banner(frame: np.ndarray, text: str, ok: bool) -> None:  # backwards-compatible alias
    draw_banner(frame, text, ok)


def draw_inspector(frame: np.ndarray, decisions: list[Decision], limit: int = 5) -> None:
    """Show rejected candidates as grey boxes with raw score and the reason they were rejected."""
    for d in sorted(decisions, key=lambda d: -d.candidate.score)[:limit]:
        if d.accepted:
            continue
        c = d.candidate
        p1, p2 = (int(c.x1), int(c.y1)), (int(c.x2), int(c.y2))
        cv2.rectangle(frame, p1, p2, (210, 210, 210), 2, cv2.LINE_AA)
        txt = f"rejected {c.score:.2f} ({d.reason}, cls {c.class_id})"
        tw, th = text_size(txt, 15)
        _fill_alpha(frame, p1[0], p1[1], p1[0] + tw + 10, p1[1] + th + 6, PANEL, 0.75)
        put_text(frame, txt, p1[0] + 5, p1[1] + 3, 15, WHITE, shadow=False)


def is_windows() -> bool:
    return sys.platform.startswith("win")
