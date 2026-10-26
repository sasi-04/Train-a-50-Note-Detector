"""Drawing: boxes, the rupee-sign label (cv2.putText cannot render U+20B9), HUD and inspector overlay."""

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

GREEN = (80, 200, 60)
RED = (60, 60, 230)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    extra = os.environ.get("RUPEE_FONT")
    for cand in ([extra] if extra else []) + _FONT_CANDIDATES:
        if cand and Path(cand).is_file():
            try:
                f = ImageFont.truetype(cand, 28)
                # A font lacking the glyph renders the same .notdef box as any other missing char.
                rupee, missing = f.getmask("\u20b9"), f.getmask("\uffff")
                if rupee.size != missing.size or bytes(rupee) != bytes(missing):
                    return cand
            except OSError:
                continue
    get_logger().warning("No font with the rupee sign found; falling back to 'Rs50'. Set RUPEE_FONT=<ttf path>.")
    return None


def rupee_supported() -> bool:
    return _font_path() is not None


@lru_cache(maxsize=256)
def _render_text(text: str, height: int, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> np.ndarray:
    """Render text to a BGR chip using Pillow (bounded cache: confidence has few distinct strings)."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(_font_path(), height)
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left + 12, int(height * 1.35)
    img = Image.new("RGB", (w, h), bg[::-1])
    ImageDraw.Draw(img).text((6 - left, (h - (bottom - top)) // 2 - top), text, font=font, fill=fg[::-1])
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def format_label(score: float, show_rupee: bool | None = None) -> str:
    """'₹50 0.87' (score is a detector confidence, not a calibrated probability)."""
    sym = DISPLAY_LABEL if (show_rupee if show_rupee is not None else rupee_supported()) else "Rs50"
    return f"{sym} {score:.2f}"


def _blit(frame: np.ndarray, chip: np.ndarray, x: int, y: int) -> None:
    h, w = chip.shape[:2]
    H, W = frame.shape[:2]
    x, y = max(0, min(x, W - w)), max(0, min(y, H - h))
    w2, h2 = min(w, W - x), min(h, H - y)
    if w2 > 0 and h2 > 0:
        frame[y : y + h2, x : x + w2] = chip[:h2, :w2]


def draw_label(frame: np.ndarray, text: str, x: int, y: int, bg=GREEN, fg=WHITE, height: int = 24) -> None:
    """Draw a text chip whose bottom-left is (x, y), above the box if there is room."""
    if rupee_supported():
        chip = _render_text(text, height, fg, bg)
    else:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        chip = np.full((th + base + 10, tw + 12, 3), bg, np.uint8)
        cv2.putText(chip, text, (6, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv2.LINE_AA)
    top = y - chip.shape[0] if y - chip.shape[0] >= 0 else y
    _blit(frame, chip, x, top)


def draw_detection(frame: np.ndarray, det: Candidate) -> None:
    p1, p2 = (int(round(det.x1)), int(round(det.y1))), (int(round(det.x2)), int(round(det.y2)))
    cv2.rectangle(frame, p1, p2, GREEN, 3, cv2.LINE_AA)
    draw_label(frame, format_label(det.score), p1[0], p1[1])


def draw_hud(frame: np.ndarray, lines: list[str], color=WHITE) -> None:
    y = 22
    for ln in lines:
        cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def draw_status_banner(frame: np.ndarray, text: str, ok: bool) -> None:
    H, W = frame.shape[:2]
    color = GREEN if ok else (90, 90, 90)
    cv2.rectangle(frame, (0, H - 34), (W, H), color, -1)
    cv2.putText(frame, text, (10, H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


def draw_inspector(frame: np.ndarray, decisions: list[Decision], limit: int = 5) -> None:
    """Show rejected candidates dashed-grey with raw score and the reason they were rejected."""
    for d in sorted(decisions, key=lambda d: -d.candidate.score)[:limit]:
        if d.accepted:
            continue
        c = d.candidate
        p1, p2 = (int(c.x1), int(c.y1)), (int(c.x2), int(c.y2))
        cv2.rectangle(frame, p1, p2, GREY, 1, cv2.LINE_AA)
        cv2.putText(
            frame, f"rejected {c.score:.2f} ({d.reason}, cls {c.class_id})",
            (p1[0] + 3, max(12, p1[1] + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA,
        )


def is_windows() -> bool:
    return sys.platform.startswith("win")
