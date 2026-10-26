"""Frame sources: webcam (with reconnect) or an image/video file for headless runs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .logging_utils import get_logger


class CameraError(RuntimeError):
    """The camera could not be opened or recovered."""


def _backends() -> list[int]:
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


class FrameSource:
    """Uniform ``read()`` over a camera index, a video file or a single image.

    Camera read failures are retried; after ``max_failures`` consecutive failures the device is
    re-opened up to ``max_reopens`` times before CameraError is raised.
    """

    def __init__(
        self,
        camera: int = 0,
        source: str | Path | None = None,
        width: int | None = 1280,
        height: int | None = 720,
        max_failures: int = 15,
        max_reopens: int = 3,
        _capture_factory=None,
    ):
        self.log = get_logger()
        self.camera, self.source = camera, Path(source) if source else None
        self.width, self.height = width, height
        self.max_failures, self.max_reopens = max_failures, max_reopens
        self._factory = _capture_factory or cv2.VideoCapture
        self._cap = None
        self._still: np.ndarray | None = None
        self._first: np.ndarray | None = None  # first frame of a video file, read while validating it
        self._failures = 0
        self._reopens = 0
        self.is_camera = self.source is None
        self.frames_read = 0
        self._open()

    # -- opening -----------------------------------------------------------------------------
    def _open(self) -> None:
        if self.source is not None:
            if not self.source.is_file():
                raise CameraError(f"Source file not found: {self.source}")
            img = cv2.imread(str(self.source))
            if img is not None:
                self._still = img
                return
            cap = self._factory(str(self.source))
            ok, first = cap.read() if cap.isOpened() else (False, None)
            if not ok or first is None:
                cap.release()
                raise CameraError(f"Could not read {self.source} as an image or video")
            self._first = first
            self._cap = cap
            return
        if self.camera < 0:
            raise CameraError(f"Camera index must be >= 0 (got {self.camera})")
        for backend in _backends():
            cap = self._factory(self.camera, backend)
            if cap.isOpened():
                if self.width and self.height:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap = cap
                self.log.info(
                    "Camera %d opened (%dx%d)", self.camera,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                return
            cap.release()
        raise CameraError(
            f"Could not open camera {self.camera}. Check that it is connected, not used by another "
            "application (Zoom/Teams/browser), and that camera access is enabled in your OS privacy "
            "settings. Try --camera 1, or --source <image|video> to run without a webcam."
        )

    # -- reading -----------------------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None, str]:
        """Return (ok, frame, status). status is '' when healthy, otherwise a user-facing message.

        Raises CameraError once recovery is exhausted; returns (False, None, 'end of ...') at EOF.
        """
        if self._still is not None:
            self.frames_read += 1
            return True, self._still.copy(), ""
        if self._first is not None:
            frame, self._first = self._first, None
            self.frames_read += 1
            return True, frame, ""
        ok, frame = self._cap.read() if self._cap is not None else (False, None)
        if ok and frame is not None and frame.size:
            self._failures = 0
            self.frames_read += 1
            return True, frame, ""
        if not self.is_camera:
            return False, None, "end of video"
        self._failures += 1
        if self._failures < self.max_failures:
            time.sleep(0.02)
            return False, None, "camera read failed, retrying..."
        return self._recover()

    def _recover(self) -> tuple[bool, np.ndarray | None, str]:
        if self._reopens >= self.max_reopens:
            self.release()
            raise CameraError("Camera stopped delivering frames and could not be recovered (disconnected?).")
        self._reopens += 1
        self.log.warning("Camera stalled; re-opening (attempt %d/%d)", self._reopens, self.max_reopens)
        self.release()
        time.sleep(0.5)
        try:
            self._open()
        except CameraError:
            pass  # next read() will fail again and eventually exhaust reopens
        self._failures = 0
        return False, None, f"camera lost, reconnecting ({self._reopens}/{self.max_reopens})..."

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
