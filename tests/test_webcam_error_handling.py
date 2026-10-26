import numpy as np
import pytest

from detector.webcam import CameraError, FrameSource


class FakeCap:
    def __init__(self, opened=True, reads=None):
        self._opened, self.reads, self.released, self.props = opened, list(reads or []), False, {}

    def isOpened(self):
        return self._opened

    def set(self, k, v):
        self.props[k] = v

    def get(self, k):
        return 640.0

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self):
        self.released = True


GOOD = (True, np.zeros((48, 64, 3), np.uint8))


def factory_for(*caps):
    caps = list(caps)
    made = []

    def f(*args):
        cap = caps.pop(0) if caps else FakeCap(opened=False)
        made.append(cap)
        return cap

    f.made = made
    return f


def test_camera_unavailable_raises_friendly_error():
    with pytest.raises(CameraError, match="Could not open camera"):
        FrameSource(camera=0, _capture_factory=factory_for(*[FakeCap(opened=False)] * 5))


def test_negative_camera_index():
    with pytest.raises(CameraError):
        FrameSource(camera=-1, _capture_factory=factory_for())


def test_transient_read_failure_then_recovery():
    f = factory_for(FakeCap(reads=[(False, None), (False, None), GOOD]))
    src = FrameSource(_capture_factory=f, max_failures=10)
    assert src.read()[0] is False
    assert src.read()[0] is False
    ok, frame, status = src.read()
    assert ok and frame.shape == (48, 64, 3) and status == ""


def test_disconnect_triggers_reopen_then_gives_up_and_releases():
    f = factory_for(FakeCap(), FakeCap(), FakeCap(), FakeCap())
    src = FrameSource(_capture_factory=f, max_failures=2, max_reopens=2)
    with pytest.raises(CameraError, match="could not be recovered"):
        for _ in range(100):
            src.read()
    assert all(c.released for c in f.made)


def test_reopen_succeeds_and_streams_again():
    f = factory_for(FakeCap(), FakeCap(reads=[GOOD]))
    src = FrameSource(_capture_factory=f, max_failures=2, max_reopens=3)
    statuses = [src.read() for _ in range(2)]
    assert "reconnecting" in statuses[-1][2]
    for _ in range(5):
        ok, *_ = src.read()
        if ok:
            break
    assert ok


def test_release_is_idempotent():
    cap = FakeCap()
    src = FrameSource(_capture_factory=factory_for(cap))
    src.release()
    src.release()
    assert cap.released


def test_missing_source_file(tmp_path):
    with pytest.raises(CameraError, match="not found"):
        FrameSource(source=tmp_path / "x.jpg")


def test_still_image_source(tmp_path):
    import cv2
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.full((20, 30, 3), 7, np.uint8))
    with FrameSource(source=p) as s:
        ok, frame, _ = s.read()
        assert ok and frame.shape == (20, 30, 3)
