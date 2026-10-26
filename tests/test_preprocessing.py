import numpy as np
import pytest

from detector.preprocessing import InvalidImageError, letterbox, preprocess


def test_letterbox_landscape_geometry():
    img = np.zeros((720, 1280, 3), np.uint8)
    boxed, meta = letterbox(img, 640)
    assert boxed.shape == (640, 640, 3)
    assert meta.ratio == pytest.approx(0.5)
    assert meta.pad_x == 0 and meta.pad_y == 140
    assert (meta.orig_w, meta.orig_h) == (1280, 720)


def test_letterbox_portrait_geometry():
    img = np.zeros((1280, 720, 3), np.uint8)
    _, meta = letterbox(img, 640)
    assert meta.pad_y == 0 and meta.pad_x == 140


def test_padding_colour_is_114():
    boxed, _ = letterbox(np.zeros((720, 1280, 3), np.uint8), 640)
    assert (boxed[0] == 114).all() and (boxed[-1] == 114).all()


def test_blob_contract_is_rgb_nchw_float01():
    img = np.zeros((480, 640, 3), np.uint8)
    img[..., 0] = 255  # BGR: pure blue
    blob, _ = preprocess(img, 320)
    assert blob.shape == (1, 3, 320, 320) and blob.dtype == np.float32
    assert 0.0 <= blob.min() and blob.max() <= 1.0
    centre = blob[0, :, 160, 160]
    assert centre[2] == pytest.approx(1.0) and centre[0] == pytest.approx(0.0)  # RGB order: blue is channel 2


@pytest.mark.parametrize("bad", [None, "x", np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32),
                                 np.zeros((10, 10, 4), np.uint8), np.zeros((1, 1, 3), np.uint8)])
def test_invalid_images_rejected(bad):
    with pytest.raises(InvalidImageError):
        preprocess(bad, 320)
