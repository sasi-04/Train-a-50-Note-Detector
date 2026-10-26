import numpy as np
import pytest

from detector.postprocessing import OutputFormatError, decode_output, describe_layout, nms
from detector.preprocessing import LetterboxMeta, letterbox

META = LetterboxMeta(ratio=0.5, pad_x=0, pad_y=140, orig_w=1280, orig_h=720, size=640)


def head(dets, n=50):
    out = np.zeros((1, 5, n), np.float32)
    for i, d in enumerate(dets):
        out[0, :, i] = d
    return out


def test_layout_detection():
    assert describe_layout((1, 5, 8400)) == "channels_first"
    assert describe_layout((1, 8400, 5)) == "channels_last"
    assert describe_layout((1, 300, 6)) == "end2end"
    with pytest.raises(OutputFormatError):
        describe_layout((2, 5, 100))
    with pytest.raises(OutputFormatError):
        describe_layout((1, 7, 7))


def test_coordinates_map_back_to_original_frame():
    # model-space box centred (320, 320), 200x100 -> original: x (320-0)/0.5, y (320-140)/0.5
    c = decode_output(head([(320, 320, 200, 100, 0.9)]), META)
    assert len(c) == 1
    assert c[0].box == pytest.approx((440, 260, 840, 460), abs=1e-3)


def test_letterbox_roundtrip_of_known_box():
    img = np.zeros((720, 1280, 3), np.uint8)
    _, meta = letterbox(img, 640)
    x1, y1, x2, y2 = 400, 200, 900, 600
    r = meta.ratio
    cx, cy, w, h = ((x1 + x2) / 2 * r + meta.pad_x, (y1 + y2) / 2 * r + meta.pad_y, (x2 - x1) * r, (y2 - y1) * r)
    c = decode_output(head([(cx, cy, w, h, 0.8)]), meta)[0]
    assert c.box == pytest.approx((x1, y1, x2, y2), abs=0.5)


def test_channels_last_equivalent():
    a = decode_output(head([(320, 320, 200, 100, 0.9)]), META)
    b = decode_output(np.transpose(head([(320, 320, 200, 100, 0.9)], n=50), (0, 2, 1)), META)
    assert a == b


def test_score_floor_and_nms():
    out = head([(320, 320, 200, 100, 0.9), (322, 321, 200, 100, 0.8), (100, 300, 40, 40, 0.04)])
    c = decode_output(out, META, score_floor=0.05, nms_iou=0.45)
    assert [round(x.score, 2) for x in c] == [0.9]  # duplicate suppressed, sub-floor dropped


def test_empty_output_gives_no_candidates():
    assert decode_output(np.zeros((1, 5, 100), np.float32), META) == []


def test_boxes_clipped_to_frame():
    c = decode_output(head([(5, 320, 200, 100, 0.9)]), META)[0]
    assert c.x1 == 0.0


def test_degenerate_box_dropped():
    assert decode_output(head([(320, 320, 0.5, 0.5, 0.9)]), META) == []


@pytest.mark.parametrize("bad", [np.full((1, 5, 10), np.nan, np.float32), np.zeros((1, 5, 10), np.int32)])
def test_bad_tensor_rejected(bad):
    with pytest.raises(OutputFormatError):
        decode_output(bad, META)


def test_multiclass_head_reports_class_ids():
    out = np.zeros((1, 6, 10), np.float32)
    out[0, :, 0] = (320, 320, 100, 100, 0.2, 0.7)
    c = decode_output(out, META)
    assert c[0].class_id == 1 and c[0].score == pytest.approx(0.7)


def test_end2end_layout():
    out = np.zeros((1, 10, 6), np.float32)
    out[0, 0] = (100, 200, 300, 400, 0.85, 0)
    c = decode_output(out, LetterboxMeta(1.0, 0, 0, 640, 640, 640))
    assert c[0].box == pytest.approx((100, 200, 300, 400)) and c[0].score == pytest.approx(0.85)


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [1, 1, 11, 11]], float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45)) == [0, 1]
