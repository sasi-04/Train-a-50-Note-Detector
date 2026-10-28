"""The rotated-copy augmentation must keep boxes on the note after a 90-degree turn."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from detector.dataset import DatasetPaths, parse_label_file, write_label_file

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_train():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("train_script", SCRIPTS / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_box_follows_the_note_after_rotation(tmp_path):
    train = load_train()
    paths = DatasetPaths(tmp_path / "ds")
    paths.ensure()
    img = np.zeros((100, 200, 3), np.uint8)          # 200 wide, 100 tall
    img[20:40, 120:180] = (0, 255, 0)                # green 'note' at x 120-180, y 20-40
    cv2.imwrite(str(paths.images / "a.jpg"), img)
    write_label_file(paths.labels / "a.txt", [(0, 150 / 200, 30 / 100, 60 / 200, 20 / 100)])
    for seed in (1, 2, 3, 4):                         # both rotation directions get exercised
        listed = train.make_rot90_copies(paths, [{"image_id": "a", "filename": "a.jpg"}], tmp_path / f"rot{seed}", seed=seed)
        assert len(listed) == 1
        out = cv2.imread(listed[0])
        assert out.shape[:2] == (200, 100)            # width/height swapped
        lab = tmp_path / f"rot{seed}" / "labels" / (Path(listed[0]).stem + ".txt")
        (c, cx, cy, w, h), = parse_label_file(lab).boxes
        H, W = out.shape[:2]
        x1, x2, y1, y2 = int((cx - w / 2) * W), int((cx + w / 2) * W), int((cy - h / 2) * H), int((cy + h / 2) * H)
        inside = out[y1 + 2:y2 - 2, x1 + 2:x2 - 2]
        assert inside.size and (inside[..., 1] > 200).mean() > 0.95   # the box encloses the green note
        assert w < h                                                    # the note is now upright


def test_images_without_boxes_are_skipped(tmp_path):
    train = load_train()
    paths = DatasetPaths(tmp_path / "ds")
    paths.ensure()
    cv2.imwrite(str(paths.images / "b.jpg"), np.full((50, 80, 3), 120, np.uint8))
    write_label_file(paths.labels / "b.txt", [])
    assert train.make_rot90_copies(paths, [{"image_id": "b", "filename": "b.jpg"}], tmp_path / "rot") == []
