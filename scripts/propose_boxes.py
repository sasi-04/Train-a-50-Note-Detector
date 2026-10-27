"""Propose bounding boxes for unlabeled Rs.50 images (colour segmentation) and review them.

    python scripts/propose_boxes.py                       # writes review sheets: reports/box_review/sheet_N.jpg
    python scripts/propose_boxes.py --accept --exclude 4,17,23 --drop-rejected

Proposals are NOT ground truth. Look at the sheets (red box = proposal, tile number in the corner), list the tiles
whose box is wrong in --exclude, then --accept writes labels (label_status=annotated) for the rest.
--drop-rejected deletes excluded/proposal-less target images so the dataset audit stays clean.
The segmentation looks for the note's pale blue-green ink, so it works best on plain backgrounds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from detector.dataset import DatasetPaths, load_manifest, save_manifest, write_label_file


def propose(img: np.ndarray, min_frac: float = 0.004):
    H, W = img.shape[:2]
    sc = 800 / max(H, W)
    sm = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(sm, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 62) & (h <= 104) & (s >= 16) & (s <= 190) & (v >= 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = st[k]
    if area / (sm.shape[0] * sm.shape[1]) < min_frac:
        return None
    pad = 0.04
    x1, y1, x2, y2 = x - pad * bw, y - pad * bh, x + bw * (1 + pad), y + bh * (1 + pad)
    return [max(0.0, x1 / sc), max(0.0, y1 / sc), min(float(W), x2 / sc), min(float(H), y2 / sc)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="reports/box_review")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--exclude", default="", help="comma-separated tile numbers whose proposal is wrong")
    ap.add_argument("--drop-rejected", action="store_true")
    ap.add_argument("--min-frac", type=float, default=0.004)
    a = ap.parse_args()

    paths = DatasetPaths(a.dataset)
    rows = load_manifest(paths)
    todo = [r for r in rows if r["denomination"] == "50_new" and r["label_status"] == "unlabeled"]
    if not todo:
        print("nothing to do: no unlabeled Rs.50 images")
        return 0
    exclude = {int(x) for x in a.exclude.split(",") if x.strip()}
    props = {}
    tiles = []
    for i, r in enumerate(todo, 1):
        img = cv2.imread(str(paths.images / r["filename"]))
        b = propose(img, a.min_frac) if img is not None else None
        props[r["image_id"]] = b
        t = img.copy()
        if b:
            cv2.rectangle(t, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), max(3, t.shape[1] // 200))
        t = cv2.resize(t, (320, int(320 * t.shape[0] / t.shape[1])))
        cv2.putText(t, str(i), (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(cv2.copyMakeBorder(t, 0, 245 - t.shape[0], 0, 2, cv2.BORDER_CONSTANT) if t.shape[0] <= 245 else t[:245])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols, per = 6, 24
    for n in range(0, len(tiles), per):
        chunk = tiles[n:n + per]
        while len(chunk) % cols:
            chunk.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"sheet_{n // per}.jpg"), np.vstack([np.hstack(chunk[j:j + cols]) for j in range(0, len(chunk), cols)]))
    print(f"{len(todo)} unlabeled Rs.50 images; {sum(b is not None for b in props.values())} have a proposal; "
          f"review sheets in {out}")
    if not a.accept:
        return 0
    accepted = rejected = 0
    drop = set()
    for i, r in enumerate(todo, 1):
        b = props[r["image_id"]]
        if b is None or i in exclude:
            rejected += 1
            drop.add(r["image_id"])
            continue
        img = cv2.imread(str(paths.images / r["filename"]))
        H, W = img.shape[:2]
        write_label_file(paths.labels / (r["image_id"] + ".txt"),
                         [(0, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H)])
        r["label_status"] = "annotated"
        r["notes"] += " | box: colour-proposed, visually reviewed"
        accepted += 1
    if a.drop_rejected and drop:
        for r in rows:
            if r["image_id"] in drop:
                (paths.images / r["filename"]).unlink(missing_ok=True)
                (paths.labels / (r["image_id"] + ".txt")).unlink(missing_ok=True)
        rows = [r for r in rows if r["image_id"] not in drop]
    save_manifest(paths, rows)
    print(f"accepted {accepted}, rejected {rejected}{' (deleted)' if a.drop_rejected else ' (still unlabeled: audit will fail)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
