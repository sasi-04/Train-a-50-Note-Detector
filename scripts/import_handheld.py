"""Import reviewed hand-held frames from the developer's scanner-app PDFs (see handheld_import.yaml).

    python scripts/import_handheld.py --pdf-dir C:/Users/me/Downloads

Each listed page is cropped to a square window (full width, centred on the note box; upper body for
negatives) so notes appear at webcam-like scale, then written as JPEG with its label (box clipped to the
window; negatives get an empty label). Re-running skips exact duplicates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import yaml

from capture_dataset import add_image
from detector.dataset import DatasetPaths, load_manifest, save_manifest, write_label_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--map", default="dataset/metadata/handheld_import.yaml")
    ap.add_argument("--dataset", default="dataset")
    a = ap.parse_args()
    from pypdf import PdfReader

    cfg = yaml.safe_load(Path(a.map).read_text(encoding="utf-8"))
    readers = {k: PdfReader(str(Path(a.pdf_dir) / v)) for k, v in cfg["pdfs"].items()}
    paths = DatasetPaths(a.dataset)
    paths.ensure()
    rows = load_manifest(paths)
    known = {r["sha256"] for r in rows if r["sha256"]}
    added = skipped = 0
    for it in cfg["items"]:
        key, num = it["page"][0], int(it["page"][1:])
        raw = np.frombuffer(readers[key].pages[num - 1].images[0].data, np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        win = min(W, H)
        if it["box_px"]:
            x1, y1, x2, y2 = it["box_px"]
            cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
        else:
            cy, cx = H * 0.45, W / 2
        y0 = int(min(max(0, cy - win / 2), H - win))
        x0 = int(min(max(0, cx - win / 2), W - win))
        crop = img[y0:y0 + win, x0:x0 + win]
        meta = {"denomination": "50_new" if it["target"] else "background", "side": it["side"], "lighting": "unknown",
                "angle": "unknown", "background": "hand_held" if it["target"] else "cluttered", "source": "phone",
                "session": it["session"], "notes": f"hand-held frame {it['page']} (scanner-app pdf)"}
        row = add_image(paths, rows, crop, meta, known)
        if row is None:
            skipped += 1
            continue
        row["split"] = it["split"]
        if it["target"]:
            bx1, by1 = max(0, x1 - x0), max(0, y1 - y0)
            bx2, by2 = min(win, x2 - x0), min(win, y2 - y0)
            write_label_file(paths.labels / (row["image_id"] + ".txt"),
                             [(0, (bx1 + bx2) / 2 / win, (by1 + by2) / 2 / win, (bx2 - bx1) / win, (by2 - by1) / win)])
            row["label_status"] = "annotated"
            row["notes"] += " | box: colour-proposed, visually reviewed"
        added += 1
    save_manifest(paths, rows)
    print(f"imported {added} images ({skipped} duplicates skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
