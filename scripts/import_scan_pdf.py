"""Import the developer's own scanner-app PDF (one photographed note per page) into the dataset.

    python scripts/import_scan_pdf.py "C:/path/Scan documents.pdf" --map dataset/metadata/scan_import.yaml

The YAML maps pages to denomination/side/session/split and documents why. Pages are rotated upright and
re-encoded as JPEG (long side <= 1600). Target (50_new) images get NO label until boxes are drawn/reviewed
(see scripts/annotate.py); negatives get empty labels. Re-running skips exact duplicates.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import yaml

from capture_dataset import add_image
from detector.dataset import DatasetPaths, load_manifest, save_manifest

ROT = {"ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--map", default="dataset/metadata/scan_import.yaml")
    ap.add_argument("--dataset", default="dataset")
    a = ap.parse_args()
    try:
        from pypdf import PdfReader
    except ImportError:
        print("error: pip install pypdf", file=sys.stderr)
        return 3
    cfg = yaml.safe_load(Path(a.map).read_text(encoding="utf-8"))
    reader = PdfReader(a.pdf)
    paths = DatasetPaths(a.dataset)
    paths.ensure()
    rows = load_manifest(paths)
    known = {r["sha256"] for r in rows if r["sha256"]}
    seen_pages: set[int] = set()
    added = skipped = 0
    for grp in cfg["pages"]:
        for pg in grp["pages"]:
            if pg in seen_pages:
                print(f"error: page {pg} mapped twice", file=sys.stderr)
                return 2
            seen_pages.add(pg)
            page = reader.pages[pg - 1]
            raw = np.frombuffer(page.images[0].data, np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                print(f"error: page {pg} image undecodable", file=sys.stderr)
                return 2
            if cfg.get("rotate") in ROT:
                img = cv2.rotate(img, ROT[cfg["rotate"]])
            meta = {"denomination": grp["denomination"], "side": grp["side"], "lighting": "unknown",
                    "angle": "unknown", "background": grp.get("background", "unknown"), "source": cfg["source"],
                    "session": grp["session"], "notes": f"{cfg['note']} {pg}"}
            row = add_image(paths, rows, img, meta, known)
            if row is None:
                skipped += 1
            else:
                row["split"] = grp["split"]
                added += 1
    missing = set(range(1, len(reader.pages) + 1)) - seen_pages
    save_manifest(paths, rows)
    print(f"imported {added} images ({skipped} duplicates skipped); unmapped pages: {sorted(missing) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
