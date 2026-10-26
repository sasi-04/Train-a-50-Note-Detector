"""Collect dataset images and their metadata.

Two modes:

  ingest   Import photographs you took with your PHONE (the assignment's required source).
           Re-encodes to JPEG (long side capped), which also strips EXIF/GPS data.
           python scripts/capture_dataset.py ingest D:/photos/s01 --denomination 50_new --side front \
               --lighting bright --background plain --session s01 --source phone

  webcam   Optional supplementary capture from the deployment webcam (SPACE saves, q quits).
           Marked source=webcam so the audit shows how much of the data is not phone-taken.

New images get an EMPTY label placeholder only for negatives; target images are left without a
label until annotated with scripts/annotate.py. Metadata you do not know stays 'unknown'.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2

from detector.dataset import (
    ANGLES, BACKGROUNDS, DENOMINATIONS, IMAGE_EXTS, LIGHTING, SIDES, SOURCES, DatasetPaths,
    load_manifest, save_manifest, sha256_file, write_label_file,
)

MAX_SIDE = 1600


def next_index(rows: list[dict], prefix: str) -> int:
    nums = [int(r["image_id"].rsplit("_", 1)[1]) for r in rows
            if r["image_id"].startswith(prefix + "_") and r["image_id"].rsplit("_", 1)[1].isdigit()]
    return max(nums, default=0) + 1


def add_image(paths: DatasetPaths, rows: list[dict], img, meta: dict, known_hashes: set[str]) -> dict | None:
    """Write one image + manifest row. Returns the row, or None if it is an exact duplicate."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    prefix = f"{meta['session']}_{meta['denomination']}_{meta['side']}"
    image_id = f"{prefix}_{next_index(rows, prefix):04d}"
    fname = image_id + ".jpg"
    dest = paths.images / fname
    ok = cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError(f"could not write {dest}")
    digest = sha256_file(dest)
    if digest in known_hashes:
        dest.unlink()
        return None
    known_hashes.add(digest)
    row = {"image_id": image_id, "filename": fname, "split": "", "sha256": digest, "label_status": "unlabeled", **meta}
    row.pop("_unused", None)
    if meta["denomination"] not in ("50_new", "mixed", "unknown"):
        write_label_file(paths.labels / (image_id + ".txt"), [])  # genuine negative: empty label
        row["label_status"] = "negative_confirmed"
    rows.append(row)
    return row


def meta_from_args(a: argparse.Namespace) -> dict:
    return {"denomination": a.denomination, "side": a.side, "lighting": a.lighting, "angle": a.angle,
            "background": a.background, "source": a.source, "session": a.session, "notes": a.notes or ""}


def cmd_ingest(a: argparse.Namespace) -> int:
    src = Path(a.folder)
    if not src.is_dir():
        print(f"error: folder not found: {src}", file=sys.stderr)
        return 2
    paths = DatasetPaths(Path(a.dataset))
    paths.ensure()
    rows = load_manifest(paths)
    known = {r["sha256"] for r in rows if r["sha256"]}
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    added = dup = bad = 0
    for p in files:
        img = cv2.imread(str(p))  # honours EXIF orientation; re-encoding drops EXIF/GPS
        if img is None:
            print(f"skip (unreadable): {p.name}")
            bad += 1
            continue
        if add_image(paths, rows, img, meta_from_args(a), known) is None:
            dup += 1
        else:
            added += 1
    save_manifest(paths, rows)
    print(f"ingested {added} images from {src} ({dup} duplicates skipped, {bad} unreadable)")
    if a.denomination in ("50_new", "mixed"):
        print("next: annotate with  python scripts/annotate.py")
    return 0


def cmd_webcam(a: argparse.Namespace) -> int:
    paths = DatasetPaths(Path(a.dataset))
    paths.ensure()
    rows = load_manifest(paths)
    known = {r["sha256"] for r in rows if r["sha256"]}
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print(f"error: cannot open camera {a.camera}", file=sys.stderr)
        return 4
    meta = meta_from_args(a)
    meta["source"] = "webcam"
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            view = frame.copy()
            cv2.putText(view, f"{meta['session']} {meta['denomination']} {meta['side']}  saved={saved}  SPACE=save q=quit",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("capture", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                if add_image(paths, rows, frame, meta, known):
                    saved += 1
                    save_manifest(paths, rows)
            elif k in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"saved {saved} webcam images")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "webcam"):
        s = sub.add_parser(name)
        if name == "ingest":
            s.add_argument("folder", help="folder of phone photos (one session, one denomination/side/condition)")
        else:
            s.add_argument("--camera", type=int, default=0)
        s.add_argument("--denomination", required=True, choices=DENOMINATIONS)
        s.add_argument("--side", default="unknown", choices=SIDES, help="front/back (na for non-notes)")
        s.add_argument("--lighting", default="unknown", choices=LIGHTING)
        s.add_argument("--angle", default="unknown", choices=ANGLES)
        s.add_argument("--background", default="unknown", choices=BACKGROUNDS)
        s.add_argument("--source", default="phone" if name == "ingest" else "webcam", choices=SOURCES)
        s.add_argument("--session", required=True, help="capture session id, e.g. s01 (sessions are never split)")
        s.add_argument("--notes", default="")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    sys.exit(cmd_ingest(a) if a.cmd == "ingest" else cmd_webcam(a))
