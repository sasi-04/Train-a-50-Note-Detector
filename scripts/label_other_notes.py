"""Write 'other note' (class 1) boxes for other-denomination images from a reviewed boxes JSON.

    python scripts/label_other_notes.py boxes.json [--mark-rest-unlabeled]

boxes.json maps image_id -> [x1, y1, x2, y2] in pixels of the stored image. Images are only touched if
their denomination is 10/20/100/200/500. With --mark-rest-unlabeled every other-denomination image NOT in
the file is set to label_status=unlabeled (empty label), which excludes it from training until it is
boxed (e.g. with scripts/propose_other_boxes.py) or removed.
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
import cv2

from detector.dataset import OTHER_CLASS_ID, OTHER_NOTE_DENOMS, DatasetPaths, load_manifest, save_manifest, write_label_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--note", default="box: reviewed (other note, class 1)")
    ap.add_argument("--mark-rest-unlabeled", action="store_true")
    a = ap.parse_args()
    boxes = json.load(open(a.boxes))
    paths = DatasetPaths(a.dataset)
    rows = load_manifest(paths)
    done = rest = 0
    for r in rows:
        if r["denomination"] not in OTHER_NOTE_DENOMS:
            continue
        b = boxes.get(r["image_id"])
        if b is None:
            if a.mark_rest_unlabeled and r["label_status"] != "annotated":
                r["label_status"] = "unlabeled"
                write_label_file(paths.labels / (r["image_id"] + ".txt"), [])
                rest += 1
            continue
        img = cv2.imread(str(paths.images / r["filename"]))
        H, W = img.shape[:2]
        x1, y1, x2, y2 = (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            print(f"skip {r['image_id']}: degenerate box", file=sys.stderr)
            continue
        write_label_file(paths.labels / (r["image_id"] + ".txt"),
                         [(OTHER_CLASS_ID, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)])
        r["label_status"] = "annotated"
        if a.note not in r["notes"]:
            r["notes"] += " | " + a.note
        done += 1
    save_manifest(paths, rows)
    print(f"boxed {done} other-note images; {rest} marked unlabeled (excluded from training)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
