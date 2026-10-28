"""Model-assisted boxing of other-denomination frames: propose class-1 ('other note') boxes, review, accept.

    python scripts/propose_other_boxes.py --model runs/rs50_r1/model.onnx            # writes review sheets
    python scripts/propose_other_boxes.py --model ... --accept --exclude 4,17,23 --drop-rejected

A helper model (trained only on the already-boxed frames) proposes ONE box per image: its best plausible
'other note' candidate. Proposals are NOT ground truth: open reports/other_box_review/sheet_*.jpg (red box,
tile number top-left), list the tiles whose box is wrong in --exclude, then --accept writes labels for the rest.
Images without a proposal or excluded stay unlabeled (excluded from training) or are deleted with --drop-rejected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from detector.dataset import OTHER_CLASS_ID, OTHER_NOTE_DENOMS, DatasetPaths, load_manifest, save_manifest, write_label_file
from detector.inference import OnnxDetector
from detector.thresholding import plausible_shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="reports/other_box_review")
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--drop-rejected", action="store_true")
    a = ap.parse_args()

    paths = DatasetPaths(a.dataset)
    rows = load_manifest(paths)
    todo = sorted([r for r in rows if r["denomination"] in OTHER_NOTE_DENOMS and r["label_status"] == "unlabeled"],
                  key=lambda r: r["image_id"])
    if not todo:
        print("nothing to do: no unlabeled other-note images")
        return 0
    det = OnnxDetector(a.model, score_floor=0.05)
    props: dict[str, tuple | None] = {}
    tiles = []
    for i, r in enumerate(todo, 1):
        img = cv2.imread(str(paths.images / r["filename"]))
        H, W = img.shape[:2]
        c = [x for x in det.infer(img).candidates
             if x.class_id == OTHER_CLASS_ID and x.score >= a.min_score and plausible_shape(x.box, W, H)]
        props[r["image_id"]] = c[0].box if c else None
        t = img.copy()
        if c:
            b = c[0].box
            cv2.rectangle(t, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), max(3, W // 160))
        t = cv2.resize(t, (320, 180))
        cv2.putText(t, f"{i} {r['denomination']}" + ("" if c else " NONE"), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        tiles.append(t)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per = 30
    for n in range(0, len(tiles), per):
        chunk = tiles[n:n + per]
        while len(chunk) % 6:
            chunk.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"sheet_{n // per}.jpg"), np.vstack([np.hstack(chunk[j:j + 6]) for j in range(0, len(chunk), 6)]))
    have = sum(v is not None for v in props.values())
    print(f"{len(todo)} unlabeled other-note images; {have} have a proposal; review sheets in {out}")
    if not a.accept:
        return 0
    exclude = {int(x) for x in a.exclude.split(",") if x.strip()}
    ok = bad = 0
    drop = set()
    for i, r in enumerate(todo, 1):
        b = props[r["image_id"]]
        if b is None or i in exclude:
            bad += 1
            drop.add(r["image_id"])
            continue
        img = cv2.imread(str(paths.images / r["filename"]))
        H, W = img.shape[:2]
        write_label_file(paths.labels / (r["image_id"] + ".txt"),
                         [(OTHER_CLASS_ID, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H)])
        r["label_status"] = "annotated"
        r["notes"] += " | box: model-proposed, visually reviewed (other note, class 1)"
        ok += 1
    if a.drop_rejected and drop:
        for r in rows:
            if r["image_id"] in drop:
                (paths.images / r["filename"]).unlink(missing_ok=True)
                (paths.labels / (r["image_id"] + ".txt")).unlink(missing_ok=True)
        rows = [r for r in rows if r["image_id"] not in drop]
    save_manifest(paths, rows)
    print(f"accepted {ok}, rejected {bad}{' (deleted)' if a.drop_rejected else ' (left unlabeled, excluded from training)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
