"""Review and draw bounding boxes (YOLO format, single class 0 = fifty_rupee_note).

    python scripts/annotate.py [--only-unlabeled] [--denomination 50_new]

Mouse : drag to draw a box; right-click removes the box under the cursor.
Keys  : s save & next | n next | p previous | c clear boxes | e confirm EMPTY (negative) |
        u undo last box | q quit
Every box is drawn by YOU; nothing is auto-annotated. Saving a target image with at least one box
sets label_status=annotated. Negatives (other denominations/objects) are saved with an empty label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from detector.dataset import DatasetPaths, load_manifest, parse_label_file, save_manifest, write_label_file

WIN = "annotate"


class Annotator:
    def __init__(self, paths: DatasetPaths, rows: list[dict], todo: list[dict]):
        self.paths, self.rows, self.todo = paths, rows, todo
        self.i = 0
        self.boxes: list[list[float]] = []  # pixel x1,y1,x2,y2
        self.drag: tuple[int, int] | None = None
        self.cur: tuple[int, int] | None = None
        self.img = None
        self.scale = 1.0

    def load(self) -> None:
        row = self.todo[self.i]
        self.img = cv2.imread(str(self.paths.images / row["filename"]))
        if self.img is None:
            self.img = np.zeros((480, 640, 3), np.uint8)
            print(f"warning: cannot read {row['filename']}")
        h, w = self.img.shape[:2]
        self.scale = min(1.0, 1100 / max(h, w))
        self.boxes = []
        lab = self.paths.labels / (row["image_id"] + ".txt")
        if lab.is_file():
            for _, cx, cy, bw, bh in parse_label_file(lab).boxes:
                self.boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])

    def render(self) -> None:
        row = self.todo[self.i]
        view = cv2.resize(self.img, None, fx=self.scale, fy=self.scale)
        for b in self.boxes:
            cv2.rectangle(view, (int(b[0] * self.scale), int(b[1] * self.scale)),
                          (int(b[2] * self.scale), int(b[3] * self.scale)), (0, 255, 0), 2)
        if self.drag and self.cur:
            cv2.rectangle(view, self.drag, self.cur, (0, 200, 255), 1)
        txt = f"[{self.i + 1}/{len(self.todo)}] {row['image_id']} {row['denomination']}/{row['side']} " \
              f"status={row['label_status']} boxes={len(self.boxes)}"
        cv2.putText(view, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, view)

    def mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag, self.cur = (x, y), (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.cur = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            x0, y0 = self.drag
            self.drag = None
            if abs(x - x0) > 8 and abs(y - y0) > 8:
                s = self.scale
                self.boxes.append([min(x0, x) / s, min(y0, y) / s, max(x0, x) / s, max(y0, y) / s])
        elif ev == cv2.EVENT_RBUTTONDOWN:
            px, py = x / self.scale, y / self.scale
            for k in reversed(range(len(self.boxes))):
                b = self.boxes[k]
                if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                    del self.boxes[k]
                    break

    def save(self, confirm_empty: bool = False) -> bool:
        row = self.todo[self.i]
        h, w = self.img.shape[:2]
        is_target = row["denomination"] in ("50_new", "mixed")
        if is_target and not self.boxes:
            print(f"{row['image_id']}: target image needs at least one box (or press n to skip)")
            return False
        if not is_target and self.boxes:
            print(f"{row['image_id']}: denomination '{row['denomination']}' is a negative; clear the boxes "
                  "or fix its denomination in the manifest")
            return False
        out = []
        for x1, y1, x2, y2 in self.boxes:
            x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
            out.append((0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        write_label_file(self.paths.labels / (row["image_id"] + ".txt"), out)
        row["label_status"] = "annotated" if out else "negative_confirmed"
        save_manifest(self.paths, self.rows)
        return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--only-unlabeled", action="store_true")
    p.add_argument("--denomination", default=None, help="restrict to one denomination (e.g. 50_new)")
    a = p.parse_args()
    paths = DatasetPaths(Path(a.dataset))
    rows = load_manifest(paths)
    todo = [r for r in rows if (not a.denomination or r["denomination"] == a.denomination)
            and (not a.only_unlabeled or r["label_status"] == "unlabeled")]
    if not todo:
        print("nothing to annotate")
        return 0
    ann = Annotator(paths, rows, todo)
    ann.load()
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, ann.mouse)
    while True:
        ann.render()
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and ann.save() and ann.i < len(todo) - 1:
            ann.i += 1
            ann.load()
        elif k == ord("e"):
            ann.boxes = []
            if ann.save() and ann.i < len(todo) - 1:
                ann.i += 1
                ann.load()
        elif k == ord("n") and ann.i < len(todo) - 1:
            ann.i += 1
            ann.load()
        elif k == ord("p") and ann.i > 0:
            ann.i -= 1
            ann.load()
        elif k == ord("c"):
            ann.boxes = []
        elif k == ord("u") and ann.boxes:
            ann.boxes.pop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
