"""Guided webcam capture: records the kind of frames test.py will actually see.

    python scripts/collect_webcam.py --session webcam_s01 --split train
    python scripts/collect_webcam.py --session webcam_s02 --split test     # a second, held-out sitting

For each stage the window tells you what to hold up. Press SPACE to start the stage (get the note ready first),
`s` to skip it, `q` to finish. A sharp frame is saved every --interval seconds while the stage runs. Frames are
marked source=webcam (supplementary data from the deployment camera; the assignment's primary data are your
phone photos). Rs.50 frames get no label yet (run scripts/propose_boxes.py); other notes and "no note" frames
get empty labels, so label them honestly: hold ONLY the note named on screen.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2

from capture_dataset import add_image
from detector.dataset import DatasetPaths, load_manifest, save_manifest
from detector.visualization import currency, put_text, text_size
from detector.webcam import CameraError, FrameSource

R = "₹"
STAGES = [
    ("50_new", "front", f"{R}50  FRONT", "New blue-green note, Gandhi portrait toward the camera. Move it: closer/farther, tilt, rotate."),
    ("50_new", "back", f"{R}50  BACK", "Same note, Hampi stone-chariot side toward the camera. Move it slowly."),
    ("50_new", "front", f"{R}50  FRONT (dim / sideways)", "Front again: turn the note sideways, tilt more, hold it lower, partly cover a corner with a finger."),
    ("10", "unknown", f"{R}10", "Hold ONLY the Rs.10 note. Show both sides, move it around."),
    ("20", "unknown", f"{R}20", "Hold ONLY the Rs.20 note. Show both sides."),
    ("100", "unknown", f"{R}100", "Hold ONLY the Rs.100 note. Show both sides."),
    ("200", "unknown", f"{R}200", "Hold ONLY the Rs.200 note. Show both sides."),
    ("500", "unknown", f"{R}500", "Hold ONLY the Rs.500 note. Show both sides."),
    ("background", "na", "NO NOTE", "No banknote in view. Face, hands, a card, a phone, paper, the room, lean left/right."),
]


def overlay(frame, title, line, big, color=(255, 255, 255)):
    out = frame.copy()
    H, W = out.shape[:2]
    roi = out[0:110, :]
    out[0:110, :] = cv2.addWeighted(roi, 0.35, roi * 0, 0.65, 0)
    put_text(out, title, 20, 10, 34, color, bold=True, shadow=False)
    put_text(out, line[:110], 20, 60, 20, (230, 230, 230), shadow=False)
    if big:
        tw, th = text_size(big, 64, True)
        put_text(out, big, (W - tw) // 2, (H - th) // 2, 64, (80, 200, 60), bold=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--session", required=True, help="e.g. webcam_s01 (one sitting = one session)")
    ap.add_argument("--split", default="", choices=["", "train", "val", "test"], help="assign these frames to a split")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20.0, help="duration of each stage")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between saved frames")
    ap.add_argument("--lighting", default="unknown", choices=["bright", "normal", "dim", "uneven", "shadow", "unknown"])
    ap.add_argument("--stages", default="all", help="comma list of stage numbers (1-based) or 'all'")
    ap.add_argument("--auto", action="store_true", help="start each stage without waiting for SPACE (testing)")
    ap.add_argument("--no-display", action="store_true", help="no window (testing)")
    a = ap.parse_args()

    chosen = list(range(len(STAGES))) if a.stages == "all" else [int(x) - 1 for x in a.stages.split(",")]
    paths = DatasetPaths(a.dataset)
    paths.ensure()
    rows = load_manifest(paths)
    known = {r["sha256"] for r in rows if r["sha256"]}
    try:
        cam = FrameSource(camera=a.camera)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    total = 0
    quit_all = False
    recent_sharp: list[float] = []
    try:
        for idx in chosen:
            den, side, title, tip = STAGES[idx]
            # -- wait for SPACE --------------------------------------------------------------------
            started = a.auto
            while not started and not quit_all:
                ok, frame, _ = cam.read()
                if not ok:
                    continue
                cv2.imshow("collect", overlay(frame, f"[{idx + 1}/{len(STAGES)}]  {title}", tip + "   SPACE = start, s = skip, q = finish", ""))
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    started = True
                elif k == ord("s"):
                    break
                elif k in (ord("q"), 27):
                    quit_all = True
            if quit_all:
                break
            if not started:
                continue
            # -- record -----------------------------------------------------------------------------
            t0 = last = time.time()
            saved = 0
            while time.time() - t0 < a.seconds:
                ok, frame, _ = cam.read()
                if not ok:
                    continue
                now = time.time()
                if now - last >= a.interval:
                    sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    recent_sharp.append(sharp)
                    del recent_sharp[:-20]
                    med = sorted(recent_sharp)[len(recent_sharp) // 2]
                    if sharp >= max(3.0, 0.6 * med):  # skip black start-up frames and motion-blurred ones (webcams are soft: compare to its own norm)
                        meta = {"denomination": den, "side": side, "lighting": a.lighting, "angle": "unknown",
                                "background": "cluttered" if den == "background" else "hand_held", "source": "webcam",
                                "session": a.session, "notes": f"guided webcam capture: {title}"}
                        row = add_image(paths, rows, frame, meta, known)
                        if row is not None:
                            row["split"] = a.split
                            saved += 1
                            total += 1
                    last = now
                if not a.no_display:
                    left = a.seconds - (now - t0)
                    cv2.imshow("collect", overlay(frame, f"RECORDING  {title}", f"{tip}   saved {saved}", f"{left:0.0f}", (60, 60, 230)))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        quit_all = True
                        break
            save_manifest(paths, rows)
            print(f"stage {idx + 1} ({title}): saved {saved} frames")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        save_manifest(paths, rows)
    print(f"done: {total} frames in session {a.session}")
    if any(STAGES[i][0] == "50_new" for i in chosen):
        print("next: python scripts/propose_boxes.py   (boxes for the Rs.50 frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
