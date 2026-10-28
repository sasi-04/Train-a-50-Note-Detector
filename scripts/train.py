"""Train the single-class Rs.50 detector with Ultralytics YOLOv8n on the LOCAL dataset only.

    python scripts/train.py                      # uses configs/train.yaml
    python scripts/train.py --epochs 60 --imgsz 640 --name rs50_640
    python scripts/train.py --resume --name rs50 # continue an interrupted run

Refuses to start unless `scripts/validate_dataset.py` would pass (valid labels, both sides of the
target, every required negative denomination, leak-free session splits). `--allow-small` overrides the
coverage gate for pipeline smoke tests only and stamps the run as NOT a deliverable.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from detector import OTHER_CLASS_NAME, TARGET_CLASS_NAME
from detector.dataset import DatasetPaths, audit_dataset, load_manifest, manifest_hash, sha256_file


def make_rot90_copies(paths: DatasetPaths, rows: list[dict], out_dir: Path, seed: int = 0) -> list[str]:
    """Training-time augmentation: a 90-degree rotated copy of every labelled-note TRAIN image, so upright notes are seen.

    Copies (image + transformed YOLO label) live under ``out_dir`` inside the run folder, never in dataset/, and are
    derived only from the given (training) rows, so no val/test image leaks in. Direction is random per image but seeded.
    """
    import random

    import cv2

    from detector.dataset import parse_label_file, write_label_file

    rng = random.Random(seed)
    img_dir, lab_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    listed: list[str] = []
    for r in rows:
        boxes = parse_label_file(paths.labels / (r["image_id"] + ".txt"), 2).boxes
        if not boxes:  # no-note images: nothing to teach about orientation
            continue
        img = cv2.imread(str(paths.images / r["filename"]))
        if img is None:
            continue
        cw = rng.random() < 0.5
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if cw else cv2.ROTATE_90_COUNTERCLOCKWISE)
        new = []
        for c, cx, cy, w, h in boxes:
            # clockwise: (x, y) -> (1 - y, x); counter-clockwise: (x, y) -> (y, 1 - x); width/height swap
            new.append((c, (1 - cy) if cw else cy, cx if cw else (1 - cx), h, w))
        stem = r["image_id"] + ("_rcw" if cw else "_rccw")
        cv2.imwrite(str(img_dir / (stem + ".jpg")), rot, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_label_file(lab_dir / (stem + ".txt"), new)
        listed.append(str((img_dir / (stem + ".jpg")).resolve()))
    return listed


def write_data_yaml(paths: DatasetPaths, run_dir: Path, final_fit: bool = False, rot90: bool = False,
                    seed: int = 0) -> Path:
    """Write absolute-path split lists + data.yaml from the manifest (labels resolved by /images/->/labels/)."""
    rows = load_manifest(paths)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        # images still 'unlabeled' (other-denomination notes not yet boxed) must not teach "no note here"
        sel = [r for r in rows if (r["split"] == sp or (final_fit and sp == "train")) and r["label_status"] != "unlabeled"]
        lst = [str((paths.images / r["filename"]).resolve()) for r in sel]
        if rot90 and sp == "train":
            lst += make_rot90_copies(paths, sel, data_dir / "rot90" , seed)  # /images/ -> /labels/ mapping holds inside rot90/
        (data_dir / f"{sp}.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")
    data = {"path": str(data_dir.resolve()), "train": "train.txt", "val": "val.txt", "test": "test.txt",
            "names": {0: TARGET_CLASS_NAME, 1: OTHER_CLASS_NAME}}
    p = data_dir / "data.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-small", action="store_true", help="skip the coverage gate (smoke tests only)")
    ap.add_argument("--allow-no-objects", action="store_true",
                    help="train without object/background negatives (recorded in run_info.json)")
    ap.add_argument("--rot90", action="store_true",
                    help="also train on a 90-degree rotated copy of every labelled-note training image (upright notes)")
    ap.add_argument("--init", default=None, help="start from this checkpoint instead of configs/train.yaml `model` (fine-tuning)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any `train:` key, e.g. --set close_mosaic=3 --set lr0=0.002 (repeatable)")
    ap.add_argument("--final-fit", action="store_true",
                    help="train on ALL images (train+val+test) for the shipped model; Ultralytics 'val' is then "
                         "just a training-progress monitor and NO held-out metric from this run is valid")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    tr = dict(cfg["train"])
    for k in ("epochs", "imgsz", "batch", "device"):
        if getattr(a, k) is not None:
            tr[k] = getattr(a, k)
    for kv in a.set:
        k, _, v = kv.partition("=")
        tr[k] = yaml.safe_load(v)
    seed = a.seed if a.seed is not None else cfg.get("seed", 0)
    name = a.name or cfg.get("run_name", "rs50")
    dataset_root = Path(a.dataset or cfg.get("dataset", "dataset"))
    run_root = Path(cfg.get("run_dir", "runs")).resolve()
    run_dir = run_root / name

    audit = audit_dataset(dataset_root, require_non_notes=not a.allow_no_objects)
    print(f"dataset audit: {audit.stats['total_images']} images, {len(audit.errors)} errors, {len(audit.warnings)} warnings")
    gate_errors = list(audit.errors) if a.allow_small else audit.readiness_blockers
    if gate_errors:
        print("Dataset is not ready for training:", file=sys.stderr)
        for b in gate_errors[:20]:
            print("  -", b, file=sys.stderr)
        print("See docs/DATASET_COLLECTION.md; run scripts/validate_dataset.py for the full report.", file=sys.stderr)
        if not a.allow_small or audit.errors:
            return 1
    elif a.allow_small and audit.readiness_blockers:
        print("WARNING: --allow-small: coverage blockers ignored; this run is NOT a deliverable:", file=sys.stderr)
        for b in audit.readiness_blockers:
            print("  -", b, file=sys.stderr)

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"error: {exc}. Install training deps: pip install -r requirements-train.txt", file=sys.stderr)
        return 3

    random.seed(seed)
    torch.manual_seed(seed)
    paths = DatasetPaths(dataset_root)
    data_yaml = write_data_yaml(paths, run_dir, a.final_fit, a.rot90, seed)

    model_src = a.init or cfg.get("model", "yolov8n.pt")
    last = run_dir / "weights" / "last.pt"
    if a.resume:
        if not last.is_file():
            print(f"error: cannot resume, {last} not found", file=sys.stderr)
            return 1
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        if str(model_src).endswith(".pt") and not Path(model_src).is_file():
            print(f"note: {model_src} will be downloaded by Ultralytics (COCO weights; internet required once)")
        model = YOLO(str(model_src))
        model.train(data=str(data_yaml), project=str(run_root), name=name, exist_ok=True, seed=seed,
                    single_cls=False,  # a checkpoint's old train_args (single_cls=True) would otherwise be inherited
                    verbose=True, **tr)
        got_nc = int(getattr(getattr(model, "model", None), "nc", 0) or 0)
        if got_nc != 2:
            print(f"error: trained model has {got_nc} class(es), expected 2 (Rs.50 + other note); not usable", file=sys.stderr)
            return 1

    best = run_dir / "weights" / "best.pt"
    info = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deliverable": not a.allow_small, "dataset_ready": audit.ready_for_training,
        "final_fit_on_all_images": a.final_fit, "rot90_copies": a.rot90, "allowed_no_object_negatives": a.allow_no_objects,
        "dataset_manifest_hash": manifest_hash(load_manifest(paths)), "dataset_stats": audit.stats,
        "effective_config": {"model": str(model_src), "seed": seed, **tr},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "best_checkpoint": str(best) if best.is_file() else None,
        "best_sha256": sha256_file(best) if best.is_file() else None,
        "note": "Best checkpoint is selected by Ultralytics on the VALIDATION split; the test split is untouched.",
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"training finished. best checkpoint: {best}\nrun info: {run_dir / 'run_info.json'}")
    print(f"next: python scripts/export_onnx.py --weights {best} --out model.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
