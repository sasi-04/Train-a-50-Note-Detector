"""Dataset manifest, YOLO label validation and the dataset-quality auditor."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POOL = "pool"

DENOMINATIONS = ["50_new", "50_old", "10", "20", "100", "200", "500", "object", "background", "mixed", "unknown"]
REQUIRED_NEGATIVE_DENOMS = ["10", "20", "100", "200", "500"]
NON_NOTE = {"object", "background"}
SIDES = ["front", "back", "na", "unknown"]
LIGHTING = ["bright", "normal", "dim", "uneven", "shadow", "unknown"]
ANGLES = ["frontal", "tilted", "steep", "unknown"]
BACKGROUNDS = ["plain", "cluttered", "hand_held", "unknown"]
SOURCES = ["phone", "webcam", "unknown"]
LABEL_STATUSES = ["unlabeled", "annotated", "negative_confirmed"]
SPLITS = ["train", "val", "test", ""]

FIELDS = ["image_id", "filename", "denomination", "side", "lighting", "angle", "background",
          "source", "session", "split", "label_status", "notes", "sha256"]

# Recommended minimums (guidelines, reported as warnings; not statistical guarantees).
RECOMMENDED = {"target_per_side": 100, "negative_per_denom": 30, "non_note": 50, "sessions": 6}

VOCAB = {"denomination": DENOMINATIONS, "side": SIDES, "lighting": LIGHTING, "angle": ANGLES,
         "background": BACKGROUNDS, "source": SOURCES, "label_status": LABEL_STATUSES}


# ---------------------------------------------------------------------------------------------
# paths + manifest I/O
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def images(self) -> Path:
        return self.root / "images" / POOL

    @property
    def labels(self) -> Path:
        return self.root / "labels" / POOL

    @property
    def manifest(self) -> Path:
        return self.root / "metadata" / "image_manifest.csv"

    def label_for(self, image_path: Path) -> Path:
        return self.labels / (image_path.stem + ".txt")

    def ensure(self) -> None:
        for d in (self.images, self.labels, self.manifest.parent):
            d.mkdir(parents=True, exist_ok=True)


def load_manifest(paths: DatasetPaths) -> list[dict[str, str]]:
    if not paths.manifest.is_file():
        return []
    with paths.manifest.open(newline="", encoding="utf-8") as fh:
        return [{k: (row.get(k) or "") for k in FIELDS} for row in csv.DictReader(fh)]


def save_manifest(paths: DatasetPaths, rows: list[dict[str, str]]) -> None:
    """Atomic write so an interrupted run cannot corrupt the manifest."""
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=paths.manifest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["image_id"]))
        os.replace(tmp, paths.manifest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(rows: list[dict[str, str]]) -> str:
    """Stable hash of the dataset (image hashes + metadata) for reproducibility reports."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["image_id"]):
        h.update("|".join(r[k] for k in FIELDS).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# YOLO label validation
# ---------------------------------------------------------------------------------------------
@dataclass
class LabelResult:
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def parse_label_file(path: Path, num_classes: int = 1) -> LabelResult:
    """Parse and validate one YOLO label file. An empty file is a valid negative."""
    res = LabelResult()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        res.issues.append(f"unreadable label file ({exc.__class__.__name__})")
        return res
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            res.issues.append(f"line {n}: expected 5 values (class cx cy w h), got {len(parts)}")
            continue
        try:
            cls_f = float(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            res.issues.append(f"line {n}: non-numeric value")
            continue
        if cls_f != int(cls_f) or not (0 <= int(cls_f) < num_classes):
            res.issues.append(f"line {n}: invalid class id {parts[0]} (valid: 0..{num_classes - 1})")
            continue
        vals = (cx, cy, w, h)
        if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
            res.issues.append(f"line {n}: NaN/Inf coordinate")
            continue
        if not all(0.0 <= v <= 1.0 for v in vals):
            res.issues.append(f"line {n}: coordinate outside [0,1]")
            continue
        if w <= 0 or h <= 0:
            res.issues.append(f"line {n}: non-positive width/height (zero-area box)")
            continue
        if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            res.issues.append(f"line {n}: box extends beyond the image")
            continue
        res.boxes.append((int(cls_f), cx, cy, w, h))
    return res


def write_label_file(path: Path, boxes: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# image fingerprints
# ---------------------------------------------------------------------------------------------
def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------------------------
@dataclass
class AuditResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    ready_for_training: bool = False
    readiness_blockers: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


def audit_dataset(root: Path | str, num_classes: int = 1, near_dup_distance: int = 5) -> AuditResult:
    """Inspect actual files on disk and the manifest; never trusts filenames for denomination."""
    paths = DatasetPaths(Path(root))
    res = AuditResult()
    err, warn = res.errors, res.warnings
    rows = load_manifest(paths)
    by_file = {r["filename"]: r for r in rows}

    disk_images = sorted(p for p in paths.images.glob("*") if p.suffix.lower() in IMAGE_EXTS) if paths.images.is_dir() else []
    other_files = [p.name for p in paths.images.glob("*")] if paths.images.is_dir() else []
    for n in other_files:
        if Path(n).suffix.lower() not in IMAGE_EXTS:
            warn.append(f"unsupported file in images/: {n}")
    disk_names = {p.name for p in disk_images}

    for r in rows:
        if r["filename"] not in disk_names:
            err.append(f"manifest row {r['image_id']}: image file {r['filename']} missing on disk")
    for p in disk_images:
        if p.name not in by_file:
            err.append(f"{p.name}: image has no manifest row (denomination/session unknown)")

    seen_hash: dict[str, str] = {}
    hashes: dict[str, int] = {}
    sizes: Counter = Counter()
    blurry = []
    usable: list[Path] = []
    box_counts: Counter = Counter()
    unlabeled = 0

    for r in rows:
        for key, allowed in VOCAB.items():
            if r[key] and r[key] not in allowed:
                err.append(f"{r['image_id']}: invalid {key} '{r[key]}' (allowed: {', '.join(allowed)})")
            elif not r[key]:
                warn.append(f"{r['image_id']}: missing metadata field '{key}' (treated as unknown)")
        if not r["session"]:
            err.append(f"{r['image_id']}: missing session (needed for leak-free splitting)")
        if r["split"] not in SPLITS:
            err.append(f"{r['image_id']}: invalid split '{r['split']}'")

    for p in disk_images:
        row = by_file.get(p.name)
        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            err.append(f"{p.name}: corrupted or unreadable image")
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 2.0:
            err.append(f"{p.name}: image is blank/uniform")
            continue
        sizes[_res_bucket(max(w, h))] += 1
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 20.0:
            blurry.append(p.name)
        digest = sha256_file(p)
        if digest in seen_hash:
            err.append(f"{p.name}: exact duplicate of {seen_hash[digest]}")
        else:
            seen_hash[digest] = p.name
        if row and row["sha256"] and row["sha256"] != digest:
            warn.append(f"{p.name}: file changed since ingest (sha256 differs from manifest)")
        hashes[p.name] = dhash(gray)
        usable.append(p)

        lab = paths.label_for(p)
        if not lab.is_file():
            err.append(f"{p.name}: missing label file {lab.name} (use an empty file for a negative)")
            unlabeled += 1
            continue
        lr = parse_label_file(lab, num_classes)
        for issue in lr.issues:
            err.append(f"{lab.name}: {issue}")
        box_counts[len(lr.boxes)] += 1
        if row:
            den = row["denomination"]
            if den in ("50_new", "mixed") and not lr.boxes and not lr.issues:
                err.append(f"{p.name}: denomination '{den}' but label is empty (missing annotation)")
            if den not in ("50_new", "mixed", "unknown", "") and lr.boxes:
                err.append(f"{p.name}: denomination '{den}' is a negative but label contains target boxes")
            if row["label_status"] == "unlabeled" and den in ("50_new", "mixed"):
                warn.append(f"{p.name}: label_status 'unlabeled' (annotation not reviewed)")
    for lab in (paths.labels.glob("*.txt") if paths.labels.is_dir() else []):
        if not any((paths.images / (lab.stem + e)).exists() for e in IMAGE_EXTS):
            warn.append(f"{lab.name}: orphan label (no image)")

    # near-duplicates + leakage -------------------------------------------------------------
    names = [p.name for p in usable]
    hv = [hashes[n] for n in names]
    leaks, same_split_dups = [], 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if hamming(hv[i], hv[j]) <= near_dup_distance:
                ra, rb = by_file.get(names[i]), by_file.get(names[j])
                if ra and rb and ra["split"] and rb["split"] and ra["split"] != rb["split"]:
                    leaks.append((names[i], names[j], ra["split"], rb["split"]))
                else:
                    same_split_dups += 1
    for a, b, sa, sb in leaks[:50]:
        err.append(f"possible leakage: {a} ({sa}) is a near-duplicate of {b} ({sb})")
    if len(leaks) > 50:
        err.append(f"... and {len(leaks) - 50} more possible leaks")
    if same_split_dups:
        warn.append(f"{same_split_dups} near-duplicate pairs within the same split/session "
                    "(expected for burst captures; consider thinning them)")

    sess_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["split"]:
            sess_splits[r["session"]].add(r["split"])
    for s, sp in sess_splits.items():
        if len(sp) > 1:
            err.append(f"session '{s}' spans multiple splits {sorted(sp)}; re-run split_dataset.py")

    # coverage ------------------------------------------------------------------------------
    den = Counter(r["denomination"] or "unknown" for r in rows)
    side = Counter((r["denomination"], r["side"]) for r in rows if r["denomination"] in ("50_new", "mixed"))
    split_counts = Counter(r["split"] or "unassigned" for r in rows)
    split_den: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        split_den[r["split"] or "unassigned"][r["denomination"]] += 1
    sessions = Counter(r["session"] for r in rows)
    source = Counter(r["source"] or "unknown" for r in rows)
    lighting = Counter(r["lighting"] or "unknown" for r in rows)

    blockers: list[str] = []
    if not rows:
        blockers.append("no images in the dataset")
    front = side[("50_new", "front")] + side[("mixed", "front")]
    back = side[("50_new", "back")] + side[("mixed", "back")]
    if front == 0:
        blockers.append("no front-side images of the new Rs.50 note")
    if back == 0:
        blockers.append("no back-side images of the new Rs.50 note")
    for d in REQUIRED_NEGATIVE_DENOMS:
        if den[d] == 0:
            blockers.append(f"no negative images of the Rs.{d} note")
    if den["object"] + den["background"] == 0:
        blockers.append("no ordinary-object / empty-background negatives")
    if len([s for s in sessions if s]) < 3:
        blockers.append("fewer than 3 capture sessions (cannot form leak-free train/val/test)")
    if split_counts.get("unassigned", 0) and len(rows) > 0:
        blockers.append("some images have no split; run scripts/split_dataset.py")
    for sp in ("train", "val", "test"):
        if rows and not split_counts.get(sp):
            blockers.append(f"split '{sp}' is empty")
        elif rows and not (split_den[sp]["50_new"] + split_den[sp]["mixed"]):
            blockers.append(f"split '{sp}' contains no target images")
    if err:
        blockers.append(f"{len(err)} validation error(s)")

    R = RECOMMENDED
    if rows:
        if front < R["target_per_side"]:
            warn.append(f"only {front} front-side target images (recommended >= {R['target_per_side']})")
        if back < R["target_per_side"]:
            warn.append(f"only {back} back-side target images (recommended >= {R['target_per_side']})")
        for d in REQUIRED_NEGATIVE_DENOMS:
            if 0 < den[d] < R["negative_per_denom"]:
                warn.append(f"only {den[d]} images of Rs.{d} (recommended >= {R['negative_per_denom']}); "
                            "per-denomination error rates will be unreliable")
        if den["object"] + den["background"] < R["non_note"]:
            warn.append(f"only {den['object'] + den['background']} object/background negatives "
                        f"(recommended >= {R['non_note']})")
        if den["50_old"] == 0:
            warn.append("no images of the OLD Rs.50 note (not the target; recommended as a hard negative)")
        if len(sessions) < R["sessions"]:
            warn.append(f"only {len(sessions)} capture sessions (recommended >= {R['sessions']})")
        if source.get("webcam", 0) and source.get("phone", 0) == 0:
            warn.append("all images are webcam captures; the assignment requires phone photos for training")

    res.stats = {
        "total_images": len(rows),
        "images_on_disk": len(disk_images),
        "manifest_hash": manifest_hash(rows) if rows else None,
        "by_denomination": dict(den),
        "target_by_side": {"front": front, "back": back,
                           "unknown": sum(v for (d, s), v in side.items() if s not in ("front", "back"))},
        "by_split": dict(split_counts),
        "by_split_and_denomination": {k: dict(v) for k, v in split_den.items()},
        "sessions": dict(sessions),
        "by_source": dict(source),
        "by_lighting": dict(lighting),
        "resolution_buckets": dict(sizes),
        "boxes_per_image": {str(k): v for k, v in sorted(box_counts.items())},
        "label_files_missing": unlabeled,
        "blurry_images_laplacian_var_lt_20": len(blurry),
        "near_duplicate_pairs_same_split": same_split_dups,
        "cross_split_near_duplicates": len(leaks),
    }
    res.readiness_blockers = blockers
    res.ready_for_training = not blockers
    return res


def _res_bucket(long_side: int) -> str:
    for lim in (640, 1024, 1600, 2560):
        if long_side <= lim:
            return f"<= {lim}"
    return "> 2560"


def render_markdown(res: AuditResult, root: Path | str) -> str:
    s = res.stats
    lines = [
        "# Dataset quality audit",
        "",
        f"Dataset root: `{root}`  ",
        f"Manifest hash: `{s.get('manifest_hash')}`  ",
        f"Status: **{'READY FOR TRAINING' if res.ready_for_training else 'NOT READY'}** "
        f"({len(res.errors)} errors, {len(res.warnings)} warnings)",
        "",
    ]
    if res.readiness_blockers:
        lines += ["## Blockers", ""] + [f"- {b}" for b in res.readiness_blockers] + [""]
    lines += [f"## Totals", "", f"- Images in manifest: {s['total_images']} (on disk: {s['images_on_disk']})",
              f"- Target front / back / unknown side: {s['target_by_side']['front']} / "
              f"{s['target_by_side']['back']} / {s['target_by_side']['unknown']}", ""]

    def table(title: str, d: dict) -> list[str]:
        out = [f"## {title}", "", "| value | count |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))] or ["| (none) | 0 |"]
        return out + [""]

    lines += table("Images per denomination", s["by_denomination"])
    lines += table("Images per split", s["by_split"])
    lines += ["## Split x denomination", "", "| split | " + " | ".join(DENOMINATIONS) + " |",
              "|---|" + "---|" * len(DENOMINATIONS)]
    for sp, d in sorted(s["by_split_and_denomination"].items()):
        lines.append(f"| {sp} | " + " | ".join(str(d.get(x, 0)) for x in DENOMINATIONS) + " |")
    lines.append("")
    lines += table("Capture sessions", s["sessions"])
    lines += table("Image source", s["by_source"])
    lines += table("Lighting condition", s["by_lighting"])
    lines += table("Resolution (long side, px)", s["resolution_buckets"])
    lines += table("Boxes per image", s["boxes_per_image"])
    lines += ["## Integrity", "",
              f"- Near-duplicate pairs inside one split: {s['near_duplicate_pairs_same_split']}",
              f"- Cross-split near-duplicate pairs (leakage): {s['cross_split_near_duplicates']}",
              f"- Very blurry images (Laplacian variance < 20): {s['blurry_images_laplacian_var_lt_20']}", ""]
    cap = 60
    lines += [f"## Errors ({len(res.errors)})", ""] + [f"- {e}" for e in res.errors[:cap]]
    if len(res.errors) > cap:
        lines.append(f"- ... {len(res.errors) - cap} more (see JSON report)")
    lines += ["", f"## Warnings ({len(res.warnings)})", ""] + [f"- {w}" for w in res.warnings[:cap]]
    if len(res.warnings) > cap:
        lines.append(f"- ... {len(res.warnings) - cap} more (see JSON report)")
    lines += ["", "_Generated from the files on disk. Recommended minimums are guidelines, not statistical guarantees._", ""]
    return "\n".join(lines)
