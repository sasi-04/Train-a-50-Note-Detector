"""Dataset auditor tests on TINY SYNTHETIC images (tooling logic only, never used for training)."""

import cv2
import numpy as np
import pytest

from detector.dataset import (DatasetPaths, audit_dataset, load_manifest, manifest_hash, save_manifest,
                              sha256_file, write_label_file)

_counter = [0]


def add(paths, rows, session, den, side="front", split="", box=True, seed=None, label_text=None):
    _counter[0] += 1
    n = _counter[0]
    rng = np.random.default_rng(seed if seed is not None else n)
    img = cv2.resize(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST)
    iid = f"{session}_{den}_{n:04d}"
    fn = iid + ".jpg"
    paths.ensure()
    cv2.imwrite(str(paths.images / fn), img)
    lab = paths.labels / (iid + ".txt")
    if label_text is not None:
        lab.write_text(label_text)
    else:
        write_label_file(lab, [(0, 0.5, 0.5, 0.5, 0.5)] if (box and den in ("50_new", "mixed")) else [])
    rows.append({"image_id": iid, "filename": fn, "denomination": den, "side": side, "lighting": "normal",
                 "angle": "frontal", "background": "plain", "source": "phone", "session": session, "split": split,
                 "label_status": "annotated", "notes": "", "sha256": sha256_file(paths.images / fn)})
    return rows[-1]


def build_complete(tmp_path):
    paths = DatasetPaths(tmp_path / "ds")
    rows = []
    plan = {"train": ["s1", "s2"], "val": ["s3"], "test": ["s4"]}
    for split, sessions in plan.items():
        for s in sessions:
            add(paths, rows, s, "50_new", "front", split)
            add(paths, rows, s, "50_new", "back", split)
            for d in ("10", "20", "100", "200", "500", "object"):
                add(paths, rows, s, d, "na", split)
    save_manifest(paths, rows)
    return paths, rows


def test_complete_dataset_is_ready(tmp_path):
    paths, _ = build_complete(tmp_path)
    res = audit_dataset(paths.root)
    assert res.errors == [] and res.readiness_blockers == [] and res.ready_for_training
    assert res.stats["target_by_side"]["front"] == 4 and res.stats["target_by_side"]["back"] == 4
    assert res.warnings  # below recommended minimums must be flagged, not hidden


def test_missing_object_negatives_blocks_unless_explicitly_allowed(tmp_path):
    paths, rows = build_complete(tmp_path)
    keep = [r for r in rows if r["denomination"] != "object"]
    for r in rows:
        if r["denomination"] == "object":
            (paths.images / r["filename"]).unlink()
            (paths.labels / (r["image_id"] + ".txt")).unlink()
    save_manifest(paths, keep)
    assert any("object" in b for b in audit_dataset(paths.root).readiness_blockers)
    res = audit_dataset(paths.root, require_non_notes=False)
    assert res.ready_for_training and any("NO ordinary-object" in w for w in res.warnings)


def test_empty_dataset_not_ready(tmp_path):
    res = audit_dataset(tmp_path / "nothing")
    assert not res.ready_for_training and "no images in the dataset" in res.readiness_blockers


def test_missing_back_side_and_negatives_block(tmp_path):
    paths = DatasetPaths(tmp_path / "ds")
    rows = []
    for s, sp in (("s1", "train"), ("s2", "val"), ("s3", "test")):
        add(paths, rows, s, "50_new", "front", sp)
    save_manifest(paths, rows)
    res = audit_dataset(paths.root)
    text = " ".join(res.readiness_blockers)
    assert "back-side" in text and "Rs.10 " in text and "Rs.500" in text and "object" in text


def test_missing_label_and_bad_label_are_errors(tmp_path):
    paths, rows = build_complete(tmp_path)
    (paths.labels / (rows[0]["image_id"] + ".txt")).unlink()
    write_label_file(paths.labels / (rows[2]["image_id"] + ".txt"), [])
    (paths.labels / (rows[4]["image_id"] + ".txt")).write_text("0 2 2 2 2\n")
    errs = " ".join(audit_dataset(paths.root).errors)
    assert "missing label file" in errs and "outside [0,1]" in errs


def test_target_with_empty_label_and_negative_with_box(tmp_path):
    paths, rows = build_complete(tmp_path)
    target = next(r for r in rows if r["denomination"] == "50_new")
    neg = next(r for r in rows if r["denomination"] == "20")
    write_label_file(paths.labels / (target["image_id"] + ".txt"), [])
    write_label_file(paths.labels / (neg["image_id"] + ".txt"), [(0, 0.5, 0.5, 0.5, 0.5)])
    errs = " ".join(audit_dataset(paths.root).errors)
    assert "label is empty" in errs and "negative but label contains target boxes" in errs


def test_corrupted_and_blank_images(tmp_path):
    paths, rows = build_complete(tmp_path)
    (paths.images / rows[0]["filename"]).write_bytes(b"garbage")
    cv2.imwrite(str(paths.images / rows[1]["filename"]), np.full((50, 50, 3), 100, np.uint8))
    errs = " ".join(audit_dataset(paths.root).errors)
    assert "corrupted" in errs and "blank" in errs


def test_exact_duplicate_detected(tmp_path):
    paths, rows = build_complete(tmp_path)
    import shutil
    shutil.copy(paths.images / rows[0]["filename"], paths.images / "dup.jpg")
    shutil.copy(paths.labels / (rows[0]["image_id"] + ".txt"), paths.labels / "dup.txt")
    r = dict(rows[0], image_id="dup", filename="dup.jpg")
    save_manifest(paths, rows + [r])
    assert any("exact duplicate" in e for e in audit_dataset(paths.root).errors)


def test_cross_split_near_duplicate_flagged_as_leakage(tmp_path):
    paths, rows = build_complete(tmp_path)
    src = next(r for r in rows if r["split"] == "train")
    img = cv2.imread(str(paths.images / src["filename"]))
    cv2.imwrite(str(paths.images / "leak.jpg"), np.clip(img.astype(int) + 1, 0, 255).astype(np.uint8))
    write_label_file(paths.labels / "leak.txt", [(0, 0.5, 0.5, 0.5, 0.5)] if src["denomination"] == "50_new" else [])
    leak = dict(src, image_id="leak", filename="leak.jpg", split="test", session="s4", sha256="")
    save_manifest(paths, rows + [leak])
    assert any("possible leakage" in e for e in audit_dataset(paths.root).errors)


def test_session_spanning_splits_is_error(tmp_path):
    paths, rows = build_complete(tmp_path)
    rows[0]["split"] = "test"
    save_manifest(paths, rows)
    assert any("spans multiple splits" in e for e in audit_dataset(paths.root).errors)


def test_invalid_vocab_and_missing_session(tmp_path):
    paths, rows = build_complete(tmp_path)
    rows[0]["lighting"] = "disco"
    rows[1]["session"] = ""
    save_manifest(paths, rows)
    errs = " ".join(audit_dataset(paths.root).errors)
    assert "invalid lighting" in errs and "missing session" in errs


def test_denomination_is_never_inferred_from_filename(tmp_path):
    paths, rows = build_complete(tmp_path)
    r = next(r for r in rows if r["denomination"] == "20")
    r["denomination"] = "50_new"  # filename still says '20'; manifest + label disagree -> must be flagged
    save_manifest(paths, rows)
    assert any("label is empty" in e for e in audit_dataset(paths.root).errors)


def test_manifest_hash_changes_with_metadata(tmp_path):
    paths, rows = build_complete(tmp_path)
    h1 = manifest_hash(rows)
    rows[0]["lighting"] = "dim"
    assert manifest_hash(rows) != h1


def test_manifest_roundtrip(tmp_path):
    paths, rows = build_complete(tmp_path)
    assert load_manifest(paths) == sorted(rows, key=lambda r: r["image_id"])
