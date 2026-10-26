"""Audit the dataset and write a human-readable report (reports/dataset_audit.md + .json).

    python scripts/validate_dataset.py
Exit code 0 only if there are no errors AND the dataset meets the minimum coverage for training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from detector.dataset import NEAR_DUP_DISTANCE, audit_dataset, render_markdown


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="reports")
    p.add_argument("--near-dup-distance", type=int, default=NEAR_DUP_DISTANCE,
                   help="max dHash Hamming distance (of 256 bits) to call two images near-duplicates")
    p.add_argument("--allow-no-objects", action="store_true",
                   help="do not block on missing object/background negatives (reported as a warning)")
    a = p.parse_args()
    res = audit_dataset(a.dataset, near_dup_distance=a.near_dup_distance, require_non_notes=not a.allow_no_objects)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.md").write_text(render_markdown(res, a.dataset), encoding="utf-8")
    (out / "dataset_audit.json").write_text(res.to_json(), encoding="utf-8")
    print(f"images: {res.stats['total_images']} | errors: {len(res.errors)} | warnings: {len(res.warnings)}")
    for e in res.errors[:15]:
        print("  ERROR:", e)
    if res.readiness_blockers:
        print("NOT READY for training:")
        for b in res.readiness_blockers:
            print("  -", b)
    else:
        print("READY for training.")
    print(f"report: {out / 'dataset_audit.md'}")
    return 0 if res.ready_for_training else 1


if __name__ == "__main__":
    sys.exit(main())
