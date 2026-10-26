"""Assign whole capture SESSIONS to train/val/test (never individual frames) and record it in the manifest.

    python scripts/split_dataset.py --seed 42 --val 0.15 --test 0.15

Near-identical frames from one session therefore cannot land in different splits. The val split is
used for threshold selection; the test split is for the final held-out report only.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict

import _bootstrap  # noqa: F401

from detector.dataset import DatasetPaths, load_manifest, save_manifest


def assign(rows: list[dict], val: float, test: float, seed: int) -> dict[str, str]:
    """Greedy, seeded session assignment minimising the gap to the requested image fractions.

    A session is placed in the split with the largest remaining deficit; target-bearing
    sessions are handled first so val/test each receive some target images when possible.
    """
    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sessions[r["session"]].append(r)
    rng = random.Random(seed)
    names = sorted(sessions)
    rng.shuffle(names)

    def n_target(s: str) -> int:
        return sum(r["denomination"] in ("50_new", "mixed") for r in sessions[s])

    names.sort(key=lambda s: (-(n_target(s) > 0), -len(sessions[s])))
    total = len(rows)
    goal = {"train": (1 - val - test) * total, "val": val * total, "test": test * total}
    have = Counter()
    target_have = Counter()
    out: dict[str, str] = {}
    for s in names:
        size, tg = len(sessions[s]), n_target(s)

        def deficit(sp: str) -> float:
            d = goal[sp] - have[sp]
            if tg and target_have[sp] == 0 and sp != "train":
                d += total  # make sure val/test each get a target-bearing session first
            return d

        sp = max(("val", "test", "train"), key=deficit)
        out[s] = sp
        have[sp] += size
        target_have[sp] += tg
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not (0 < a.val < 0.5 and 0 < a.test < 0.5 and a.val + a.test < 0.6):
        print("error: --val/--test must be in (0, 0.5) and sum to < 0.6", file=sys.stderr)
        return 2
    paths = DatasetPaths(a.dataset)
    rows = load_manifest(paths)
    if not rows:
        print("error: manifest is empty; ingest images first", file=sys.stderr)
        return 1
    if any(not r["session"] for r in rows):
        print("error: some images have no session id", file=sys.stderr)
        return 1
    mapping = assign(rows, a.val, a.test, a.seed)
    if len(mapping) < 3:
        print(f"warning: only {len(mapping)} session(s); at least 3 are needed for train/val/test", file=sys.stderr)
    for r in rows:
        r["split"] = mapping[r["session"]]
    save_manifest(paths, rows)
    counts = Counter(r["split"] for r in rows)
    print("split (images):", dict(counts), "| sessions:", Counter(mapping.values()))
    for sp in ("train", "val", "test"):
        print(f"  {sp}: {dict(Counter(r['denomination'] for r in rows if r['split'] == sp))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
