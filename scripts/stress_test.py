"""False-positive stress test: run the deployed detector over NEGATIVE images only.

    python scripts/stress_test.py --model model.onnx --split val

Reports, per negative category, how many images were tested and how many produced an accepted
Rs.50 at each threshold; the confidence distribution of the strongest false ₹50 score per image; the
hardest negatives (saved as annotated JPEGs for manual review); and which categories are confused most.
Use --split test only for the final report; thresholds must already be frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from detector.config import load_config
from detector.dataset import DatasetPaths, load_manifest
from detector.inference import OnnxDetector
from detector.metrics import collect_predictions, wilson_interval

THRESHOLDS = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.onnx")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    ap.add_argument("--threshold", type=float, default=None, help="operating threshold to highlight")
    ap.add_argument("--top", type=int, default=20, help="hardest negatives to save")
    ap.add_argument("--config", default=None, help="inference config (default configs/inference.yaml)")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    paths = DatasetPaths(Path(a.dataset))
    rows = [r for r in load_manifest(paths) if a.split == "all" or r["split"] == a.split]
    rows = [r for r in rows if r["denomination"] not in ("50_new", "mixed") and r["label_status"] != "unlabeled"]
    if not rows:
        print(f"error: no negative images in split '{a.split}'", file=sys.stderr)
        return 1
    cfg = load_config(a.config)
    op = a.threshold if a.threshold is not None else cfg.threshold
    det = OnnxDetector(a.model, score_floor=0.01, nms_iou=cfg.nms_iou)
    recs = collect_predictions(det, rows, paths)

    hardest = sorted(((max(r.target_scores(), default=0.0), r) for r in recs), key=lambda t: -t[0])
    cats = sorted({r.meta["denomination"] for r in recs})
    table = {}
    for c in cats:
        sub = [(s, r) for s, r in hardest if r.meta["denomination"] == c]
        n = len(sub)
        per_t = {f"{t:.2f}": sum(s >= t for s, _ in sub) for t in sorted(set(THRESHOLDS + [op]))}
        k = per_t[f"{op:.2f}"]
        table[c] = {"tested": n, "false_positive_images_at_threshold": k,
                    "fp_rate_ci95": list(wilson_interval(k, n)), "fp_images_by_threshold": per_t,
                    "max_false_score": max((s for s, _ in sub), default=0.0)}
    scores = np.array([s for s, _ in hardest])
    out = Path(a.out)
    (out / "stress").mkdir(parents=True, exist_ok=True)
    for rank, (s, r) in enumerate(hardest[: a.top], 1):
        img = cv2.imread(str(paths.images / r.meta["filename"]))
        for box, sc, cl in r.preds:
            if cl == 0:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img, f"{sc:.2f}", (int(box[0]), max(15, int(box[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(out / "stress" / f"{rank:02d}_{r.meta['denomination']}_{s:.2f}_{r.image_id}.jpg"), img)

    result = {"split": a.split, "operating_threshold": op, "negative_images": len(recs), "by_category": table,
              "false_score_quantiles": {q: float(np.quantile(scores, q / 100)) for q in (50, 90, 95, 99, 100)},
              "hardest": [{"image_id": r.image_id, "denomination": r.meta["denomination"], "max_score": s,
                           "accepted_at_threshold": s >= op, "needs_manual_review": s >= op * 0.8} for s, r in hardest[: a.top]]}
    (out / f"stress_{a.split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [f"# False-positive stress test ({a.split} split, negatives only)", "",
          f"Operating threshold: **{op:.2f}**. Negative images tested: **{len(recs)}**.", "",
          "| category | tested | false ₹50 at threshold | 95% CI | max false score | " + " | ".join(f">={t:.2f}" for t in THRESHOLDS) + " |",
          "|---|---|---|---|---|" + "---|" * len(THRESHOLDS)]
    for c, t in table.items():
        lo, hi = t["fp_rate_ci95"]
        md.append(f"| {c} | {t['tested']} | {t['false_positive_images_at_threshold']} | {lo:.0%}-{hi:.0%} | {t['max_false_score']:.2f} | "
                  + " | ".join(str(t["fp_images_by_threshold"].get(f"{x:.2f}", "")) for x in THRESHOLDS) + " |")
    md += ["", "Strongest false-₹50 score per image, quantiles: " + ", ".join(f"p{q}={v:.2f}" for q, v in result["false_score_quantiles"].items()),
           "", f"Hardest negatives (annotated JPEGs in `{out / 'stress'}`):", ""]
    md += [f"- {h['image_id']} ({h['denomination']}): {h['max_score']:.2f}" + (" **ACCEPTED - review**" if h["accepted_at_threshold"] else
           (" near-miss - review" if h["needs_manual_review"] else "")) for h in result["hardest"]]
    md += ["", "Small categories give wide confidence intervals; a 0% observed rate is not a guarantee.", ""]
    (out / f"stress_{a.split}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
