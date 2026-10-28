"""Evaluate the deployed ONNX model (via the same OpenCV pipeline as test.py) on a dataset split.

  # 1) choose the operating threshold on VALIDATION data only:
  python scripts/evaluate.py --model model.onnx --split val --select-threshold --write-config

  # 2) final held-out report with the frozen threshold (read from configs/inference.yaml):
  python scripts/evaluate.py --model model.onnx --split test

The test split can never be used to select a threshold (the flag is refused). Everything reported is
computed from files on disk; groups with no images are simply absent (NOT TESTED).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import yaml

from detector.config import DEFAULT_CONFIG_PATH, load_config
from detector.dataset import DatasetPaths, load_manifest, manifest_hash, sha256_file
from detector.inference import OnnxDetector
from detector.metrics import (DEFAULT_THRESHOLDS, average_precision, collect_predictions, group_stats,
                              latency_summary, note_type_table, overall, sweep)
from detector.thresholding import select_threshold

REQUIRED_SCENARIOS = {
    "front side": lambda r: r.meta["denomination"] == "50_new" and r.meta["side"] == "front",
    "back side": lambda r: r.meta["denomination"] == "50_new" and r.meta["side"] == "back",
    "dim lighting": lambda r: r.meta["denomination"] == "50_new" and r.meta["lighting"] in ("dim", "uneven", "shadow"),
    "angled": lambda r: r.meta["denomination"] == "50_new" and r.meta["angle"] in ("tilted", "steep"),
}


def hardware_info() -> dict:
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:  # noqa: BLE001 - non-Windows or restricted registry
        pass
    return {"cpu": cpu, "logical_cores": __import__("os").cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}


def plot_curves(pts, out_dir: Path, split: str, chosen: float | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    th = [p.threshold for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([p.recall for p in pts], [p.precision for p in pts])
    ax[0].set(xlabel="recall (detection level)", ylabel="precision", title=f"PR curve ({split})", xlim=(0, 1.02), ylim=(0, 1.02))
    ax[1].plot(th, [p.recall for p in pts], label="recall")
    ax[1].plot(th, [p.neg_fp_rate for p in pts], label="negative-image FP rate")
    ax[1].plot(th, [p.precision for p in pts], label="precision", alpha=0.6)
    if chosen is not None:
        ax[1].axvline(chosen, color="k", ls="--", label=f"threshold {chosen:.2f}")
    ax[1].set(xlabel="score threshold", title=f"threshold trade-off ({split})", ylim=(0, 1.02))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"threshold_analysis_{split}.png", dpi=110)
    plt.close(fig)


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.1%}" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def group_table(title: str, groups: dict) -> list[str]:
    out = [f"### {title}", "", "| group | images | pos | neg | TP img | FN img | FP img | image recall | neg-image FP rate (95% CI upper) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, g in groups.items():
        fp = f"{g['fp_images']}/{g['negative_images']}" + (f" ({fmt(g['neg_image_fp_rate'], True)}, <= {fmt(g['neg_image_fp_rate_ci95_upper'], True)})" if g["negative_images"] else "")
        out.append(f"| {k or 'unknown'} | {g['images']} | {g['positive_images']} | {g['negative_images']} | {g['tp_images']} | "
                   f"{g['fn_images']} | {g['fp_images']} | {fmt(g['image_recall'], True)} | {fp if g['negative_images'] else 'n/a'} |")
    return out + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.onnx")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--threshold", type=float, default=None, help="default: configs/inference.yaml")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--select-threshold", action="store_true", help="choose threshold on this split (val only)")
    ap.add_argument("--max-neg-fp-rate", type=float, default=0.0, help="allowed negative-image FP rate for rule 1")
    ap.add_argument("--min-recall", type=float, default=0.80, help="minimum detection recall for rule 1")
    ap.add_argument("--write-config", action="store_true", help="write the selected threshold to configs/inference.yaml")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="inference config to read (and write with --write-config)")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    if a.select_threshold and a.split != "val":
        print("error: thresholds may only be selected on the validation split, never on test/train", file=sys.stderr)
        return 2
    if a.write_config and not a.select_threshold:
        print("error: --write-config requires --select-threshold", file=sys.stderr)
        return 2

    paths = DatasetPaths(Path(a.dataset))
    all_rows = load_manifest(paths)
    rows = [r for r in all_rows if r["split"] == a.split and r["label_status"] != "unlabeled"]
    if not rows:
        print(f"error: no images in split '{a.split}'. Collect data, then run split_dataset.py.", file=sys.stderr)
        return 1
    cfg = load_config(a.config)
    detector = OnnxDetector(a.model, imgsz=a.imgsz, score_floor=0.01, nms_iou=cfg.nms_iou)
    print(f"evaluating {len(rows)} '{a.split}' images with {a.model} ...")
    t0 = time.time()
    records = collect_predictions(detector, rows, paths,
                                  lambda i, n: print(f"\r  {i}/{n}", end="", flush=True) if i % 10 == 0 or i == n else None)
    print(f"\n  done in {time.time() - t0:.0f}s")

    pts = sweep(records, DEFAULT_THRESHOLDS, a.iou)
    selection = None
    if a.select_threshold:
        best, why = select_threshold(pts, a.max_neg_fp_rate, a.min_recall)
        threshold, selection = best.threshold, {"rule": why, "max_neg_fp_rate": a.max_neg_fp_rate, "min_recall": a.min_recall}
        status = "selected on validation split"
        if why.startswith("rule 2"):
            print(f"WARNING: no threshold met max negative-image FP rate {a.max_neg_fp_rate} with recall >= {a.min_recall}; "
                  "using the F0.5 fallback. The model likely needs more/better data rather than a different threshold.")
    else:
        if a.threshold is not None:
            threshold, status = a.threshold, "set by --threshold"
        else:
            threshold = cfg.threshold
            status = "from configs/inference.yaml" + ("" if cfg.threshold_validated else " (NOT validation-derived)")
    thr_pt = next((p for p in pts if abs(p.threshold - threshold) < 1e-9), None)
    res = overall(records, threshold, a.iou)

    # ---- second class: 'other note' (red box) ----------------------------------------------------------------
    two_class = (detector.num_classes or 1) > 1
    other_thr = other_status = other_sel = res_other = None
    if two_class:
        pts1 = sweep(records, DEFAULT_THRESHOLDS, a.iou, target_class=1)
        if a.select_threshold:
            best1, why1 = select_threshold(pts1, a.max_neg_fp_rate, a.min_recall)
            other_thr, other_sel, other_status = best1.threshold, {"rule": why1}, "selected on validation split"
        else:
            other_thr, other_status = cfg.other_threshold, "from configs/inference.yaml"
        res_other = overall(records, other_thr, a.iou, target_class=1)
        res_other["threshold"] = other_thr
        with (Path(a.out) / f"threshold_sweep_other_{a.split}.csv").open("w", newline="") as fh:
            w1 = csv.writer(fh)
            w1.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "neg_images", "neg_images_with_fp"])
            Path(a.out).mkdir(parents=True, exist_ok=True)
            for p in pts1:
                w1.writerow([p.threshold, p.tp, p.fp, p.fn, f"{p.precision:.4f}", f"{p.recall:.4f}" if p.recall == p.recall else "",
                             p.neg_images, p.neg_images_with_fp])

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": a.split, "threshold": threshold, "threshold_status": status, "threshold_selection": selection,
        "iou_threshold": a.iou, "images": len(rows),
        "model": {"path": str(a.model), "sha256": sha256_file(Path(a.model)), "imgsz": detector.imgsz,
                  "size_bytes": Path(a.model).stat().st_size},
        "dataset": {"manifest_hash": manifest_hash(all_rows), "total_images": len(all_rows)},
        "hardware": hardware_info(),
        "overall": res, "ap50_all_scores": average_precision(records, a.iou),
        "by_denomination": group_stats(records, threshold, lambda r: r.meta["denomination"], a.iou),
        "by_side": group_stats(records, threshold, lambda r: f"{r.meta['denomination']}/{r.meta['side']}", a.iou),
        "by_lighting": group_stats(records, threshold, lambda r: r.meta["lighting"], a.iou),
        "by_angle": group_stats(records, threshold, lambda r: r.meta["angle"], a.iou),
        "by_background": group_stats(records, threshold, lambda r: r.meta["background"], a.iou),
        "by_session": group_stats(records, threshold, lambda r: r.meta["session"], a.iou),
        "required_scenarios": {name: (group_stats([r for r in records if fn(r)], threshold, lambda r: name, a.iou).get(name, "NOT TESTED"))
                               for name, fn in REQUIRED_SCENARIOS.items()},
        "latency": latency_summary(records),
        "two_class_model": two_class, "other_threshold": other_thr, "other_threshold_status": other_status,
        "other_threshold_selection": other_sel, "other_note_overall": res_other,
        "app_output_by_denomination": note_type_table(records, threshold, other_thr, lambda r: r.meta["denomination"]),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"threshold_sweep_{a.split}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "neg_images", "neg_images_with_fp", "neg_image_fp_rate"])
        for p in pts:
            w.writerow([p.threshold, p.tp, p.fp, p.fn, f"{p.precision:.4f}", f"{p.recall:.4f}" if p.recall == p.recall else "",
                        f"{p.f1:.4f}", p.neg_images, p.neg_images_with_fp, f"{p.neg_fp_rate:.4f}" if p.neg_images else ""])
    plot_curves(pts, out, a.split, threshold)
    (out / f"eval_{a.split}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md = [f"# Evaluation: `{a.split}` split", "",
          f"- Generated: {report['generated_utc']}", f"- Model: `{a.model}` sha256 `{report['model']['sha256'][:16]}` (imgsz {detector.imgsz})",
          f"- Dataset manifest hash: `{report['dataset']['manifest_hash']}`; images in this split: {len(rows)}",
          f"- Threshold: **{threshold:.2f}** ({status}); IoU for matching: {a.iou}",
          f"- Hardware: {report['hardware']['cpu']}, {report['hardware']['logical_cores']} logical cores, OpenCV {cv2.__version__}", ""]
    if selection:
        md += [f"- Selection: {selection['rule']}", ""]
    if a.split == "train":
        md += ["> **Training split: metrics are optimistic (the model was trained on these images).**", ""]
    md += ["## Overall", "",
           f"- Detection level: TP {res['det_tp']}, FP {res['det_fp']}, FN {res['det_fn']}; precision {fmt(res['precision'], True)}, "
           f"recall {fmt(res['recall'], True)}, F1 {fmt(res['f1'])}; AP@{a.iou} {fmt(res['ap50'])} (all scores)",
           f"- Image level: {res['positive_images']} positive / {res['negative_images']} negative images; "
           f"missed positives {res['fn_images']}; **negative images with a false ₹50: {res['fp_images']}/{res['negative_images']}**"
           + (f" (95% CI upper bound {fmt(res['neg_image_fp_rate_ci95_upper'], True)})" if res["negative_images"] else ""),
           "- Image-level confusion matrix (rows actual, cols predicted):", "",
           "| | predicted ₹50 | predicted none |", "|---|---|---|",
           f"| actual ₹50 | {res['confusion_image_level']['actual_target__pred_target']} | {res['confusion_image_level']['actual_target__pred_none']} |",
           f"| actual none | {res['confusion_image_level']['actual_none__pred_target']} | {res['confusion_image_level']['actual_none__pred_none']} |", ""]
    md += ["## What the app shows, per denomination (green ₹50 wins over red 'Other note')", "",
           f"Thresholds: ₹50 {threshold:.2f}" + (f", other note {other_thr:.2f} ({other_status})" if two_class else " (single-class model: no red box)"),
           "", "| denomination | images | shown ₹50 (green) | shown Other note (red) | shown nothing |", "|---|---|---|---|---|"]
    for k, g in report["app_output_by_denomination"].items():
        md.append(f"| {k} | {g['images']} | {g['shown_rs50']} | {g['shown_other']} | {g['shown_none']} |")
    md.append("")
    if res_other:
        md += [f"- 'Other note' class at threshold {other_thr:.2f}: TP {res_other['det_tp']}, FP {res_other['det_fp']}, FN {res_other['det_fn']}; "
               f"precision {fmt(res_other['precision'], True)}, recall {fmt(res_other['recall'], True)}; AP@{a.iou} {fmt(res_other['ap50'])}",
               ""]
        if other_sel:
            md += [f"- Other-note threshold selection: {other_sel['rule']}", ""]
    md += group_table("By denomination", report["by_denomination"])
    md += group_table("Target note by side", {k: v for k, v in report["by_side"].items() if k.startswith(("50_new", "mixed"))})
    md += group_table("By lighting", report["by_lighting"])
    md += group_table("By angle", report["by_angle"])
    md += group_table("By capture session", report["by_session"])
    md += ["### Required scenarios", ""]
    for name, g in report["required_scenarios"].items():
        md.append(f"- {name}: " + ("**NOT TESTED** (no images)" if g == "NOT TESTED" else
                  f"{g['tp_images']}/{g['positive_images']} found, {g['fn_images']} missed"))
    lat = report["latency"]
    md += ["", "## Latency (this machine, per image, includes decode)", "",
           f"- Inference (forward) median {lat['inference_ms_median']:.1f} ms, p95 {lat['inference_ms_p95']:.1f} ms",
           f"- Full pipeline (pre+forward+post) median {lat['pipeline_ms_median']:.1f} ms, p95 {lat['pipeline_ms_p95']:.1f} ms "
           f"(~{lat['throughput_fps_pipeline']:.1f} FPS excluding camera/display)", ""]
    if a.select_threshold and thr_pt:
        md += [f"## Threshold trade-off at {threshold:.2f} (val)", "",
               f"- precision {fmt(thr_pt.precision, True)}, recall {fmt(thr_pt.recall, True)}, negative-image FP {thr_pt.neg_images_with_fp}/{thr_pt.neg_images}",
               f"- Full sweep: `threshold_sweep_{a.split}.csv`, plot `threshold_analysis_{a.split}.png`", ""]
    md += ["Scores are detector confidences, not calibrated probabilities. Rates from few images are unreliable; see the CI column.", ""]
    (out / f"eval_{a.split}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:24]))

    if a.write_config:
        cfg_path = Path(a.config)
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        if two_class:
            data.update(other_class_id=1, other_threshold=float(other_thr))
        data.update(threshold=float(threshold), threshold_validated=True, imgsz=detector.imgsz,
                    threshold_source=f"val sweep, {len(rows)} images, model {report['model']['sha256'][:12]}, {report['generated_utc'][:10]}")
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"wrote threshold {threshold:.2f} to {cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
