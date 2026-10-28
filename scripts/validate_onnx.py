"""Validate an exported ONNX model against the deployment pipeline (and the source checkpoint).

    python scripts/validate_onnx.py --model model.onnx [--weights runs/.../best.pt] [--images dir]

Checks: file/graph validity, input/output spec, loads in OpenCV (cv2.dnn), real inference, numeric
sanity, OpenCV vs ONNX Runtime raw-tensor agreement, and (with --weights) PyTorch vs OpenCV raw-tensor
and decoded-detection parity. Writes reports/onnx_validation.{json,md}; exit 0 only if every check passes.

If --images is omitted, deterministic SYNTHETIC test images are used. They exercise the numerics and
geometry only; they say nothing about banknote detection quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from detector.dataset import IMAGE_EXTS, sha256_file
from detector.inference import OnnxDetector, read_sidecar
from detector.postprocessing import decode_output, describe_layout, iou_one_to_many
from detector.preprocessing import preprocess

TOL_RAW_BOX_PX = 0.5      # raw box coordinate difference, in 640-px model space
TOL_RAW_SCORE = 2e-3      # raw class-score difference
MIN_DET_AGREEMENT = 0.99  # fraction of reference detections matched at IoU >= 0.9


def synthetic_images(n: int = 6, size=(720, 1280)) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    imgs = []
    for i in range(n):
        h, w = size if i % 2 == 0 else (size[1], size[0])
        img = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (200, 180, 90), -1)
        imgs.append(img)
    return imgs


def load_images(folder: str | None, limit: int = 12) -> tuple[list[np.ndarray], str]:
    if not folder:
        return synthetic_images(), "synthetic (numerics only)"
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)[:limit]
    imgs = [im for im in (cv2.imread(str(p)) for p in paths) if im is not None]
    if not imgs:
        raise SystemExit(f"error: no readable images in {folder}")
    return imgs, f"{len(imgs)} images from {folder}"


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, passed: bool | None, detail: str) -> None:
        self.checks.append({"check": name, "status": "PASS" if passed else ("SKIP" if passed is None else "FAIL"),
                            "detail": detail})
        print(f"[{self.checks[-1]['status']}] {name}: {detail}")

    @property
    def ok(self) -> bool:
        return all(c["status"] != "FAIL" for c in self.checks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.onnx")
    ap.add_argument("--weights", default=None, help="source .pt checkpoint for PyTorch parity")
    ap.add_argument("--images", default=None, help="folder of real images (recommended: dataset/images/pool)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    R = Report()
    model = Path(a.model)
    info: dict = {"model": str(model)}

    if not model.is_file():
        R.add("file exists", False, f"{model} not found")
        return finish(R, a.out, info)
    info.update(size_bytes=model.stat().st_size, sha256=sha256_file(model))
    R.add("file exists", True, f"{model.stat().st_size / 1e6:.1f} MB, sha256 {info['sha256'][:12]}")

    meta = read_sidecar(model)
    imgsz = a.imgsz or meta.get("imgsz") or 640

    try:
        import onnx
        g = onnx.load(str(model))
        onnx.checker.check_model(g)
        ins = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim],
                i.type.tensor_type.elem_type) for i in g.graph.input if i.name not in {x.name for x in g.graph.initializer}]
        outs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.graph.output]
        info.update(inputs=str(ins), outputs=str(outs), opset=[o.version for o in g.opset_import])
        R.add("ONNX graph valid", True, f"inputs {ins}, outputs {outs}, opset {info['opset']}")
        R.add("input is float32 NCHW 3-channel", len(ins) == 1 and ins[0][2] == 1 and ins[0][1][1] == 3,
              f"{ins[0][1]} elem_type={ins[0][2]}")
    except ImportError:
        R.add("ONNX graph valid", None, "onnx package not installed")
    except Exception as exc:  # noqa: BLE001 - report any checker failure
        R.add("ONNX graph valid", False, str(exc)[:200])
        return finish(R, a.out, info)

    t0 = time.perf_counter()
    try:
        det = OnnxDetector(model, imgsz=imgsz, score_floor=0.001)
    except Exception as exc:  # noqa: BLE001
        R.add("loads in OpenCV (cv2.dnn)", False, str(exc))
        return finish(R, a.out, info)
    R.add("loads in OpenCV (cv2.dnn)", True, f"OpenCV {cv2.__version__}, init {(time.perf_counter() - t0) * 1000:.0f} ms")
    layout = describe_layout(det.layout_shape)
    R.add("output layout recognised", True, f"shape {det.layout_shape} -> {layout}")
    nc = (det.layout_shape[1] - 4) if layout == "channels_first" else None
    if nc is not None:
        R.add("class head is Rs.50 (+ optional 'other note')", nc in (1, 2), f"{nc} class channel(s)")

    images, desc = load_images(a.images)
    info["test_images"] = desc
    R.add("test inputs", True, desc)

    raw_cv, blobs = [], []
    try:
        for im in images:
            blob, _ = preprocess(im, det.imgsz)
            blobs.append(blob)
            det.net.setInput(blob)
            raw_cv.append(np.asarray(det.net.forward(det.net.getUnconnectedOutLayersNames())[0]))
        finite = all(np.isfinite(r).all() for r in raw_cv)
        lo, hi = min(r[:, 4:].min() for r in raw_cv), max(r[:, 4:].max() for r in raw_cv)
        R.add("inference runs, output finite", finite, f"{len(raw_cv)} images")
        R.add("class scores within [0,1]", -1e-4 <= lo and hi <= 1 + 1e-4, f"min {lo:.4f} max {hi:.4f}")
    except Exception as exc:  # noqa: BLE001
        R.add("inference runs, output finite", False, str(exc)[:200])
        return finish(R, a.out, info)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        d_box = d_sc = 0.0
        for blob, rc in zip(blobs, raw_cv):
            ro = sess.run(None, {name: blob})[0]
            d_box = max(d_box, float(np.abs(ro[:, :4] - rc[:, :4]).max()))
            d_sc = max(d_sc, float(np.abs(ro[:, 4:] - rc[:, 4:]).max()))
        R.add("OpenCV vs ONNX Runtime raw tensors", d_box <= TOL_RAW_BOX_PX and d_sc <= TOL_RAW_SCORE,
              f"max box diff {d_box:.4g} px, max score diff {d_sc:.3g}")
    except ImportError:
        R.add("OpenCV vs ONNX Runtime raw tensors", None, "onnxruntime not installed")

    if a.weights:
        try:
            import torch
            from ultralytics import YOLO
            yolo = YOLO(a.weights)
            net = yolo.model.eval().float()
            d_box = d_sc = 0.0
            agree = total = 0
            for im, blob, rc in zip(images, blobs, raw_cv):
                with torch.no_grad():
                    out = net(torch.from_numpy(blob))
                ref = (out[0] if isinstance(out, (list, tuple)) else out).numpy()
                d_box = max(d_box, float(np.abs(ref[:, :4] - rc[:, :4]).max()))
                d_sc = max(d_sc, float(np.abs(ref[:, 4:] - rc[:, 4:]).max()))
                _, meta_lb = preprocess(im, det.imgsz)
                a_det = decode_output(rc, meta_lb, 0.001, 0.45)[:20]
                b_det = decode_output(ref, meta_lb, 0.001, 0.45)[:20]
                for b in b_det:
                    total += 1
                    if a_det:
                        boxes = np.array([d.box for d in a_det])
                        ious = iou_one_to_many(np.array(b.box), boxes)
                        if ious.max() >= 0.9 and a_det[int(ious.argmax())].class_id == b.class_id:
                            agree += 1
            R.add("PyTorch vs OpenCV raw tensors", d_box <= TOL_RAW_BOX_PX and d_sc <= TOL_RAW_SCORE,
                  f"max box diff {d_box:.4g} px, max score diff {d_sc:.3g}")
            frac = agree / total if total else 1.0
            R.add("PyTorch vs OpenCV decoded detections", frac >= MIN_DET_AGREEMENT,
                  f"{agree}/{total} top detections matched at IoU>=0.9 ({frac:.1%})")
        except ImportError as exc:
            R.add("PyTorch parity", None, f"missing dependency: {exc}")
    else:
        R.add("PyTorch parity", None, "no --weights given")

    lat = []
    for _ in range(10):
        lat.append(det.infer(images[0]).timings.inference_ms)
    info["latency_ms_median"] = float(np.median(lat))
    R.add("latency measured", True, f"median {np.median(lat):.1f} ms/frame forward pass over {len(lat)} runs (this machine)")
    return finish(R, a.out, info)


def finish(R: Report, out_dir: str, info: dict) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info.update(opencv=cv2.__version__, numpy=np.__version__, passed=R.ok, checks=R.checks)
    (out / "onnx_validation.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    md = [f"# ONNX validation: {'PASS' if R.ok else 'FAIL'}", "", f"Model: `{info.get('model')}`", "",
          "| check | status | detail |", "|---|---|---|"]
    md += [f"| {c['check']} | {c['status']} | {c['detail'].replace('|', '/')} |" for c in R.checks]
    (out / "onnx_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("RESULT:", "PASS" if R.ok else "FAIL", "->", out / "onnx_validation.md")
    return 0 if R.ok else 1


if __name__ == "__main__":
    sys.exit(main())
