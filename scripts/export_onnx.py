"""Export a trained Ultralytics checkpoint to ONNX for the OpenCV deployment path.

    python scripts/export_onnx.py --weights runs/rs50/weights/best.pt --out model.onnx

Settings are chosen for cv2.dnn compatibility: fixed input size, opset 12, no embedded NMS
(NMS is done in src/detector/postprocessing.py), FP32. A sidecar <name>.meta.json records the
input size, class names and hashes so test.py can pick the right size automatically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from detector import DISPLAY_LABEL, TARGET_CLASS_NAME
from detector.dataset import sha256_file


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="trained .pt checkpoint (e.g. best.pt)")
    p.add_argument("--out", default="model.onnx")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--no-simplify", action="store_true", help="skip onnx-simplifier (avoids an extra pip install)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = p.parse_args()

    weights, out = Path(a.weights), Path(a.out)
    if not weights.is_file():
        print(f"error: checkpoint not found: {weights}", file=sys.stderr)
        return 2
    if out.exists() and not a.force:
        print(f"error: {out} exists; pass --force to overwrite", file=sys.stderr)
        return 2

    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    exported = Path(model.export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=not a.no_simplify,
                                 dynamic=False, half=False, nms=False, batch=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out.resolve():
        shutil.move(str(exported), out)  # Ultralytics writes next to the .pt; keep only the requested artifact

    graph = onnx.load(str(out))
    onnx.checker.check_model(graph)
    out_shape = [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim]
    meta = {
        "format": "onnx", "imgsz": a.imgsz, "opset": a.opset, "output_shape": out_shape,
        "class_names": {int(k): v for k, v in names.items()},
        "target_class": TARGET_CLASS_NAME, "display_label": DISPLAY_LABEL, "num_classes": len(names),
        "preprocess": "letterbox pad=114, RGB, /255, NCHW float32",
        "source_checkpoint": str(weights), "source_checkpoint_sha256": sha256_file(weights),
        "onnx_sha256": sha256_file(out), "onnx_size_bytes": out.stat().st_size,
        "ultralytics": ultralytics.__version__, "torch": torch.__version__, "onnx": onnx.__version__,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB), output shape {out_shape}")
    print("next: python scripts/validate_onnx.py --model", out, "--weights", weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
