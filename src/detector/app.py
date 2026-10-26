"""Command-line webcam application behind the root-level test.py."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import ConfigError, load_config, validate_imgsz, validate_threshold
from .inference import InferenceError, ModelLoadError, OnnxDetector
from .logging_utils import get_logger, setup_logging
from .preprocessing import InvalidImageError
from .thresholding import TemporalConfirmer, accepted, decide
from .visualization import (
    GREEN, draw_detection, draw_hud, draw_inspector, draw_status_banner, rupee_supported,
)

EXIT_OK, EXIT_USAGE, EXIT_MODEL, EXIT_CAMERA, EXIT_INFERENCE = 0, 2, 3, 4, 5
WINDOW = "Rs50 Vision - q to quit, i inspector, +/- threshold"
MAX_CONSECUTIVE_INFERENCE_ERRORS = 30


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test.py",
        description="Live Indian Rs.50 (new, light blue-green) banknote detector. "
        "Runs an exported ONNX model through OpenCV's DNN module.",
        epilog="Example: python test.py --model model.onnx",
    )
    p.add_argument("--model", default="model.onnx", help="path to the exported ONNX model (default: model.onnx)")
    p.add_argument("--threshold", type=float, default=None,
                   help="confidence threshold in (0,1); default comes from configs/inference.yaml")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    p.add_argument("--source", default=None, help="image or video file to use instead of the webcam")
    p.add_argument("--imgsz", type=int, default=None, help="model input size; default read from model metadata/config")
    p.add_argument("--width", type=int, default=1280, help="requested camera width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="requested camera height (default: 720)")
    p.add_argument("--confirm-frames", type=int, default=None,
                   help="require N detections in the recent window before showing one (1 = off, default)")
    p.add_argument("--inspector", action="store_true", help="start with the decision inspector overlay enabled")
    p.add_argument("--no-display", action="store_true", help="headless: no window; stop after --max-frames")
    p.add_argument("--max-frames", type=int, default=None, help="stop after this many processed frames")
    p.add_argument("--summary-json", default=None, help="write a run summary (timings, counts) to this JSON file")
    p.add_argument("--config", default=None, help="inference config YAML (default: configs/inference.yaml)")
    p.add_argument("--log-file", default=None, help="also write logs to this file")
    p.add_argument("--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__} (OpenCV {cv2.__version__})")
    return p


def _pct(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _error_canvas(w: int, h: int, lines: list[str]) -> np.ndarray:
    canvas = np.full((h, w, 3), 35, np.uint8)
    y = h // 2 - 12 * len(lines)
    for ln in lines:
        cv2.putText(canvas, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        y += 32
    return canvas


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.log_file)
    log = get_logger()

    try:
        cfg = load_config(args.config)
        threshold = validate_threshold(args.threshold) if args.threshold is not None else cfg.threshold
        imgsz = validate_imgsz(args.imgsz) if args.imgsz is not None else None
        confirm = args.confirm_frames if args.confirm_frames is not None else cfg.confirm_frames
        if confirm < 1 or confirm > cfg.window_frames:
            raise ConfigError(f"--confirm-frames must be between 1 and {cfg.window_frames}")
        if args.max_frames is not None and args.max_frames < 1:
            raise ConfigError("--max-frames must be >= 1")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    threshold_note = "validated on val split" if (cfg.threshold_validated and args.threshold is None) else (
        "set by --threshold" if args.threshold is not None else "NOT VALIDATED (provisional default)"
    )
    log.info("Threshold %.2f (%s)", threshold, threshold_note)
    if not rupee_supported():
        log.warning("Rupee sign font not found; label will read 'Rs50'")

    try:
        detector = OnnxDetector(args.model, imgsz=imgsz, score_floor=min(cfg.candidate_floor, threshold),
                                nms_iou=cfg.nms_iou)
    except ModelLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Train and export a model first (see README), or pass --model <path to .onnx>.", file=sys.stderr)
        return EXIT_MODEL

    from .webcam import CameraError, FrameSource

    try:
        source = FrameSource(camera=args.camera, source=args.source, width=args.width, height=args.height)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CAMERA

    still_image = source._still is not None
    max_frames = args.max_frames
    if max_frames is None and (args.no_display):
        max_frames = 1 if still_image else 100
    confirmer = TemporalConfirmer(confirm, cfg.window_frames, cfg.release_threshold_ratio)
    inspector = args.inspector
    show = not args.no_display

    totals: deque[float] = deque(maxlen=5000)
    infers: deque[float] = deque(maxlen=5000)
    recent: deque[float] = deque(maxlen=30)
    processed = detected_frames = read_failures = inference_errors = consecutive_errors = 0
    exit_code = EXIT_OK
    window_ready = False
    t_prev = time.perf_counter()

    try:
        while max_frames is None or processed < max_frames:
            try:
                ok, frame, status = source.read()
            except CameraError as exc:
                print(f"error: {exc}", file=sys.stderr)
                exit_code = EXIT_CAMERA
                break
            if not ok:
                if status.startswith("end of"):
                    log.info("Reached %s", status)
                    break
                read_failures += 1
                if show:
                    cv2.imshow(WINDOW, _error_canvas(args.width, args.height, ["Camera unavailable", status]))
                    window_ready = True
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            result = None
            banner, ok_state = "Searching for Rs.50 note...", False
            shown = None
            decisions = []
            try:
                result = detector.infer(frame)
                decisions = decide(result.candidates, threshold, cfg.target_class_id, frame.shape[1], frame.shape[0])
                shown = confirmer.update(decisions, threshold)
                consecutive_errors = 0
            except (InferenceError, InvalidImageError) as exc:
                inference_errors += 1
                consecutive_errors += 1
                banner = f"INFERENCE ERROR: {exc}"[:90]
                log.error("Inference failed: %s", exc)
                if consecutive_errors >= MAX_CONSECUTIVE_INFERENCE_ERRORS:
                    print("error: inference keeps failing; aborting.", file=sys.stderr)
                    exit_code = EXIT_INFERENCE
                    break
            processed += 1
            if shown is not None:
                detected_frames += 1
                banner, ok_state = f"Rs.50 DETECTED  score {shown.score:.2f}", True
            elif confirmer.enabled and confirmer.state.startswith("confirming") and any(d.accepted for d in decisions):
                banner = f"Confirming... {confirmer.state}"

            if result is not None:
                infers.append(result.timings.inference_ms)
                totals.append(result.timings.total_ms)

            if show:
                out = frame
                if shown is not None:
                    draw_detection(out, shown)
                if inspector and result is not None:
                    draw_inspector(out, decisions)
                now = time.perf_counter()
                recent.append(now - t_prev)
                t_prev = now
                fps = 1.0 / statistics.fmean(recent) if recent else 0.0
                hud = [
                    f"threshold {threshold:.2f} [{threshold_note}]",
                    f"inference {infers[-1]:.1f} ms | {fps:.1f} fps" if infers else "inference n/a",
                ]
                if confirmer.enabled:
                    hud.append(f"temporal: {confirmer.state}")
                if inspector:
                    top = max((d for d in decisions), key=lambda d: d.candidate.score, default=None)
                    hud.append("top: none" if top is None else
                               f"top raw score {top.candidate.score:.2f} -> {top.reason}")
                draw_hud(out, hud)
                draw_status_banner(out, banner, ok_state)
                cv2.imshow(WINDOW, out)
                window_ready = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("i"):
                    inspector = not inspector
                elif key in (ord("+"), ord("=")):
                    threshold = min(0.95, round(threshold + 0.05, 2))
                    threshold_note = "set live"
                elif key in (ord("-"), ord("_")):
                    threshold = max(0.05, round(threshold - 0.05, 2))
                    threshold_note = "set live"
                if processed > 3 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        log.info("Interrupted")
    except cv2.error as exc:
        print(f"error: display failed ({str(exc).strip().splitlines()[-1]}). "
              "If OpenCV is the headless build, reinstall opencv-python or use --no-display.", file=sys.stderr)
        exit_code = EXIT_CAMERA
    finally:
        source.release()
        if window_ready or show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    summary = {
        "frames_processed": processed,
        "frames_with_accepted_detection": detected_frames,
        "camera_read_failures": read_failures,
        "inference_errors": inference_errors,
        "threshold": threshold,
        "threshold_status": threshold_note,
        "temporal_confirm_frames": confirm,
        "opencv_version": cv2.__version__,
        "model": str(detector.model_path),
        "imgsz": detector.imgsz,
        "model_init_ms": round(detector.init_ms, 1),
        "inference_ms": {"mean": round(statistics.fmean(infers), 2) if infers else None,
                         "median": round(_pct(infers, 50), 2) if infers else None,
                         "p95": round(_pct(infers, 95), 2) if infers else None},
        "total_ms": {"mean": round(statistics.fmean(totals), 2) if totals else None,
                     "median": round(_pct(totals, 50), 2) if totals else None,
                     "p95": round(_pct(totals, 95), 2) if totals else None},
    }
    log.info("Done: %d frames, %d with accepted detection; inference median %s ms",
             processed, detected_frames, summary["inference_ms"]["median"])
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    return run(argv)
