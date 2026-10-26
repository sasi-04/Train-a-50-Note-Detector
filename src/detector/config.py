"""Inference configuration: defaults, YAML loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "inference.yaml"

# Used only if configs/inference.yaml is absent. NOT validation-derived.
FALLBACK_THRESHOLD = 0.5


class ConfigError(ValueError):
    """Raised for invalid user-supplied configuration values."""


@dataclass
class InferenceConfig:
    imgsz: int = 640
    threshold: float = FALLBACK_THRESHOLD
    # True only once scripts/evaluate.py has selected the threshold on validation data.
    threshold_validated: bool = False
    threshold_source: str = "built-in fallback (not validated)"
    nms_iou: float = 0.45
    # Raw-score floor applied before NMS. Kept far below any sensible operating
    # threshold so the inspector can show *why* something was rejected.
    candidate_floor: float = 0.05
    target_class_id: int = 0
    confirm_frames: int = 1  # 1 == temporal confirmation disabled
    window_frames: int = 5
    release_threshold_ratio: float = 0.8  # hysteresis: stay "locked" above ratio*threshold
    extra: dict[str, Any] = field(default_factory=dict)


def validate_threshold(value: float, name: str = "threshold") -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number, got {value!r}") from None
    if v != v or not (0.0 < v < 1.0):
        raise ConfigError(f"{name} must be strictly between 0 and 1 (got {value!r})")
    return v


def validate_imgsz(value: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"imgsz must be an integer, got {value!r}") from None
    if v < 32 or v > 2048 or v % 32 != 0:
        raise ConfigError(f"imgsz must be a multiple of 32 between 32 and 2048 (got {value})")
    return v


def load_config(path: Path | str | None = None) -> InferenceConfig:
    """Load inference.yaml. A missing file yields defaults; a malformed one raises ConfigError."""
    cfg = InferenceConfig()
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.is_file():
        return cfg
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cannot parse {p}: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a YAML mapping")
    known = {f.name for f in fields(InferenceConfig)}
    for key, val in data.items():
        if key in known and key != "extra":
            setattr(cfg, key, val)
        else:
            cfg.extra[key] = val
    cfg.threshold = validate_threshold(cfg.threshold)
    cfg.imgsz = validate_imgsz(cfg.imgsz)
    if cfg.threshold_validated and cfg.threshold_source.startswith("built-in"):
        cfg.threshold_source = f"{p.name}"
    return cfg
