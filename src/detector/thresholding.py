"""Accept/reject decisions, temporal confirmation, and validation-driven threshold selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .postprocessing import Candidate

ACCEPTED = "accepted"
BELOW_THRESHOLD = "below_threshold"
WRONG_CLASS = "wrong_class"
INVALID_BOX = "invalid_box"
IMPLAUSIBLE_SHAPE = "implausible_shape"

# Geometry of a banknote seen by a camera. A Rs.50 is 135x66 mm (aspect 2.05); perspective, rotation and
# partial occlusion stretch that, but never to a hairline strip or a speck. These limits are fixed from the
# physical object, not tuned on evaluation images (added after a live false positive on a thin window frame).
MIN_BOX_FRAC = 0.05   # shorter box side >= 5% of the shorter frame side
MAX_ASPECT = 4.5      # longer side / shorter side


def plausible_shape(box: tuple[float, float, float, float], frame_w: int, frame_h: int,
                    min_frac: float = MIN_BOX_FRAC, max_aspect: float = MAX_ASPECT) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0 or frame_w <= 0 or frame_h <= 0:
        return False
    return min(w, h) >= min_frac * min(frame_w, frame_h) and max(w, h) / min(w, h) <= max_aspect


@dataclass(frozen=True)
class Decision:
    candidate: Candidate
    accepted: bool
    reason: str


def decide(
    candidates: Iterable[Candidate],
    threshold: float,
    target_class_id: int = 0,
    frame_w: int | None = None,
    frame_h: int | None = None,
    min_box_frac: float = MIN_BOX_FRAC,
    max_aspect: float = MAX_ASPECT,
) -> list[Decision]:
    """Label every candidate accepted/rejected with a machine-readable reason."""
    out: list[Decision] = []
    for c in candidates:
        if c.class_id != target_class_id:
            out.append(Decision(c, False, WRONG_CLASS))
        elif c.width <= 0 or c.height <= 0 or (frame_w is not None and c.x1 >= frame_w) or (
            frame_h is not None and c.y1 >= frame_h
        ):
            out.append(Decision(c, False, INVALID_BOX))
        elif frame_w and frame_h and not plausible_shape(c.box, frame_w, frame_h, min_box_frac, max_aspect):
            out.append(Decision(c, False, IMPLAUSIBLE_SHAPE))
        elif c.score < threshold:
            out.append(Decision(c, False, BELOW_THRESHOLD))
        else:
            out.append(Decision(c, True, ACCEPTED))
    return out


def accepted(decisions: Iterable[Decision]) -> list[Candidate]:
    return [d.candidate for d in decisions if d.accepted]


class TemporalConfirmer:
    """Require k hits in the last m frames to enter the 'detected' state, with hysteresis to leave.

    ``confirm_frames <= 1`` disables the filter entirely (pass-through), so evaluation and
    debugging see the raw per-frame decision. A box is only ever shown for the *current*
    frame, never replayed from an earlier one.
    """

    def __init__(self, confirm_frames: int = 1, window_frames: int = 5, release_ratio: float = 0.8, grace: int = 2):
        if confirm_frames < 1 or window_frames < confirm_frames:
            raise ValueError("need 1 <= confirm_frames <= window_frames")
        if not (0.0 < release_ratio <= 1.0):
            raise ValueError("release_ratio must be in (0, 1]")
        self.confirm_frames = confirm_frames
        self.window_frames = window_frames
        self.release_ratio = release_ratio
        self.grace = grace
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.confirm_frames > 1

    def reset(self) -> None:
        self._hits: deque[bool] = deque(maxlen=self.window_frames)
        self._active = False
        self._misses = 0

    @property
    def state(self) -> str:
        if not self.enabled:
            return "off"
        if self._active:
            return "locked"
        return f"confirming {sum(self._hits)}/{self.confirm_frames}"

    def update(self, decisions: Sequence[Decision], threshold: float) -> Candidate | None:
        """Return the candidate to display for this frame, or None."""
        best = max((d.candidate for d in decisions if d.accepted), key=lambda c: c.score, default=None)
        if not self.enabled:
            return best
        release = threshold * self.release_ratio
        loose = max(
            (d.candidate for d in decisions if d.reason in (ACCEPTED, BELOW_THRESHOLD) and d.candidate.score >= release),
            key=lambda c: c.score,
            default=None,
        )
        if not self._active:
            self._hits.append(best is not None)
            if best is not None and sum(self._hits) >= self.confirm_frames:
                self._active, self._misses = True, 0
                return best
            return None
        if loose is not None:
            self._misses = 0
            return loose
        self._misses += 1
        if self._misses >= self.grace:
            self.reset()
        return None


# ---------------------------------------------------------------------------
# Threshold selection (used by scripts/evaluate.py on the VALIDATION split only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    tp: int
    fp: int
    fn: int
    neg_images_with_fp: int
    neg_images: int
    precision: float
    recall: float
    f1: float

    @property
    def neg_fp_rate(self) -> float:
        return self.neg_images_with_fp / self.neg_images if self.neg_images else float("nan")


def select_threshold(
    sweep: Sequence[SweepPoint],
    max_neg_fp_rate: float = 0.0,
    min_recall: float = 0.80,
    beta: float = 0.5,
) -> tuple[SweepPoint, str]:
    """Pick the operating point favouring false-positive avoidance.

    1. Among points whose negative-image FP rate <= ``max_neg_fp_rate`` and recall >= ``min_recall``,
       choose the highest-recall one; ties are resolved to the MIDDLE of the tied range.
    2. If none qualifies, fall back to the point maximising F-beta (beta<1 weights precision more).
    The returned string says which rule fired so the report is honest about it.
    """
    if not sweep:
        raise ValueError("empty sweep")
    ok = [p for p in sweep if p.neg_images and p.neg_fp_rate <= max_neg_fp_rate and p.recall >= min_recall]
    if ok:
        top = max(p.recall for p in ok)
        plateau = sorted((p for p in ok if p.recall == top), key=lambda p: p.threshold)
        best = plateau[len(plateau) // 2]  # middle of the plateau: equal margin to both failure modes
        return best, (
            f"rule 1: negative-image FP rate <= {max_neg_fp_rate:.3f} and recall >= {min_recall:.2f}; "
            f"highest recall; the {len(plateau)} tied thresholds span {plateau[0].threshold:.2f}-{plateau[-1].threshold:.2f}, "
            "the middle one is used (not the extreme) so it keeps margin on both sides"
        )

    def fbeta(p: SweepPoint) -> float:
        d = beta * beta * p.precision + p.recall
        return (1 + beta * beta) * p.precision * p.recall / d if d > 0 else 0.0

    best = max(sweep, key=lambda p: (fbeta(p), p.threshold))
    return best, f"rule 2 (fallback): no point met the FP/recall constraints; maximised F{beta:g}"
