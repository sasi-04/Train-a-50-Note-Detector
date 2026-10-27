import pytest

from detector.postprocessing import Candidate
from detector.thresholding import (ACCEPTED, BELOW_THRESHOLD, INVALID_BOX, WRONG_CLASS, SweepPoint, TemporalConfirmer,
                                   accepted, decide, select_threshold)


def cand(score, cls=0, box=(10, 10, 100, 100)):
    return Candidate(*box, score, cls)


def test_accept_reject_reasons():
    ds = decide([cand(0.9), cand(0.4), cand(0.95, cls=3), cand(0.9, box=(5, 5, 5, 50))], 0.5)
    assert [d.reason for d in ds] == [ACCEPTED, BELOW_THRESHOLD, WRONG_CLASS, INVALID_BOX]
    assert len(accepted(ds)) == 1


def test_hairline_strip_is_rejected_as_implausible_shape():
    # the live false positive: a 1-px-wide, full-height "box" on a window frame, score above threshold
    strip = Candidate(838, 0, 842, 700, 0.57, 0)
    d = decide([strip], 0.5, frame_w=1280, frame_h=720)[0]
    assert not d.accepted and d.reason == "implausible_shape"


def test_plausible_shapes_for_a_note_in_either_orientation():
    from detector.thresholding import plausible_shape
    assert plausible_shape((100, 100, 500, 300), 1280, 720)      # landscape note, aspect 2
    assert plausible_shape((100, 100, 300, 500), 1280, 720)      # sideways note
    assert plausible_shape((100, 100, 400, 340), 1280, 720)      # near-square (close-up, tilted)
    assert not plausible_shape((100, 100, 130, 130), 1280, 720)  # speck: < 5% of the short side
    assert not plausible_shape((0, 0, 1280, 120), 1280, 720)     # long band, aspect > 4.5
    assert not plausible_shape((10, 10, 10, 90), 1280, 720)      # zero width


def test_threshold_boundary_is_inclusive():
    assert decide([cand(0.5)], 0.5)[0].accepted


def test_wrong_class_never_accepted_even_at_high_score():
    assert not decide([cand(0.99, cls=1)], 0.1)[0].accepted


def test_temporal_disabled_is_passthrough():
    t = TemporalConfirmer(1)
    assert not t.enabled
    assert t.update(decide([cand(0.9)], 0.5), 0.5).score == 0.9
    assert t.update(decide([cand(0.3)], 0.5), 0.5) is None


def test_temporal_requires_k_of_m_and_never_replays_stale_box():
    t = TemporalConfirmer(confirm_frames=3, window_frames=5, grace=2)
    hit, miss = decide([cand(0.9)], 0.5), decide([], 0.5)
    assert t.update(hit, 0.5) is None
    assert t.update(hit, 0.5) is None
    assert t.update(hit, 0.5) is not None  # 3rd hit -> locked
    assert t.update(miss, 0.5) is None     # no stale box shown during a miss
    assert t.update(miss, 0.5) is None     # grace exhausted -> reset
    assert t.state.startswith("confirming 0")


def test_temporal_hysteresis_keeps_lock_slightly_below_threshold():
    t = TemporalConfirmer(2, 3, release_ratio=0.8)
    hit = decide([cand(0.9)], 0.5)
    t.update(hit, 0.5)
    assert t.update(hit, 0.5) is not None
    weak = decide([cand(0.45)], 0.5)  # below 0.5 but above 0.4
    assert t.update(weak, 0.5) is not None


def test_temporal_single_flicker_not_shown():
    t = TemporalConfirmer(2, 3)
    assert t.update(decide([cand(0.9)], 0.5), 0.5) is None


def test_temporal_invalid_args():
    with pytest.raises(ValueError):
        TemporalConfirmer(4, 3)


def sp(t, tp, fp, fn, negfp, neg=20):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn)
    return SweepPoint(t, tp, fp, fn, negfp, neg, p, r, 2 * p * r / (p + r))


def test_select_threshold_prefers_zero_fp_with_best_recall():
    pts = [sp(0.3, 98, 12, 2, 5), sp(0.5, 95, 3, 5, 1), sp(0.7, 90, 0, 10, 0), sp(0.9, 60, 0, 40, 0)]
    best, why = select_threshold(pts, max_neg_fp_rate=0.0, min_recall=0.8)
    assert best.threshold == 0.7 and why.startswith("rule 1")


def test_select_threshold_falls_back_when_constraints_unmet():
    pts = [sp(0.3, 98, 12, 2, 5), sp(0.5, 95, 3, 5, 1), sp(0.9, 30, 0, 70, 0)]
    best, why = select_threshold(pts, 0.0, 0.8)
    assert why.startswith("rule 2") and best.threshold == 0.5


def test_select_threshold_ties_pick_middle_of_plateau_not_the_edge():
    pts = [sp(t / 100, 90, 0, 10, 0) for t in range(40, 95, 5)] + [sp(0.95, 50, 0, 50, 0)]
    best, why = select_threshold(pts)
    assert best.threshold == 0.65 and "middle" in why
