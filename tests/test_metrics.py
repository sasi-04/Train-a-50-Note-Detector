import numpy as np
import pytest

from detector.metrics import (ImageRecord, average_precision, group_stats, match, overall, sweep, wilson_interval)

GT = np.array([[100, 100, 300, 200.0]])


def rec(preds, gt=GT, den="50_new", iid="i"):
    return ImageRecord(iid, {"denomination": den}, np.asarray(gt, float).reshape(-1, 4), preds)


GOOD = ((102, 101, 298, 199), 0.9, 0)
OFF = ((500, 500, 600, 600), 0.8, 0)


def test_match_tp_fp_fn():
    assert match(rec([GOOD, OFF]), 0.5) == (1, 1, 0)
    assert match(rec([OFF]), 0.5) == (0, 1, 1)
    assert match(rec([GOOD]), 0.95) == (0, 0, 1)  # below threshold -> missed


def test_wrong_class_ignored_in_matching():
    assert match(rec([((102, 101, 298, 199), 0.9, 2)]), 0.5) == (0, 0, 1)


def test_duplicate_detections_count_one_tp_one_fp():
    assert match(rec([GOOD, ((101, 100, 299, 200), 0.85, 0)]), 0.5) == (1, 1, 0)


def test_negative_image_fp_counted_once_per_image():
    neg = rec([GOOD, OFF], gt=np.zeros((0, 4)), den="20")
    pts = sweep([neg, rec([], gt=np.zeros((0, 4)), den="10")], [0.5])
    assert pts[0].neg_images == 2 and pts[0].neg_images_with_fp == 1 and pts[0].neg_fp_rate == 0.5


def test_sweep_monotonic_fp():
    recs = [rec([GOOD]), rec([OFF], gt=np.zeros((0, 4)), den="500")]
    pts = sweep(recs, [0.1, 0.5, 0.85, 0.95])
    assert [p.fp for p in pts] == [1, 1, 0, 0]
    assert [p.tp for p in pts] == [1, 1, 1, 0]


def test_group_stats_by_denomination():
    recs = [rec([GOOD]), rec([], den="50_new"), rec([OFF], gt=np.zeros((0, 4)), den="20"),
            rec([], gt=np.zeros((0, 4)), den="20")]
    g = group_stats(recs, 0.5, lambda r: r.meta["denomination"])
    assert g["50_new"]["positive_images"] == 2 and g["50_new"]["tp_images"] == 1 and g["50_new"]["fn_images"] == 1
    assert g["20"]["negative_images"] == 2 and g["20"]["fp_images"] == 1


def test_overall_confusion_adds_up():
    recs = [rec([GOOD]), rec([], den="50_new"), rec([OFF], gt=np.zeros((0, 4)), den="20")]
    o = overall(recs, 0.5)
    c = o["confusion_image_level"]
    assert sum(c.values()) == 3 and c["actual_target__pred_target"] == 1 and c["actual_none__pred_target"] == 1
    assert o["precision"] == pytest.approx(0.5) and o["recall"] == pytest.approx(0.5)


def test_average_precision_perfect_and_degenerate():
    assert average_precision([rec([GOOD])]) == pytest.approx(1.0)
    assert average_precision([rec([OFF])]) == pytest.approx(0.0)
    assert np.isnan(average_precision([rec([OFF], gt=np.zeros((0, 4)))]))


def test_class_aware_matching_and_negatives():
    # a labelled 'other note' (class 1): positive for class 1, negative for class 0
    r = ImageRecord("o", {"denomination": "20"}, np.array([[100, 100, 300, 200.0]]),
                    [((102, 101, 298, 199), 0.9, 1), ((102, 101, 298, 199), 0.6, 0)], gt_cls=np.array([1]))
    assert not r.positive and r.has(1)
    assert match(r, 0.5, target_class=1) == (1, 0, 0)
    assert match(r, 0.5, target_class=0) == (0, 1, 0)  # a Rs.50 box on an other note is a FP for class 0
    pts0 = sweep([r], [0.5], target_class=0)
    assert pts0[0].neg_images == 1 and pts0[0].neg_images_with_fp == 1
    pts1 = sweep([r], [0.5], target_class=1)
    assert pts1[0].neg_images == 0 and pts1[0].tp == 1


def test_shown_label_follows_app_priority():
    from detector.metrics import note_type_table, shown_label
    both = ImageRecord("b", {"d": "x"}, np.zeros((0, 4)), [((10, 10, 300, 200), 0.8, 0), ((10, 10, 300, 200), 0.9, 1)])
    assert shown_label(both, 0.5, 0.5) == "rs50"          # Rs.50 wins over other
    other = ImageRecord("o", {"d": "y"}, np.zeros((0, 4)), [((10, 10, 300, 200), 0.9, 1)])
    assert shown_label(other, 0.5, 0.5) == "other" and shown_label(other, 0.5, None) == "none"
    t = note_type_table([both, other], 0.5, 0.5, lambda r: r.meta["d"])
    assert t["x"]["shown_rs50"] == 1 and t["y"]["shown_other"] == 1


def test_wilson_interval_properties():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35  # 0/10 observed does NOT mean 0% true rate
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
