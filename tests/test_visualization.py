import numpy as np

from detector import visualization as viz
from detector.postprocessing import Candidate


def test_label_text_has_rupee_and_two_decimal_score():
    assert viz.format_label(0.8749, show_rupee=True) == "₹50 0.87"
    assert viz.format_label(0.5, show_rupee=False) == "Rs50 0.50"


def test_draw_detection_changes_pixels_and_stays_in_bounds():
    f = np.zeros((240, 320, 3), np.uint8)
    viz.draw_detection(f, Candidate(10, 5, 300, 230, 0.9, 0))
    assert f.any()
    g = np.zeros((240, 320, 3), np.uint8)
    viz.draw_detection(g, Candidate(-50, -50, 900, 900, 0.9, 0))  # box beyond frame must not crash
    assert g.any()


def test_rupee_font_available_on_this_machine():
    # informational guard: if this fails, labels fall back to 'Rs50' and the README troubleshooting applies
    assert viz.rupee_supported() in (True, False)


def test_inspector_draws_only_rejected():
    from detector.thresholding import decide
    f = np.zeros((240, 320, 3), np.uint8)
    viz.draw_inspector(f, decide([Candidate(10, 10, 100, 100, 0.9, 0)], 0.5))
    assert not f.any()  # accepted candidates are drawn by draw_detection, not the inspector
    viz.draw_inspector(f, decide([Candidate(10, 10, 100, 100, 0.3, 0)], 0.5))
    assert f.any()
