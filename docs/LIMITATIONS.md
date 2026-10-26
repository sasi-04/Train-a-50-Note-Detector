# Limitations

1. **No real model or real photographs yet.** Nothing about banknote accuracy, false-positive behaviour or the
   operating threshold has been measured. The default threshold in `configs/inference.yaml` is a provisional placeholder
   and `test.py` says so on screen.
2. **Not an authentication system.** It localises a visual pattern. It cannot tell a genuine note from a good fake, and
   a high score is not evidence of authenticity.
3. **Scores are not probabilities.** No calibration has been measured; do not read 0.87 as "87 % likely".
4. **Small-data statistics.** With tens of images per negative denomination, an observed 0 false positives still allows
   a true rate of tens of percent (see the Wilson intervals in the reports). Rates are only meaningful with the denominators shown.
5. **Distribution shift.** Different cameras, rooms, note wear or lighting than your sessions can degrade results; the
   reviewer's webcam and notes are different from the developer's.
6. **CPU speed.** On the 2-core development laptop the forward pass is ≈ 62 ms at 480 px (≈ 115 ms at 640), so live FPS is
   low-teens at best; "real-time" is not claimed beyond what the HUD measures on the viewing machine.
7. **Scenes with several notes.** Multiple or overlapping ₹50 notes in one frame were not studied.
8. **Platform coverage.** Tested on Windows 10 / Python 3.14 only. Linux/macOS paths and fonts are handled but untested.
9. **Temporal filter is a display aid.** It adds latency and is not counted in any metric.
10. **Verified-with-toys only:** export parity, error handling and tooling were verified with random-init and
    synthetic-rectangle models; passing those tests does not imply good detection on banknotes.
11. **Licence:** training/export depend on AGPL-3.0 Ultralytics.
12. **Webcam disconnect** is covered by simulated-camera tests; a physical unplug was not performed.
