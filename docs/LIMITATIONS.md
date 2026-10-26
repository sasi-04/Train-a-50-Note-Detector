# Limitations

1. **Tiny, narrow dataset.** 117 images from one physical note per denomination, essentially one room/person, no lighting or angle
   metadata, only 5 no-note frames and no other denominations held in hand. Held-out results were mixed (test: ₹50 back 6/6, front
   1/8, one ₹20 mistaken for ₹50) and the *shipped* model was retrained on everything, so it has no held-out score. The threshold
   (0.54) comes from a 13-target validation set with a narrow FP-free range (0.45–0.62).
2. **Not an authentication system.** It localises a visual pattern. It cannot tell a genuine note from a good fake, and
   a high score is not evidence of authenticity.
3. **Scores are not probabilities.** No calibration has been measured; do not read 0.87 as "87 % likely".
4. **Small-data statistics.** With tens of images per negative denomination, an observed 0 false positives still allows
   a true rate of tens of percent (see the Wilson intervals in the reports). Rates are only meaningful with the denominators shown.
5. **Distribution shift.** Different cameras, rooms, note wear or lighting than your sessions can degrade results; the
   reviewer's webcam and notes are different from the developer's.
6. **CPU speed.** On the 2-core development laptop the shipped model runs ≈ 91 ms/frame end to end (≈ 11 FPS) at 480 px; faster CPUs
   will do better. "Real-time" is not claimed beyond what the HUD measures on the viewing machine.
13. **Boxes were proposed, not hand-drawn.** The 51 target boxes came from colour/segmentation proposals that were visually reviewed
    (some are a few percent loose). Labelled frames come from near-duplicate bursts, so the effective number of distinct views is
    far smaller than the image count.
14. **Privacy.** Hand-held frames show the developer's face and a lanyard; wide shots may show part of an ID card.
7. **Scenes with several notes.** Multiple or overlapping ₹50 notes in one frame were not studied.
8. **Platform coverage.** Tested on Windows 10 / Python 3.14 only. Linux/macOS paths and fonts are handled but untested.
9. **Temporal filter is a display aid.** It adds latency and is not counted in any metric.
10. **Mechanics vs quality:** the unit/integration tests prove loading, decoding, geometry and error handling (some with toy models);
    they do not prove detection quality. Quality evidence is only the small held-out evaluation above.
11. **Licence:** training/export depend on AGPL-3.0 Ultralytics.
12. **Webcam disconnect** is covered by simulated-camera tests; a physical unplug was not performed.
