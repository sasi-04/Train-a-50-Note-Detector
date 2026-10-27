# Limitations

1. **Narrow dataset.** 738 images (117 phone/scanner + 621 webcam frames) of **one physical copy of each note**, one person, a few
   rooms, no lighting/angle metadata; consecutive webcam frames are highly correlated, so the effective sample is much smaller.
   Held-out webcam test: 0/167 non-₹50 images labelled ₹50 (true rate up to ≈ 2 %), but only 23/36 ₹50 frames found (blurred, backlit
   or partly hidden ones are missed). The *shipped* model was fine-tuned on everything, so it has no held-out score. The threshold
   (0.46) comes from one validation session (29 target frames).
2. **Not an authentication system.** It localises a visual pattern. It cannot tell a genuine note from a good fake, and
   a high score is not evidence of authenticity.
3. **Scores are not probabilities.** No calibration has been measured; do not read 0.87 as "87 % likely".
4. **Small-data statistics.** With tens of images per negative denomination, an observed 0 false positives still allows
   a true rate of tens of percent (see the Wilson intervals in the reports). Rates are only meaningful with the denominators shown.
5. **Distribution shift.** Different cameras, rooms, note wear or lighting than your sessions can degrade results; the
   reviewer's webcam and notes are different from the developer's.
6. **CPU speed.** On the 2-core development laptop the shipped model runs ≈ 87 ms/frame end to end (≈ 11 FPS) at 480 px; faster CPUs
   will do better. "Real-time" is not claimed beyond what the HUD measures on the viewing machine.
13. **Box provenance.** 51 of the 156 ₹50 boxes came from colour-segmentation proposals that were visually reviewed; the other 105
    (webcam) were drawn by hand from grid views and verified by overlay. Some boxes are a few percent loose. Labelled frames come from
    near-duplicate bursts, so the effective number of distinct views is far smaller than the image count.
14. **Privacy.** Webcam and hand-held frames show the developer's face, a lanyard, and bystanders in the background.
7. **Scenes with several notes.** Multiple or overlapping ₹50 notes in one frame were not studied.
8. **Platform coverage.** Tested on Windows 10 / Python 3.14 only. Linux/macOS paths and fonts are handled but untested.
9. **Temporal filter is a display aid.** It adds latency and is not counted in any metric.
10. **Mechanics vs quality:** the unit/integration tests prove loading, decoding, geometry and error handling (some with toy models);
    they do not prove detection quality. Quality evidence is only the small held-out evaluation above.
11. **Licence:** training/export depend on AGPL-3.0 Ultralytics.
12. **Webcam disconnect** is covered by simulated-camera tests; a physical unplug was not performed.
