# ONNX validation: PASS

Model: `model.onnx`

| check | status | detail |
|---|---|---|
| file exists | PASS | 12.7 MB, sha256 c4cc29027008 |
| ONNX graph valid | PASS | inputs [('images', [1, 3, 480, 480], 1)], outputs [('output0', [1, 5, 4725])], opset [12] |
| input is float32 NCHW 3-channel | PASS | [1, 3, 480, 480] elem_type=1 |
| loads in OpenCV (cv2.dnn) | PASS | OpenCV 5.0.0, init 215 ms |
| output layout recognised | PASS | shape (1, 5, 4725) -> channels_first |
| single-class head (nc == 1) | PASS | 1 class channel(s) |
| test inputs | PASS | 12 images from dataset\images\pool |
| inference runs, output finite | PASS | 12 images |
| class scores within [0,1] | PASS | min 0.0000 max 0.8624 |
| OpenCV vs ONNX Runtime raw tensors | PASS | max box diff 0.0009766 px, max score diff 1.43e-06 |
| PyTorch vs OpenCV raw tensors | PASS | max box diff 0.0008698 px, max score diff 1.18e-06 |
| PyTorch vs OpenCV decoded detections | PASS | 107/107 top detections matched at IoU>=0.9 (100.0%) |
| latency measured | PASS | median 147.2 ms/frame forward pass over 10 runs (this machine) |
