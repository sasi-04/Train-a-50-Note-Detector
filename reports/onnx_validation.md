# ONNX validation: PASS

Model: `model.onnx`

| check | status | detail |
|---|---|---|
| file exists | PASS | 12.7 MB, sha256 ce5a6d1ce637 |
| ONNX graph valid | PASS | inputs [('images', [1, 3, 480, 480], 1)], outputs [('output0', [1, 5, 4725])], opset [12] |
| input is float32 NCHW 3-channel | PASS | [1, 3, 480, 480] elem_type=1 |
| loads in OpenCV (cv2.dnn) | PASS | OpenCV 5.0.0, init 259 ms |
| output layout recognised | PASS | shape (1, 5, 4725) -> channels_first |
| single-class head (nc == 1) | PASS | 1 class channel(s) |
| test inputs | PASS | 12 images from dataset\images\pool |
| inference runs, output finite | PASS | 12 images |
| class scores within [0,1] | PASS | min 0.0000 max 0.9416 |
| OpenCV vs ONNX Runtime raw tensors | PASS | max box diff 0.0004883 px, max score diff 7.15e-07 |
| PyTorch vs OpenCV raw tensors | PASS | max box diff 0.0009155 px, max score diff 9.83e-07 |
| PyTorch vs OpenCV decoded detections | PASS | 18/18 top detections matched at IoU>=0.9 (100.0%) |
| latency measured | PASS | median 119.9 ms/frame forward pass over 10 runs (this machine) |
