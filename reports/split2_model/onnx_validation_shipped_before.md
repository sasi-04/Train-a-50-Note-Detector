# ONNX validation: PASS

Model: `model.onnx`

| check | status | detail |
|---|---|---|
| file exists | PASS | 12.7 MB, sha256 ee2a0455e060 |
| ONNX graph valid | PASS | inputs [('images', [1, 3, 480, 480], 1)], outputs [('output0', [1, 6, 4725])], opset [12] |
| input is float32 NCHW 3-channel | PASS | [1, 3, 480, 480] elem_type=1 |
| loads in OpenCV (cv2.dnn) | PASS | OpenCV 5.0.0, init 244 ms |
| output layout recognised | PASS | shape (1, 6, 4725) -> channels_first |
| class head is Rs.50 (+ optional 'other note') | PASS | 2 class channel(s) |
| test inputs | PASS | 12 images from dataset\images\pool |
| inference runs, output finite | PASS | 12 images |
| class scores within [0,1] | PASS | min 0.0000 max 0.8859 |
| OpenCV vs ONNX Runtime raw tensors | PASS | max box diff 0.0007324 px, max score diff 3.34e-06 |
| PyTorch vs OpenCV raw tensors | PASS | max box diff 0.001007 px, max score diff 2.09e-06 |
| PyTorch vs OpenCV decoded detections | PASS | 55/55 top detections matched at IoU>=0.9 (100.0%) |
| latency measured | PASS | median 114.2 ms/frame forward pass over 10 runs (this machine) |
