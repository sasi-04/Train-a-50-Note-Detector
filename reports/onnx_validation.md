# ONNX validation: PASS

Model: `runs\rs50_final3\model_final3.onnx`

| check | status | detail |
|---|---|---|
| file exists | PASS | 12.7 MB, sha256 93a630e81cfb |
| ONNX graph valid | PASS | inputs [('images', [1, 3, 480, 480], 1)], outputs [('output0', [1, 6, 4725])], opset [12] |
| input is float32 NCHW 3-channel | PASS | [1, 3, 480, 480] elem_type=1 |
| loads in OpenCV (cv2.dnn) | PASS | OpenCV 5.0.0, init 237 ms |
| output layout recognised | PASS | shape (1, 6, 4725) -> channels_first |
| class head is Rs.50 (+ optional 'other note') | PASS | 2 class channel(s) |
| test inputs | PASS | synthetic (numerics only) |
| inference runs, output finite | PASS | 6 images |
| class scores within [0,1] | PASS | min 0.0000 max 0.8098 |
| OpenCV vs ONNX Runtime raw tensors | PASS | max box diff 0.001556 px, max score diff 1.67e-06 |
| PyTorch vs OpenCV raw tensors | PASS | max box diff 0.001892 px, max score diff 3.76e-06 |
| PyTorch vs OpenCV decoded detections | PASS | 66/66 top detections matched at IoU>=0.9 (100.0%) |
| latency measured | PASS | median 113.7 ms/frame forward pass over 10 runs (this machine) |
