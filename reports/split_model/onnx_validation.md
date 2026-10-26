# ONNX validation: PASS

Model: `runs\rs50_split\model_split.onnx`

| check | status | detail |
|---|---|---|
| file exists | PASS | 12.7 MB, sha256 983d1e3221d7 |
| ONNX graph valid | PASS | inputs [('images', [1, 3, 480, 480], 1)], outputs [('output0', [1, 5, 4725])], opset [12] |
| input is float32 NCHW 3-channel | PASS | [1, 3, 480, 480] elem_type=1 |
| loads in OpenCV (cv2.dnn) | PASS | OpenCV 5.0.0, init 189 ms |
| output layout recognised | PASS | shape (1, 5, 4725) -> channels_first |
| single-class head (nc == 1) | PASS | 1 class channel(s) |
| test inputs | PASS | 12 images from dataset\images\pool |
| inference runs, output finite | PASS | 12 images |
| class scores within [0,1] | PASS | min 0.0000 max 0.8338 |
| OpenCV vs ONNX Runtime raw tensors | PASS | max box diff 0.0005493 px, max score diff 8.94e-07 |
| PyTorch vs OpenCV raw tensors | PASS | max box diff 0.000885 px, max score diff 8.64e-07 |
| PyTorch vs OpenCV decoded detections | PASS | 238/238 top detections matched at IoU>=0.9 (100.0%) |
| latency measured | PASS | median 132.7 ms/frame forward pass over 10 runs (this machine) |
