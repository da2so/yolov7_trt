# YOLOv7 TRT Plugin

This supplies TRT model (with batchedNMSPlugin) for yolov7 model. 

# Use

## 1. Set up

- onnx: 1.8.0
- torch: 1.9.0a0+df837d0
- onnx-graphsurgeon: 0.2.8
- tensorrt: 7.2.2.3
- CUDA: 11.2
- Driver Version: 460.73.01


## 2. Torch to ONNX


```bash
python export.py --weights yolov7.pt
```


## 3. Add batchedNMSPlugin into ONNX model using onnx_graphsurgeon
```bash
python add_nmsplugin.py --onnx_path yolov7.onnx
```
## 4. ONNX to TRT
```bash
trtexec --onnx=yolov7_gs.onnx --fp16 --workspace --saveEngine=yolov7_gs.trt
```

## 5. Inference

```bash
python detect_trt_plugin.py --model yolov7_gs.trt --image_folder inference/images --classes coco.yaml
```