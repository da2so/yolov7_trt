trtexec --onnx=yolov7.onnx --saveEngine=yolov7.trt --fp16 --workspace=1024
trtexec --onnx=yolov7_gs.onnx --saveEngine=yolov7_gs.trt --fp16 --workspace=1024
