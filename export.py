import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def export(save_img=False):
    weights, imgsz, trace = opt.weights, opt.img_size, not opt.no_trace

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    img = torch.zeros(1, 3, imgsz, imgsz).to(device)  # image size(1,3,320,192) iDetection

    
    if half:
        model = model.half()  # to FP16
        img = img.half()
    model.eval()
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    class ProcModel(nn.Module):
        def __init__(self, model, class_num):
            super(ProcModel, self).__init__()
            self.model = model
            self.class_num = class_num
        def forward(self, x):

            out = self.model(x)[0] # out shape = [batch, num_object, 85], 85 = class_num(80)+bbox(4)+confidence(1)
            bbox_out = torch.unsqueeze(out[:,:,:4], 2) # bbox_out shape = [batch, num_object, 1, bbox], bbox = [cx,cy,w,h]

            x1 = bbox_out[:,:,:,0] - bbox_out[:,:,:,2] / 2
            y1 = bbox_out[:,:,:,1] - bbox_out[:,:,:,3] / 2
            x2 = bbox_out[:,:,:,0] + bbox_out[:,:,:,2] / 2
            y2 = bbox_out[:,:,:,1] + bbox_out[:,:,:,3] / 2
            bbox_out = torch.stack((x1,y1,x2,y2), dim=3) # bbox_out shape = [batch, num_object, 1, bbox], bbox = [x1,y1,x2,y2]

            conf_out = out[:,:,4] # [batch, num_object, 1]

            conf_out = torch.reshape(conf_out, (conf_out.shape[1],)) # [batch, num_object]
            class_out = torch.mul(out[:,:,5:].transpose(1,2) , conf_out).transpose(1,2) # [batch, num_object, num_classes] 
            return [bbox_out, class_out]

    procmodel = ProcModel(model, 80)
    
    import onnx

    print(f'ONNX: starting export with onnx {onnx.__version__}...')
    f = str(weights).replace('.pt', '.onnx')  #  yolov7.pt -> yolov7.onnnx
    input_names = ['images']
    output_names = ['bbox_out','class_out']
    train = False
    opset_version = 12

    torch.onnx.export(procmodel, img, f, verbose=False, opset_version=opset_version,
                      training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=not train,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    print(f'ONNX: export success, saved as {f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                export()
                strip_optimizer(opt.weights)
        else:
            export()
