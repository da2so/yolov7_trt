"""Runtime(TFLite, TensorRT) for NetsPresso model.
- Author: Jimin Lee, Junghoon Kim
- Contact: jimin.lee@nota.ai, junghoon.kim@nota.ai
"""
import cv2
import numpy as np
import time
import argparse
import yaml
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

# import lib for tensorrt
try:
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
except ImportError:
    print("Failed to load tensorrt, pycuda")
    trt = None
    cuda = None



class ModelWrapper():
    """Abstract class for model wrapper."""
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self._inputs = None
        self._outputs = None
        self._input_size = None

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, value: str):
        self._model_path = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

    @abstractmethod
    def load_model(self):
        """Set up model. please specify self.model """
        pass

    @abstractmethod
    def inference(self, input_images):
        """Run inference."""
        pass


class TRTWrapper(ModelWrapper):
    """TensorRT model wrapper."""
    def __init__(self, model_path, batch):
        super(TRTWrapper, self).__init__(model_path)
        self._batch = batch
        self._bindings = None

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value

    @property
    def bindings(self):
        return self._bindings

    @bindings.setter
    def bindings(self, value):
        self._bindings = value

    def load_model(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        trt.init_libnvinfer_plugins(None, "")
        with open(self.model_path, 'rb') as f:
            try:
                engine = runtime.deserialize_cuda_engine(f.read())
                self.model = engine
            except:
                sys.exit("Could not load model")
        
        self.alloc_buf()

    def inference(self, input_image):
        image = input_image.transpose(0, 3, 1, 2)
        self.inputs[0].cpu = image.ravel()

        with self.model.create_execution_context() as context:
            [cuda.memcpy_htod_async(inp.gpu, inp.cpu, self.stream) for inp in self.inputs]
            context.execute_async(self.batch, self.bindings, self.stream.handle, None)
            [cuda.memcpy_dtoh_async(out.cpu, out.gpu, self.stream) for out in self.outputs]
            self.stream.synchronize()

        num_detections = self.outputs[0].cpu # detection된 object개수
        nmsed_boxes = self.outputs[1].cpu # detection된 object coordinate
        nmsed_scores = self.outputs[2].cpu # detection된 object confidence
        nmsed_classes = self.outputs[3].cpu # detection된 object class number
        result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        return result

    def alloc_buf(self):
        inputs = []
        outputs = []
        bindings = []
        engine = self.model
        input_size = engine.get_binding_shape(0)[2:4] if engine.get_binding_shape(0)[1]==3 else input_shape[1:3]

        class HostDeviceMem(object):
            def __init__(self, cpu_mem, gpu_mem):
                self.cpu = cpu_mem
                self.gpu = gpu_mem

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            cpu_mem = cuda.pagelocked_empty(size, dtype)
            gpu_mem = cuda.mem_alloc(cpu_mem.nbytes)
            bindings.append(int(gpu_mem))

            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(cpu_mem, gpu_mem))
            else:
                outputs.append(HostDeviceMem(cpu_mem, gpu_mem))

        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.input_size = input_size
        self.stream = cuda.Stream()


def create_model_wrapper(extension: str, model_path: str, batch_size: int, device: str):
    """Create model wrapper class."""
    assert trt and cuda, f"Loading TensorRT, Pycuda lib failed."
    model_wrapper = TRTWrapper(model_path, batch_size)

    return model_wrapper

class Colors():
    """Color class."""
    def __init__(self):
        hex = ('B55151', 'FF3636', 'FF36A2', 'CB72A2', 'EC3AFF', '3B1CFF', '7261E1', '6991BF', '00B1BD', '00BD8B',
               '00DA33', 'BEEF4D', '8B8B8B', 'FFB300', '7F5903', '411C06', '795454', '495783', '624F70', '7A7D62')
        self.palette = [self.hextorgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, m, bgr=False):
        c = self.palette[int(m) % self.n]
        return (c[2], c[1], c[0])

    @staticmethod
    def hextorgb(hex):
        return tuple(int(hex[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class InferenceRunner():
    """Inference Runner."""
    def __init__(
        self,
        model_wrapper: ModelWrapper,
        img_folder: str,
        conf_thres: float,
        iou_thres: float,
        save_dir: str
    ):
        self.model = model_wrapper
        self.img_folder = img_folder
        self.save_dir = save_dir

    def run(self):
        for i, filename in enumerate(os.listdir(self.img_folder)):
            # save path
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / filename
            # load image
            img = cv2.imread(os.path.join(self.img_folder, filename))
            # if image load failed
            if img is None:
                continue
            preproc_image = self.preprocess_image(img)
            # inference
            inf_res = self.model.inference(preproc_image)
            self.print_result(preproc_image, inf_res, i, save_path)

    def preprocess_image(self, raw_bgr_image):
        """
        Description:
            Converting BGR image to RGB,
            Resizing and padding it to target size,
            Normalizing to [0, 1]
            Transforming to NCHW format
        Argument:
            raw_bgr_image: a numpy array from cv2 (BGR) (H, W, C)
        Return:
            preprocessed_image: preprocessed image (1, C, resized_H, resized_W)
            original_image: the original image (H, W, C)
            origin_h: height of the original image
            origin_w: width of the origianl image
        """

        input_size = self.model.input_size
        original_image = raw_bgr_image
        origin_h, origin_w, origin_c = original_image.shape
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # Calculate width and height and paddings
        r_w = input_size[1] / origin_w
        r_h = input_size[0] / origin_h
        if r_h > r_w:
            tw = input_size[1]
            th = int(r_w *  origin_h)
            tx1 = tx2 = 0
            ty1 = int((input_size[0] - th) / 2)
            ty2 = input_size[0] - th - ty1
        else:
            tw = int(r_h * origin_w)
            th = input_size[0]
            tx1 = int((input_size[1] - tw) / 2)
            tx2 = input_size[1] - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        preprocessed_image = np.ascontiguousarray(image)
        return preprocessed_image

    def print_result(self, result_image, result_label, count, save_path):
        num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = result_label
        
        colors = Colors()
        h, w = result_image.shape[1:3]
        result_image = np.squeeze(result_image)
        result_image *= 255
        result_image = result_image.astype(np.uint8)
        print("--------------------------------------------------------------")
        for i in range(int(num_detections)):
            detected = str(classes[int(nmsed_classes[i])]).replace('‘', '').replace('’', '')
            confidence_str = str(nmsed_scores[i])
            # unnormalize depending on the visualizing image size
            x1 = int(nmsed_boxes[0+i*4])
            y1 = int(nmsed_boxes[1+i*4])
            x2 = int(nmsed_boxes[2+i*4])
            y2 = int(nmsed_boxes[3+i*4])
            color = colors(int(nmsed_classes[i]), True)
            result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            text_size, _ = cv2.getTextSize(str(detected), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            result_image = cv2.rectangle(result_image, (x1, y1-5-text_h), (x1+text_w, y1), color, -1)
            result_image = cv2.putText(result_image, str(detected), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print("Detect " + str(i+1) + "(" + str(detected) + ")")
            print("Coordinates : [{:d}, {:d}, {:d}, {:d}]".format(x1, y1, x2, y2))
            print("Confidence : {:.7f}".format(nmsed_scores[i]))
            print("")
        print("--------------------------------------------------------------\n\n")
        cv2.imwrite(str(save_path), cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, default='yolov7_gs.trt', help='model path')
    parser.add_argument('--image_folder', required=True, default='inference/images', help='image path')
    parser.add_argument('--conf_thres', required=False, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', required=False, default=0.60, help='iou threshold')
    parser.add_argument('--batch', required=False, default=1, help='batch size')
    parser.add_argument('--device', required=False, default='CPU', help='CPU or GPU for openvino')
    parser.add_argument('--classes', required=True, default='coco.yaml', help='yaml file with class info')
    parser.add_argument('--save_dir', required=False, default='./result', help='save directory')

    args = parser.parse_args()
    
    # load class info(.yaml)
    with open(args.classes) as f:
        classes = yaml.safe_load(f)
        classes = classes['names']

    # load model 
    extension = os.path.splitext(args.model)[1]

    model_wrapper = create_model_wrapper(
        extension=extension,
        model_path=args.model,
        batch_size=args.batch,
        device=args.device
    )
    model_wrapper.load_model()

    # load image, inference, print result
    inference_runner = InferenceRunner(model_wrapper, args.image_folder, args.conf_thres, args.iou_thres, args.save_dir)
    inference_runner.run()


