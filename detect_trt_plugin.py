import cv2
import numpy as np
import time
import argparse
import yaml
import os
import sys
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



class TRTWrapper():
    """TensorRT model wrapper."""
    def __init__(self, model_path, batch):
        self.model_path = model_path
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
        runtime = trt.Runtime(TRT_LOGGER) # serialized ICudEngine을 deserialized하기 위한 클래스 객체
        trt.init_libnvinfer_plugins(None, "") # plugin 사용을 위함
        with open(self.model_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read()) # trt 모델을 읽어 serialized ICudEngine을 deserialized함
        
        self.context = self.engine.create_execution_context() # ICudEngine을 이용해 inference를 실행하기 위한 context class생        assert self.engine 
        assert self.context
        
        self.alloc_buf()

    def inference(self, input_image):
        image = input_image.transpose(0, 3, 1, 2) # NHWC to NWHC
        image = np.ascontiguousarray(image) 
        cuda.memcpy_htod(self.inputs[0]['allocation'], image) # input image array(host)를 GPU(device)로 보내주는 작업
        self.context.execute_v2(self.allocations) #inference 실행!
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation']) # GPU에서 작업한 값을 host로 보냄
        
        num_detections = self.outputs[0]['host_allocation'] # detection된 object개수
        nmsed_boxes = self.outputs[1]['host_allocation'] # detection된 object coordinate
        nmsed_scores = self.outputs[2]['host_allocation'] # detection된 object confidence
        nmsed_classes = self.outputs[3]['host_allocation'] # detection된 object class number
        result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        return result


    def alloc_buf(self):
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i): # i번째 binding이 input인지 확인
                is_input = True 
            name = self.engine.get_binding_name(i) # i번째 binding의 이름
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i))) # i번째 binding의 data type
            shape = self.context.get_binding_shape(i) # i번째 binding의 shape

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize # data type의 bit수
            for s in shape:
                size *= s # data type * 각 shape(e.g input의 경우 [1,3,640,640]) element 을 곱하여 size에 할당

            allocation = cuda.mem_alloc(size) # 해당 size만큼의 GPU memory allocation함
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i): # binding이 input이면
                self.inputs.append(binding)
            else: # 아니면 binding은 모두 output임
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))        

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs


def create_model_wrapper(model_path: str, batch_size: int, device: str):
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
        model_wrapper: TRTWrapper,
        img_folder: str,
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
        

        times = []
        iterations = 200
        for i in range(20):  # GPU warmup iterations
            self.model.inference(preproc_image)
        for i in range(iterations):
            start = time.time()
            self.model.inference(preproc_image)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(
            1000 * np.average(times)))
        print("Average Throughput: {:.1f} ips".format(
            1 / np.average(times)))
            
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

        input_size = self.model.input_spec()[0][-2:] # h,w = 640,640
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
            detected = str(classes[int(nmsed_classes[0][i])]).replace('‘', '').replace('’', '')
            confidence_str = str(nmsed_scores[0][i])
            # unnormalize depending on the visualizing image size
            x1 = int(nmsed_boxes[0][i][0])
            y1 = int(nmsed_boxes[0][i][1])
            x2 = int(nmsed_boxes[0][i][2])
            y2 = int(nmsed_boxes[0][i][3])
            color = colors(int(nmsed_classes[0][i]), True)
            result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            text_size, _ = cv2.getTextSize(str(detected), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            result_image = cv2.rectangle(result_image, (x1, y1-5-text_h), (x1+text_w, y1), color, -1)
            result_image = cv2.putText(result_image, str(detected), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print("Detect " + str(i+1) + "(" + str(detected) + ")")
            print("Coordinates : [{:d}, {:d}, {:d}, {:d}]".format(x1, y1, x2, y2))
            print("Confidence : {:.7f}".format(nmsed_scores[0][i]))
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
    model_wrapper = create_model_wrapper(
        model_path=args.model,
        batch_size=args.batch,
        device=args.device
    )
    model_wrapper.load_model()

    # load image, inference, print result
    inference_runner = InferenceRunner(model_wrapper, args.image_folder, args.save_dir)
    inference_runner.run()


