import argparse
import numpy as np
from pathlib import Path

import onnx_graphsurgeon as gs
import onnx

def create_attrs(input_h, input_w, topK, keepTopK):
    attrs = {}
    attrs["shareLocation"] = 1
    attrs["backgroundLabelId"] = -1
    attrs["numClasses"] = 80
    attrs["topK"] = topK
    attrs["keepTopK"] = keepTopK
    attrs["scoreThreshold"] = 0.25
    attrs["iouThreshold"] = 0.6
    attrs["isNormalized"] = False
    attrs["clipBoxes"] = False

    # 001 is the default plugin version the parser will search for, and therefore can be omitted,
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs

def create_and_add_plugin_node(graph):
    batch_size = graph.inputs[0].shape[0]
    input_h = graph.inputs[0].shape[2]
    input_w = graph.inputs[0].shape[3]
    tensors = graph.tensors()

    boxes_tensor = tensors["bbox_out"] # match with onnx model output name
    confs_tensor = tensors["class_out"] # match with onnx model output name
    topK = 100
    keepTopK = 50

    num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1]) # do not change
    nmsed_boxes = gs.Variable(name="nmsed_boxes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK, 4])  # do not change
    nmsed_scores = gs.Variable(name="nmsed_scores").to_variable(dtype=np.float32, shape=[batch_size, keepTopK])  # do not change
    nmsed_classes = gs.Variable(name="nmsed_classes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK])  # do not change
    new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]  # do not change

    mns_node = gs.Node( # define nms plugin
        op="BatchedNMSDynamic_TRT", # match with batchedNMSPlugn
        attrs=create_attrs(input_h, input_w, topK, keepTopK), # set attributes for nms plugin
        inputs=[boxes_tensor, confs_tensor],
        outputs=new_outputs)

    graph.nodes.append(mns_node) # nms plugin added 
    graph.outputs = new_outputs

    return graph.cleanup().toposort()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='yolov7.onnx', help='onnx model path')

    opt = parser.parse_args()
    
    output_model_path = f"{opt.onnx_path[:-5]}_gs.onnx" # yolov7.onnx -> yolov7_gs.onnx
    graph = gs.import_onnx(onnx.load(opt.onnx_path)) # load onnx model

    graph = create_and_add_plugin_node(graph)
    onnx.save(gs.export_onnx(graph), output_model_path) # save model
    print(f'ONNX GraphSurgeon: Add NMS plugin into onnx model, saved as {output_model_path}')
