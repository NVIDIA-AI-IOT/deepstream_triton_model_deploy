# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from PIL import Image
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.compat.v1 as tf1

tf1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

DEFAULT_FROZEN_GRAPH_NAME = "frozen_inference_graph.pb"
DEFAULT_MAX_BATCHSIZE = 1
DEFAULT_INPUT_NAME = "image_tensor"
DEFAULT_BOXES_NAME = "detection_boxes"
DEFAULT_CLASSES_NAME = "detection_classes"
DEFAULT_SCORES_NAME = "detection_scores"
DEFAULT_NUM_DETECTIONS_NAME = "num_detections"
DEFAULT_PRECISION = "FP32"
DEFAULT_NMS = False
# Default workspace size : 512MB
DEFAULT_MAX_WORKSPACE_SIZE = 1 << 29
DEFAULT_MIN_SEGMENT_SIZE = 10
DEFAULT_GPU_MEMORY_FRACTION = 0.6

TfConfig = tf.ConfigProto()
# TfConfig.gpu_options.allow_growth=True
TfConfig.gpu_options.allow_growth = False
TfConfig.gpu_options.per_process_gpu_memory_fraction = DEFAULT_GPU_MEMORY_FRACTION


def loadGraphDef(modelFile):
    graphDef = tf.GraphDef()
    with open(modelFile, "rb") as f:
        graphDef.ParseFromString(f.read())
    return graphDef


def saveGraphDef(graphDef, outputFilePath):
    with open(outputFilePath, "wb") as f:
        f.write(graphDef.SerializeToString())
        print("---------saved graphdef to {}".format(outputFilePath))


def updateNmsCpu(graphDef):
    for node in graphDef.node:
        # if 'NonMaxSuppressionV' in node.name and not node.device:
        if "NonMaxSuppression" in node.name and "TRTEngineOp" not in node.name:
            # node.device = '/device:CPU:0'
            node.device = "/job:localhost/replica:0/task:0/device:CPU:0"


def main():

    parser = argparse.ArgumentParser(description="Offline tf-trt GraphDef")
    parser.add_argument(
        "--modelPath",
        type=str,
        default=DEFAULT_FROZEN_GRAPH_NAME,
        help="path to frozen model",
        required=True,
    )
    parser.add_argument(
        "--gpu_mem_fraction",
        type=float,
        default=DEFAULT_GPU_MEMORY_FRACTION,
        help="Tensorflow gpu memory fraction, suggested value [0.2, 0.6]",
    )
    parser.add_argument(
        "--nms", type=bool, default=DEFAULT_NMS, help="to offload NMS operation to CPU"
    ),
    parser.add_argument(
        "--precision", type=str, default=DEFAULT_PRECISION, help="Precision mode to use"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=DEFAULT_MAX_BATCHSIZE,
        help="Specify max batch size",
    )
    parser.add_argument(
        "--save_graph", type=str, default=None, help="TF-TRT optimized model file"
    )
    parser.add_argument(
        "--min_segment_size",
        type=int,
        default=DEFAULT_MIN_SEGMENT_SIZE,
        help="the minimum number of nodes required for a subgraph to be replaced by TRTEngineOp",
    )
    args = parser.parse_args()
    saveGraphPath = args.save_graph
    if not saveGraphPath:
        saveGraphPath = (
            "frozen_tfrtr_"
            + args.precision.lower()
            + "_bs"
            + str(args.max_batch_size)
            + "_mss"
            + str(args.min_segment_size)
            + ".pb"
        )
    TfConfig.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem_fraction
    outputNames = [
        DEFAULT_BOXES_NAME,
        DEFAULT_CLASSES_NAME,
        DEFAULT_SCORES_NAME,
        DEFAULT_NUM_DETECTIONS_NAME,
    ]
    nnGraphDef = loadGraphDef(args.modelPath)
    converter = trt.TrtGraphConverter(
        is_dynamic_op=True,
        input_graph_def=nnGraphDef,
        nodes_blacklist=outputNames,
        max_batch_size=args.max_batch_size,
        max_workspace_size_bytes=DEFAULT_MAX_WORKSPACE_SIZE,
        precision_mode=args.precision,
        minimum_segment_size=args.min_segment_size,
    )
    trtGraphDef = converter.convert()
    print("-------tf-trt model has been rebuilt.")
    if args.nms == True:
        # Update NMS to CPU and save the model
        print("-------updateNMS to CPU.")
        updateNmsCpu(trtGraphDef)
        saveGraphPath = "nms_" + saveGraphPath
    saveGraphDef(trtGraphDef, saveGraphPath)


if __name__ == "__main__":
    main()
