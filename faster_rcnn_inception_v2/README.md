# TensorFlow FasterRCNN Inception V2 Model with Deepstream #

We are using Deepstream-5.0 with Triton Inference Server to deploy the FasterRCNN with Inception V2 model trained on the MSCOCO dataset for object detection. 

### Prerequisites: ###

[DeepStream SDK 5.0](https://developer.nvidia.com/deepstream-sdk)

Download and install DeepStream SDK or use DeepStream docker image (nvcr.io/nvidia/deepstream:5.0-20.07-triton) for x86 and (nvcr.io/nvidia/deepstream-l4t:5.0-20.07-samples) for NVIDIA Jetson.

Follow the instructions mentioned in the quick start guide: (https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide/deepstream_quick_start.html)

### Obtaining the model ###

```bash
$wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
$tar xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

### Optimizing the model with TF-TRT ###

```
$docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3
$docker pull nvcr.io/nvidia/l4t-tensorflow:r32.4.3-tf1.15-py3
$docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  -v /home/$USER/triton_blog/:/workspace/triton_blog nvcr.io/nvidia/tensorflow:20.03-tf1-py3
$docker run --runtime=nvidia -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  -v /home/$USER/triton_blog/:/workspace/triton_blog nvcr.io/nvidia/l4t-tensorflow:r32.4.3-tf1.15-py3
$python3 export_nms_only.py --modelPath faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --gpu_mem_fraction 0.6 --nms True --precision FP16 --max_batch_size 8 --min_segment_size 5
```

### Deepstream Configuration Files ###

There are two configuration file:
1. Inference configuration file
	* Sets the parameters for inference. This file takes the model configuration file sets the parameters for pre/post-processing
2. Application configuration file
	* Sets the configuration group to create a DeepStream pipeline. In this file you can set different configuration groups like source, sink, primary-gie, osd etc. Each group is calling a gstreamer-plugin. For more information on these plugins and configuration please check (https://docs.nvidia.com/metropolis/deepstream/plugin-manual/index.html#page/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_details.html) (https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)

These files are located at faster_rcnn_inception_v2/config

### Run the Application ###

To run the application, make sure that the paths to the configuration files and input video stream are correct, then launch the reference app with the application configuration file

`cd $DEEPSTREAM_DIR/samples/configs/deepstream-app-trtis`
`deepstream-app -c source1_primary_faster_rcnn_inception_v2.txt`

## Performance ##

Performance across 4 1080p streams with FP16 and TF-TRT optimizations

| Model                      | WxH       | Perf  | Hardware         | # Streams | # Batch size |
|----------------------------|-----------|-------|------------------|-----------|--------------|
| TF FasterRCNN Inception V2 | 1920x1080 | 32.36 | NVIDIA T4        | 4         | 4            |
| TF FasterRCNN Inception V2 | 1920x1080 | 14.92 | NVIDIA Jetson NX | 4         | 4            |

<p align="left">
  <img src="faster_rcnn_output.png">
</p>