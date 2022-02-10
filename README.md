# Yolov4 - TensorRT | Fps Arttırma

Along the same line as Demo #3, these 2 demos showcase how to convert pre-trained yolov3 and yolov4 models through ONNX to TensorRT engines.  The code for these 2 demos has gone through some significant changes.  More specifically, I have recently updated the implementation with a "yolo_layer" plugin to speed up inference time of the yolov3/yolov4 models.

My current "yolo_layer" plugin implementation is based on TensorRT's [IPluginV2IOExt](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_i_o_ext.html).  It only works for **TensorRT 6+**.  I'm thinking about updating the code to support TensorRT 5 if I have time late on.

I developed my "yolo_layer" plugin by referencing similar plugin code by [wang-xinyu](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4) and [dongfangduoshou123](https://github.com/dongfangduoshou123/YoloV3-TensorRT/blob/master/seralizeEngineFromPythonAPI.py).  So big thanks to both of them.

Assuming this repository has been cloned at "${HOME}/project/tensorrt_demos", follow these steps:

1. Install "pycuda".

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/yolo
   $ ./install_pycuda.sh
   ```

2. Install **version "1.4.1" (not the latest version)** of python3 **"onnx"** module.  Note that the "onnx" module would depend on "protobuf" as stated in the [Prerequisite](#prerequisite) section.  Reference: [information provided by NVIDIA](https://devtalk.nvidia.com/default/topic/1052153/jetson-nano/tensorrt-backend-for-onnx-on-jetson-nano/post/5347666/#5347666).

   ```shell
   $ sudo pip3 install onnx==1.4.1
   ```

3. Go to the "plugins/" subdirectory and build the "yolo_layer" plugin.  When done, a "libyolo_layer.so" would be generated.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/plugins
   $ make
   ```

4. Download the pre-trained yolov3/yolov4 COCO models and convert the targeted model to ONNX and then to TensorRT engine.  I use "yolov4-416" as example below.  (Supported models: "yolov3-tiny-288", "yolov3-tiny-416", "yolov3-288", "yolov3-416", "yolov3-608", "yolov3-spp-288", "yolov3-spp-416", "yolov3-spp-608", "yolov4-tiny-288", "yolov4-tiny-416", "yolov4-288", "yolov4-416", "yolov4-608", "yolov4-csp-256", "yolov4-csp-512", "yolov4x-mish-320", "yolov4x-mish-640", and [custom models](https://jkjung-avt.github.io/trt-yolo-custom-updated/) such as "yolov4-416x256".)

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/yolo
   $ ./download_yolo.sh
   $ python3 yolo_to_onnx.py -m yolov4-416
   $ python3 onnx_to_tensorrt.py -m yolov4-416
   ```

   The last step ("onnx_to_tensorrt.py") takes a little bit more than half an hour to complete on my Jetson Nano DevKit.  When that is done, the optimized TensorRT engine would be saved as "yolov4-416.trt".

   In case "onnx_to_tensorrt.py" fails (process "Killed" by Linux kernel), it could likely be that the Jetson platform runs out of memory during conversion of the TensorRT engine.  This problem might be solved by adding a larger swap file to the system.  Reference: [Process killed in onnx_to_tensorrt.py Demo#5](https://github.com/jkjung-avt/tensorrt_demos/issues/344).

5. Test the TensorRT "yolov4-416" engine with the "dog.jpg" image.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg -O ${HOME}/Pictures/dog.jpg
   $ python3 trt_yolo.py --image ${HOME}/Pictures/dog.jpg \
                         -m yolov4-416
   ```

   This is a screenshot of the demo against JetPack-4.4, i.e. TensorRT 7.

   !["yolov4-416" detection result on dog.jpg](doc/dog_trt_yolov4_416.jpg)

6. The "trt_yolo.py" demo program could also take various image inputs.  Refer to step 5 in Demo #1 again.

   For example, I tested my own custom trained ["yolov4-crowdhuman-416x416"](https://github.com/jkjung-avt/yolov4_crowdhuman) TensorRT engine with the "Avengers: Infinity War" movie trailer:

   [![Testing with the Avengers: Infinity War trailer](https://raw.githubusercontent.com/jkjung-avt/yolov4_crowdhuman/master/doc/infinity_war.jpg)](https://youtu.be/7Qr_Fq18FgM)

7. (Optional) Test other models than "yolov4-416".

8. (Optional) If you would like to stream TensorRT YOLO detection output over the network and view the results on a remote host, check out my [trt_yolo_mjpeg.py example](https://github.com/jkjung-avt/tensorrt_demos/issues/226).

9. Similar to step 5 of Demo #3, I created an "eval_yolo.py" for evaluating mAP of the TensorRT yolov3/yolov4 engines.  Refer to [README_mAP.md](README_mAP.md) for details.

   ```shell
   $ python3 eval_yolo.py -m yolov3-tiny-288
   $ python3 eval_yolo.py -m yolov4-tiny-416
   ......
   $ python3 eval_yolo.py -m yolov4-608
   $ python3 eval_yolo.py -l -m yolov4-csp-256
   ......
   $ python3 eval_yolo.py -l -m yolov4x-mish-640
   ```

   I evaluated all these TensorRT yolov3/yolov4 engines with COCO "val2017" data and got the following results.  I also checked the FPS (frames per second) numbers on my Jetson Nano DevKit with JetPack-4.4 (TensorRT 7).

   | TensorRT engine         | mAP @<br>IoU=0.5:0.95 |  mAP @<br>IoU=0.5  | FPS on Nano |
   |:------------------------|:---------------------:|:------------------:|:-----------:|
   | yolov3-tiny-288 (FP16)  |         0.077         |        0.158       |     35.8    |
   | yolov3-tiny-416 (FP16)  |         0.096         |        0.202       |     25.5    |
   | yolov3-288 (FP16)       |         0.331         |        0.601       |     8.16    |
   | yolov3-416 (FP16)       |         0.373         |        0.664       |     4.93    |
   | yolov3-608 (FP16)       |         0.376         |        0.665       |     2.53    |
   | yolov3-spp-288 (FP16)   |         0.339         |        0.594       |     8.16    |
   | yolov3-spp-416 (FP16)   |         0.391         |        0.664       |     4.82    |
   | yolov3-spp-608 (FP16)   |         0.410         |        0.685       |     2.49    |
   | yolov4-tiny-288 (FP16)  |         0.179         |        0.344       |     36.6    |
   | yolov4-tiny-416 (FP16)  |         0.196         |        0.387       |     25.5    |
   | yolov4-288 (FP16)       |         0.376         |        0.591       |     7.93    |
   | yolov4-416 (FP16)       |         0.459         |        0.700       |     4.62    |
   | yolov4-608 (FP16)       |         0.488         |        0.736       |     2.35    |
   | yolov4-csp-256 (FP16)   |         0.336         |        0.502       |     12.8    |
   | yolov4-csp-512 (FP16)   |         0.436         |        0.630       |     4.26    |
   | yolov4x-mish-320 (FP16) |         0.400         |        0.581       |     4.79    |
   | yolov4x-mish-640 (FP16) |         0.470         |        0.668       |     1.46    |

10. Check out my blog posts for implementation details:

   * [TensorRT ONNX YOLOv3](https://jkjung-avt.github.io/tensorrt-yolov3/)
   * [TensorRT YOLOv4](https://jkjung-avt.github.io/tensorrt-yolov4/)
   * [Verifying mAP of TensorRT Optimized SSD and YOLOv3 Models](https://jkjung-avt.github.io/trt-detection-map/)
   * For training your own custom yolov4 model: [Custom YOLOv4 Model on Google Colab](https://jkjung-avt.github.io/colab-yolov4/)
   * For adapting the code to your own custom trained yolov3/yolov4 models: [TensorRT YOLO For Custom Trained Models (Updated)](https://jkjung-avt.github.io/trt-yolo-custom-updated/)

<a name="int8_and_dla"></a>
