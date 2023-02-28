# YOLOv7_OpenVINO
This repository will demostrate how to deploy a offical YOLOv7 pre-trained model with OpenVINO runtime api
## 1. Install requirements
### ***Python***
```shell
  $ pip install -r python/requirements.txt
 ```

### ***C++*** (Ubuntu)
Please follow the Guides to install [OpenVINO](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html) and [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

## 2. Prepare the model
Download YOLOv7 pre-trained weight from [YOLOv7](https://github.com/WongKinYiu/yolov7)

## 3. Export the ONNX model
```shell
  $ git clone git@github.com:WongKinYiu/yolov7.git
  $ cd yolov7/models
  $ python export.py --weights yolov7.pt
 ```
 
## 4. Run inference
 The input image can be found in [YOLOv7's repository](https://github.com/WongKinYiu/yolov7/blob/main/inference/images/horses.jpg)
 ### ***Python***
 ```shell
  $ python python/image.py -m yolov7.onnx -i data/horse.jpg -d "CPU" -p False
 ```
 You can also try running the code with [Preprocessing API](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Preprocessing_Overview.html) for performance optimization.
 ```shell
  $ python python/image.py -m yolov7.onnx -i data/horse.jpg -p
 ```

- -i = path to image or video source;
- -m = Path to IR .xml or .onnx file;
- -d = Device name, e.g "CPU";
- -p = with/without preprocessing api
- -bs = Batch size;
- -n = number of infer requests;
- -g = with/without grid in model
  
 ### ***C++*** (Ubuntu)
Compile the source code
```shell
  $ cd cpp
  $ mkdir build && cd build
  $ source '~/intel/openvino_2022.1.0.643/bin/setupvars.sh'
  $ cmake ..
  $ make
 ```
You can also uncomment the code in ```CMakeLists.txt``` to trigger [Preprocessing API](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Preprocessing_Overview.html) for performance optimization.

Run inference
 ```shell
  $ yolov7 yolov7.onnx data/horses.jpg 'CPU'
 ```
## 5. Results
 
 ![horse_res](https://user-images.githubusercontent.com/91237924/179361905-44fcd4ac-7a9e-41f0-bd07-b6cf07245c04.jpg)

## Inference on multiple images at once
```shell
  $ python python/image.py -m weights/yolov7.onnx -i data/images/  -O data/output/
```

 ## 6. Run with webcam
 You can also run the sample with webcam for real-time detection
  ```shell
  $ python python/webcam.py -m yolov7.onnx -i 0
 ```
 
Tips: you can switch the device name to **"GPU"** to improve the performance.

## 7. Further optimization
Try this notebook ([yolov7-optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/develop/notebooks/226-yolov7-optimization)) and quantize your YOLOv7 model to INT8.
