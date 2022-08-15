# YOLOv7_OpenVINO
This repository will demostrate how to deploy a offical YOLOv7 pre-trained model with OpenVINO runtime api
## 1 Intall Requirements
### ***Python***
```shell
  pip install -r requirements
 ```

### ***C++*** (Ubuntu)
Pleas follow the Guides to install [OpenVINO](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino) and [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

## 2 Download YOLOv7 pre-trained weight from [YOLOv7](https://github.com/WongKinYiu/yolov7)

## 3 Export the ONNX model
```shell
  git clone git@github.com:WongKinYiu/yolov7.git
  cd yolov7/models
  python export.py --weights yolov7.pt
 ```
 
## 4 Run inference
 the input image can be found in [YOLOv7's repository](https://github.com/WongKinYiu/yolov7/blob/main/inference/images/horses.jpg)
 ### ***Python***
 ```shell
  python python/main.py -m yolov7.onnx -i data/horse.jpg
 ```

 ### ***C++*** (Ubuntu)
Compile the source code
```shell
  cd cpp
  mkdir build && cd build
  cmake ..
  make
 ```
 Run inference
 ```shell
  yolov7 yolov7.onnx data/horses.jpg 'CPU'
 ```
## 5 Results
 
 ![horse_res](https://user-images.githubusercontent.com/91237924/179361905-44fcd4ac-7a9e-41f0-bd07-b6cf07245c04.jpg)
