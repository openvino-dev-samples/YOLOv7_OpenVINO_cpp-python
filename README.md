# YOLOv7_OpenVINO
This repository will demostrate how to dopley a offical YOLOv7 pre-trained model with OpenVINO runtime api

## 1 Download YOLOv7 pre-trained weight from [YOLOv7](https://github.com/WongKinYiu/yolov7)

## 2 Export the ONNX model
```shell
  git clone git@github.com:WongKinYiu/yolov7.git
  cd yolov7/models
  python export.py --weights yolov7.pt
 ```
 
 ## 3 Run inference
 the input image can be found in [YOLOv7's repository](https://github.com/WongKinYiu/yolov7/blob/main/inference/images/horses.jpg)
 ```shell
  python YOLOV7.py -m yolov7.onnx -i horse.jpg
 ```
 ## 4 Results
 
 ![horse_res](https://user-images.githubusercontent.com/91237924/179361905-44fcd4ac-7a9e-41f0-bd07-b6cf07245c04.jpg)
