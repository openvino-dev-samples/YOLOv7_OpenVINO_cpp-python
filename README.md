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
  $ python python/image.py -m yolov7.onnx -i data/horse.jpg
 ```

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


## 6. Run with webcam
 You can also run the sample with webcam for real-time detection
  ```shell
  $ python python/webcam.py -m yolov7.onnx -i 0
 ```
 
Tips: you can switch the device name to **"GPU"** to boost the performance.

## 7. Further optimization
Try this notebook ([yolov7-optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/develop/notebooks/226-yolov7-optimization)) and quantize your YOLOv7 model to INT8.


## 8. Run IoT web app to count people with yolov7-tiny model
This is a feature to run `webcam.py` to render computer vision results in the browser. 
The idea for web browser access is for remote deployments purposes and for convenience purposes use a browser to monitor results if the remote computer can be accessed via VPN or port forwarding. 
The web app also contains a rest endpoint for people and video frame rate which can be logged on an external source requesting locally data via a GET requests to the web app.

* To run IoT app `cd` into `python` and use an additional `arg` to use flask
* `$ python webcam.py -i 0 -m ./models/yolov7-tiny.onnx --use-flask`
* Open browser dial into `localhost:5000` or the IP of the computer on port 5000 after setting up appropriate firewall rules
* People count rest API GET request: `http://localhost:5000/people-count/` to log data from external IoT platform
* FPS rest API GET request: `http://localhost:5000/fps/` to log data from external IoT platform
* Future testing to include MQTT if desired

## 0. Remote Deployment IoT Results Viewed In Browser
 
 ![iot_res](/data/iot_screenshot.PNG)







