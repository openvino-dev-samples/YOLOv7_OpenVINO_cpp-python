import yolov7
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-i', '--input', required=True, type=int,
                      help='Required. Webcam ID.')
    parser.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    parser.add_argument('-d', '--device', required=False, default='CPU', type=str,
                      help='Device name.')
    parser.add_argument('-p', '--pre_api', required=False, action='store_true', 
                      help='Preprocessing api.')
    parser.add_argument('-bs', '--batchsize', required=False, default=1, type=int,
                      help='Batch size.')
    parser.add_argument('-n', '--nireq', required=False, default=2, type=int,
                      help='number of infer request.')
    
    args = parser.parse_args()
    yolov7_detector=yolov7.YOLOV7_OPENVINO(args.model, args.device, args.pre_api, args.batchsize, args.nireq)
    yolov7_detector.infer_cam(args.input)
