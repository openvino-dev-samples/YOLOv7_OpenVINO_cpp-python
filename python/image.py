import yolov7
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path to an image file.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-d', '--device', required=False, default='CPU', type=str,
                      help='Device name.')
    args.add_argument('-p', '--pre_api', required=False, default=True, type=bool,
                      help='Device name.')
    args = parser.parse_args()
    yolov7_detector=yolov7.YOLOV7_OPENVINO(args.model, args.device, args.pre_api, 1)
    yolov7_detector.infer_image(args.input)
