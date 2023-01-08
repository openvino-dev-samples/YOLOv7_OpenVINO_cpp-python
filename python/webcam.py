import yolov7
import argparse
import webapp_utils



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-i', '--input', required=True, type=int,
                      help='Required. Webcam ID.')
    parser.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    parser.add_argument('-d', '--device', required=False, default='CPU', type=str,
                      help='Device name.')
    parser.add_argument('-p', '--pre_api', required=False, default=True, type=bool,
                      help='Device name.')
    parser.add_argument('-bs', '--batchsize', required=False, default=1, type=int,
                      help='Batch size.')
    parser.add_argument('-n', '--nireq', required=False, default=2, type=int,
                      help='number of infer request.')
    parser.add_argument('-g', '--grid', required=False, action='store_true', 
                      help='With grid in model.')
    parser.add_argument('-c', '--conf', required=False, default=.5, type=float,
                      help='infer confidence')
        
    parser.add_argument('--use-flask', default=False, action='store_true')
    parser.add_argument('--no-flask', dest='use-flask', action='store_false')
    
    args = parser.parse_args()
    yolov7_detector=yolov7.YOLOV7_OPENVINO(args.model, args.device, args.pre_api, 
                                           args.batchsize, args.nireq, args.grid, 
                                           args.conf, args.use_flask)
    yolov7_detector.infer_cam(args.input)
