
import threading
import time
from pathlib import Path
import cv2

from flask import jsonify, request, make_response, render_template, Response


class EndpointHandler(object):
    def __init__(self, action):
        self.action = action 

    def __call__(self, *args, **kwargs):
        response = self.action(*args, **request.view_args)
        return make_response(response)

class FlaskAppWrapper(object):
    def __init__(self, app, **configs):
        self.app = app
        self.configs(**configs)

    def configs(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointHandler(handler), methods=methods, *args, **kwargs)

    def run(self, **kwargs):
        self.app.run(**kwargs)
        
        
class WebAppUtils:
    def __init__(self):
        self.framecopy = None
        self.net_people_count = None
        self.current_fps = None
    
    def favicon(self):
        return 'dummy', 200
    
    # used to render computer vision in browser
    def gen_frames(self):

        while True:
            if self.framecopy is None:
                continue

            ret, buffer = cv2.imencode('.jpg', self.framecopy)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def get_people(self):
        if self.net_people_count != None:
            return jsonify(self.net_people_count)
        else:
            return jsonify("server error"),500

    def get_fps(self):
        if self.current_fps != None:
            return jsonify(self.current_fps)
        else:
            return jsonify("server error"),500

    def video_feed(self):
        # Video streaming route. Put this in the src attribute of an img tag
        return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def index(self):
        # Video streaming home page
        return render_template('index.html')


class VideoPlayer:
    """
    Code is from the open_Vino notebooks 
    https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/115-async-api
    
    Custom video player to fulfill FPS requirements. You can set target FPS and output size,
    flip the video horizontally or skip first N frames.

    :param source: Video source. It could be either camera device or video file.
    :param size: Output frame size.
    :param flip: Flip source horizontally.
    :param fps: Target FPS.
    :param skip_first_frames: Skip first N frames.
    """

    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0):
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            raise RuntimeError(
                f"Cannot open {'camera' if isinstance(source, int) else ''} {source}"
            )
        # skip first N frames
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        # fps of input file
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        if self.__input_fps <= 0:
            self.__input_fps = 60
        # target fps given by user
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__flip = flip
        self.__size = None
        self.__interpolation = None
        if size is not None:
            self.__size = size
            # AREA better for shrinking, LINEAR better for enlarging
            self.__interpolation = (
                cv2.INTER_AREA
                if size[0] < self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                else cv2.INTER_LINEAR
            )
        # first frame
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False

    """
    Start playing.
    """

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    """
    Stop playing and release resources.
    """

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                break

            # fulfill target fps
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                # replace by current frame
                with self.__lock:
                    self.__frame = frame

            t2 = time.time()
            # time to wait [s] to fulfill input fps
            wait_time = 1 / self.__input_fps - (t2 - t1)
            # wait until
            time.sleep(max(0, wait_time))

        self.__frame = None

    """
    Get current frame.
    """

    def next(self):
        with self.__lock:
            if self.__frame is None:
                return None
            # need to copy frame, because can be cached and reused if fps is low
            frame = self.__frame.copy()
        if self.__size is not None:
            frame = cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip:
            frame = cv2.flip(frame, 1)
        return frame



class AsyncPipeline:
    def __init__(self, ie, model, plugin_config, device='CPU', max_num_requests=0):
        cache_path = Path("model_cache")
        cache_path.mkdir(exist_ok=True)
        # Enable model cachine for GPU devices
        if "GPU" in device and "GPU" in ie.available_devices:
            ie.set_config({"CACHE_DIR": str(cache_path)}, device_name="GPU")

        self.model = model
        self.logger = logging.getLogger()

        self.logger.info('Loading network to {} plugin...'.format(device))
        self.exec_net = ie.load_network(network=self.model.net, device_name=device,
                                        config=plugin_config, num_requests=max_num_requests)
        if max_num_requests == 0:
            # ExecutableNetwork doesn't allow creation of additional InferRequests. Reload ExecutableNetwork
            # +1 to use it as a buffer of the pipeline
            self.exec_net = ie.load_network(network=self.model.net, device_name=device,
                                            config=plugin_config, num_requests=len(self.exec_net.requests) + 1)

        self.empty_requests = deque(self.exec_net.requests)
        self.completed_request_results = {}
        self.callback_exceptions = {}
        self.event = threading.Event()

    def inference_completion_callback(self, status, callback_args):
        try:
            request, id, meta, preprocessing_meta = callback_args
            if status != 0:
                raise RuntimeError('Infer Request has returned status code {}'.format(status))
            raw_outputs = {key: blob.buffer for key, blob in request.output_blobs.items()}
            self.completed_request_results[id] = (raw_outputs, meta, preprocessing_meta)
            self.empty_requests.append(request)
        except Exception as e:
            self.callback_exceptions.append(e)
        self.event.set()

    def submit_data(self, inputs, id, meta):
        request = self.empty_requests.popleft()
        if len(self.empty_requests) == 0:
            self.event.clear()
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        request.set_completion_callback(py_callback=self.inference_completion_callback,
                                        py_data=(request, id, meta, preprocessing_meta))
        request.async_infer(inputs=inputs)

    def get_raw_result(self, id):
        if id in self.completed_request_results:
            return self.completed_request_results.pop(id)
        return None

    def get_result(self, id):
        result = self.get_raw_result(id)
        if result:
            raw_result, meta, preprocess_meta = result
            return self.model.postprocess(raw_result, preprocess_meta), meta
        return None

    def is_ready(self):
        return len(self.empty_requests) != 0

    def has_completed_request(self):
        return len(self.completed_request_results) != 0

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        if len(self.empty_requests) == 0:
            self.event.wait()
            
            
            
# used to render computer vision in browser
def gen_frames():
    global framecopy

    while True:
        if framecopy is None:
            continue

        ret, buffer = cv2.imencode('.jpg', framecopy)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


