import threading
import time
from collections import deque

import cv2
import numpy as np
from .camera_stream import CameraStream


class CalibratedCamera:
    def __init__(self, source_id, calib_file_path=None, name="Camera", buffer_size=30):
        self.name = name
        self.source_id = source_id
        self.stream = CameraStream(source_id, name, buffer_size)
        self.mtx = None
        self.dist = None
        self.new_camera_matrix = None
        self.roi = None
        self.mapx = None
        self.mapy = None

        if calib_file_path:
            self._load_calibration(calib_file_path)

        self.current_frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.running = False
        self.process_thread = None

    def _load_calibration(self, calib_file_path):
        try:
            with np.load(calib_file_path) as data:
                self.mtx = data["mtx"]
                self.dist = data["dist"]
            print(f"[{self.name}] Calibration loaded successfully.")
        except Exception as e:
            print(
                f"[{self.name}] WARNING: Could not load calibration file '{calib_file_path}'. "
                f"Using raw image. Error: {e}"
            )

    def _init_undistort_maps(self, frame_shape):
        h, w = frame_shape[:2]
        if self.mtx is not None:
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h)
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.new_camera_matrix, (w, h), 5
            )

    def _apply_calibration(self, frame):
        if frame is None:
            return None

        if self.mapx is None:
            self._init_undistort_maps(frame.shape)

        if self.mapx is not None:
            frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            if self.roi is not None:
                x, y, w, h = self.roi
                if w > 0 and h > 0:
                    frame = frame[y : y + h, x : x + w]

        return frame

    def start(self):
        if self.running:
            return self

        self.stream.start()
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        return self

    def _process_loop(self):
        last_timestamp = None

        while self.running:
            frame, timestamp = self.stream.get_frame()

            if frame is None or timestamp is None or timestamp == last_timestamp:
                time.sleep(0.001)
                continue

            last_timestamp = timestamp

            if self.mtx is not None:
                frame = self._apply_calibration(frame)

            with self.lock:
                self.current_frame = frame
                self.timestamp = timestamp

            with self.buffer_lock:
                self.frame_buffer.append((timestamp, frame))

    def get_frame(self):
        with self.lock:
            return self.current_frame, self.timestamp

    def get_nearest_frame(self, target_timestamp):
        with self.buffer_lock:
            if not self.frame_buffer:
                return None, None, None

            nearest = min(self.frame_buffer, key=lambda x: abs(x[0] - target_timestamp))
            timestamp, frame = nearest
            time_diff = abs(timestamp - target_timestamp)

            return frame, timestamp, time_diff

    def read(self):
        ret, frame = self.stream.capture.read()
        timestamp = time.time() if ret else None
        if ret and self.mtx is not None:
            frame = self._apply_calibration(frame)
        return ret, frame, timestamp

    def stop(self):
        self.running = False
        if self.process_thread is not None:
            self.process_thread.join(timeout=1.0)
        self.stream.stop()

    def is_opened(self):
        return self.stream.is_opened()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def show_camera(
    source_id, calib_file_path=None, window_name="Camera", target_height=480
):
    cam = CalibratedCamera(source_id, calib_file_path, window_name)

    if not cam.is_opened():
        print(f"Error: Could not open camera {source_id}")
        return

    print(f"Showing camera '{window_name}'. Press 'q' to exit.")

    while True:
        ret, frame, _ = cam.read()
        if not ret:
            break

        if target_height:
            frame = resize_to_height(frame, target_height)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyWindow(window_name)


def resize_to_height(image, target_height):
    if image is None:
        return None
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))
