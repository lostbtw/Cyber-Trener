import cv2
import threading
import time


class CameraStream:
    def __init__(self, source_id, name="Camera"):
        self.name = name
        self.source_id = source_id
        self.capture = cv2.VideoCapture(source_id)
        self.current_frame = None
        self.timestamp = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()
        return self

    def _grab_loop(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.01)
                continue

            timestamp = time.time()

            with self.lock:
                self.current_frame = frame
                self.timestamp = timestamp

    def get_frame(self):
        with self.lock:
            return self.current_frame, self.timestamp

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()

    def is_opened(self):
        return self.capture.isOpened()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
