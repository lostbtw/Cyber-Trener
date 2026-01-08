import cv2
import numpy as np
import threading
import time


class CalibratedCamera:
    """A camera class that applies lens distortion correction using calibration data."""

    def __init__(self, source_id, calib_file_path=None, name="Camera"):
        self.name = name
        self.source_id = source_id
        self.capture = cv2.VideoCapture(source_id)

        self.mtx = None
        self.dist = None
        self.new_camera_matrix = None
        self.roi = None
        self.mapx = None
        self.mapy = None

        if calib_file_path:
            self._load_calibration(calib_file_path)
        self.current_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def _load_calibration(self, calib_file_path):
        """Load calibration data from .npz file."""
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
        """Initialize undistortion maps for the given frame size."""
        h, w = frame_shape[:2]
        if self.mtx is not None:
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h)
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.new_camera_matrix, (w, h), 5
            )

    def _apply_calibration(self, frame):
        """Apply lens distortion correction to frame."""
        if self.mapx is None:
            self._init_undistort_maps(frame.shape)

        if self.mapx is not None:
            frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            # Crop to remove black borders
            if self.roi is not None:
                x, y, w, h = self.roi
                if w > 0 and h > 0:
                    frame = frame[y : y + h, x : x + w]

        return frame

    def start(self):
        """Start the camera capture thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        """Background thread that continuously captures and processes frames."""
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.mtx is not None:
                frame = self._apply_calibration(frame)

            with self.lock:
                self.current_frame = frame

    def get_frame(self):
        """Get the current calibrated frame."""
        with self.lock:
            return self.current_frame

    def read(self):
        """Read a single frame (blocking, without threading)."""
        ret, frame = self.capture.read()
        if ret and self.mtx is not None:
            frame = self._apply_calibration(frame)
        return ret, frame

    def stop(self):
        """Stop the camera capture thread and release resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()

    def is_opened(self):
        """Check if the camera is opened."""
        return self.capture.isOpened()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
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
        ret, frame = cam.read()
        if not ret:
            break

        # Resize for display
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
