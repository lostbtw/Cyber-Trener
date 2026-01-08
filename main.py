import cv2
import numpy as np
import threading
import time

# ================= CONFIGURATION =================
# Adjust these IDs based on which is which on your system
SRC_SIDE = 1
SRC_FRONT = 0

# Calibration File Paths
PATH_CALIB_SIDE = "tools\calibration_results\camera_params_calibration_mobile.mp4.npz"
PATH_CALIB_FRONT = "tools\calibration_results\camera_params_calibration_laptop.mp4.npz"

DISPLAY_HEIGHT = 480


class CalibratedCamera:
    def __init__(self, source_id, calib_file_path, name="Camera"):
        self.name = name
        self.capture = cv2.VideoCapture(source_id)

        try:
            with np.load(calib_file_path) as data:
                self.mtx = data["mtx"]
                self.dist = data["dist"]
            print(f"[{name}] Calibration loaded successfully.")
        except Exception as e:
            print(
                f"[{name}] WARNING: Could not load calibration file '{calib_file_path}'. Using raw image."
            )
            self.mtx = None
            self.dist = None

        self.new_camera_matrix = None
        self.roi = None
        self.mapx = None
        self.mapy = None

        self.current_frame = None
        self.running = False
        self.lock = threading.Lock()

    def init_undistort_maps(self, frame_shape):
        h, w = frame_shape[:2]
        if self.mtx is not None:
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h)
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.new_camera_matrix, (w, h), 5
            )

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.01)
                continue

            if self.mapx is None:
                self.init_undistort_maps(frame.shape)

            if self.mapx is not None:
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
                if self.roi is not None:
                    x, y, w, h = self.roi
                    if w > 0 and h > 0:
                        frame = frame[y : y + h, x : x + w]

            with self.lock:
                self.current_frame = frame

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def stop(self):
        self.running = False
        self.capture.release()


def resize_to_height(image, target_height):
    if image is None:
        return None
    (h, w) = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))


def main():
    cam_side = CalibratedCamera(SRC_SIDE, PATH_CALIB_SIDE, "SIDE (Left)")
    cam_front = CalibratedCamera(SRC_FRONT, PATH_CALIB_FRONT, "FRONT (Right)")

    cam_side.start()
    cam_front.start()
    print("Starting system... Press 'q' to exit.")
    time.sleep(1.0)

    while True:
        frame_side = cam_side.get_frame()
        frame_front = cam_front.get_frame()

        if frame_side is None or frame_front is None:
            continue

        view_side = resize_to_height(frame_side, DISPLAY_HEIGHT)
        view_front = resize_to_height(frame_front, DISPLAY_HEIGHT)

        combined_view = cv2.hconcat([view_side, view_front])

        cv2.putText(
            combined_view,
            "SIDE CAMERA",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        width_side = view_side.shape[1]
        cv2.putText(
            combined_view,
            "FRONT CAMERA",
            (width_side + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("CyberTrener - Synchronized View", combined_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam_side.stop()
    cam_front.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
