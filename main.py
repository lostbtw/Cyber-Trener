import cv2
import time
from camera import CalibratedCamera, resize_to_height

# ================= CONFIGURATION =================
# Adjust these IDs based on which is which on your system
SRC_SIDE = 1
SRC_FRONT = 0

# Calibration File Paths
PATH_CALIB_SIDE = "tools/calibration_results/camera_params_calibration_mobile.mp4.npz"
PATH_CALIB_FRONT = "tools/calibration_results/camera_params_calibration_laptop.mp4.npz"

DISPLAY_HEIGHT = 480


if __name__ == "__main__":
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
