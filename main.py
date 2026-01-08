import cv2
import time
from calibrated_camera import CalibratedCamera, resize_to_height
from camera_synchronizer import CameraSynchronizer

# ================= CONFIGURATION =================
# Adjust these IDs based on which is which on your system
SRC_SIDE = 1  # Slave camera
SRC_FRONT = 0  # Master camera
MAX_SYNC_DIFF_MS = 40  # Maximum allowed time difference between cameras in milliseconds

# Calibration File Paths
PATH_CALIB_SIDE = "tools/calibration_results/camera_params_calibration_mobile.mp4.npz"
PATH_CALIB_FRONT = "tools/calibration_results/camera_params_calibration_laptop.mp4.npz"

DISPLAY_HEIGHT = 480

if __name__ == "__main__":
    cam_front = CalibratedCamera(SRC_FRONT, PATH_CALIB_FRONT, "FRONT (Master)")
    cam_side = CalibratedCamera(SRC_SIDE, PATH_CALIB_SIDE, "SIDE (Slave)")
    synchronizer = CameraSynchronizer(cam_front, cam_side, MAX_SYNC_DIFF_MS)

    cam_front.start()
    cam_side.start()
    print("Starting system with camera synchronization...")
    print(f"Master: FRONT camera, Slave: SIDE camera")
    print(f"Max sync difference: {MAX_SYNC_DIFF_MS}ms")
    print("Press 'q' to exit.")
    time.sleep(1.5)

    out_of_sync_count = 0

    while True:
        frame_front, frame_side, synced, time_diff_ms = synchronizer.get_synced_pair()

        if frame_front is None or frame_side is None:
            continue

        if not synced:
            out_of_sync_count += 1
            print(
                f"OUT OF SYNC: Time difference = {time_diff_ms:.2f}ms (>{MAX_SYNC_DIFF_MS}ms) "
                f"[Total: {out_of_sync_count}]"
            )
            continue

        view_front = resize_to_height(frame_front, DISPLAY_HEIGHT)
        view_side = resize_to_height(frame_side, DISPLAY_HEIGHT)

        combined_view = cv2.hconcat([view_side, view_front])

        label_side = f"SIDE | latency: {time_diff_ms:.1f}ms"
        label_front = "FRONT"

        cv2.putText(
            combined_view,
            label_side,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        width_side = view_side.shape[1]
        cv2.putText(
            combined_view,
            label_front,
            (width_side + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        cv2.imshow("CyberTrener", combined_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nTotal out-of-sync frames: {out_of_sync_count}")
    cam_front.stop()
    cam_side.stop()
    cv2.destroyAllWindows()
