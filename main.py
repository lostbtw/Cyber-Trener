import cv2
import time
from calibrated_camera import CalibratedCamera, resize_to_height
from camera_synchronizer import CameraSynchronizer
from pose_detector import PoseDetector

# ================= CONFIGURATION =================
# Adjust these IDs based on which is which on your system
SRC_SIDE = 1  # Slave camera
SRC_FRONT = 0  # Master camera
MAX_SYNC_DIFF_MS = 40  # Maximum allowed time difference between cameras in milliseconds
ROTATE_SIDE_CAMERA = True  # Rotate side camera if it is mounted vertically

# Calibration File Paths
PATH_CALIB_SIDE = "tools/calibration_results/camera_params_calibration_mobile.mp4.npz"
PATH_CALIB_FRONT = "tools/calibration_results/camera_params_calibration_laptop.mp4.npz"

DISPLAY_HEIGHT = 480

# Pose detection settings
POSE_MIN_DETECTION_CONFIDENCE = 0.7
POSE_MIN_TRACKING_CONFIDENCE = 0.5

if __name__ == "__main__":
    cam_front = CalibratedCamera(SRC_FRONT, PATH_CALIB_FRONT, "FRONT (Master)")
    cam_side = CalibratedCamera(SRC_SIDE, PATH_CALIB_SIDE, "SIDE (Slave)")
    synchronizer = CameraSynchronizer(cam_front, cam_side, MAX_SYNC_DIFF_MS)

    detector_front = PoseDetector(
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    )
    detector_side = PoseDetector(
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    )

    cam_front.start()
    cam_side.start()
    print("Starting CyberTrener with pose detection...")
    print(f"Master: FRONT camera, Slave: SIDE camera")
    print(f"Max sync difference: {MAX_SYNC_DIFF_MS}ms")
    print("Press 'q' to exit.")
    time.sleep(1.5)

    out_of_sync_count = 0
    frame_count = 0

    while True:
        frame_front, frame_side, synced, time_diff_ms = synchronizer.get_synced_pair()

        if frame_front is None or frame_side is None:
            continue

        if ROTATE_SIDE_CAMERA:
            frame_side = cv2.rotate(frame_side, cv2.ROTATE_90_CLOCKWISE)

        if not synced:
            out_of_sync_count += 1
            print(
                f"OUT OF SYNC: Time difference = {time_diff_ms:.2f}ms (>{MAX_SYNC_DIFF_MS}ms) "
                f"[Total: {out_of_sync_count}]"
            )
            continue

        frame_count += 1

        results_front = detector_front.detect(frame_front)
        results_side = detector_side.detect(frame_side)

        data_front = detector_front.get_exercise_data(results_front, frame_front.shape)
        data_side = detector_side.get_exercise_data(results_side, frame_side.shape)

        elbow_angles_front = detector_front.get_elbow_angles(data_front["pose_world"])
        elbow_angles_side = detector_side.get_elbow_angles(data_side["pose_world"])

        view_front = frame_front.copy()
        view_side = frame_side.copy()

        detector_front.draw_landmarks(view_front, results_front, draw_pose=True)
        detector_side.draw_landmarks(view_side, results_side, draw_pose=True)

        view_front = resize_to_height(view_front, DISPLAY_HEIGHT)
        view_side = resize_to_height(view_side, DISPLAY_HEIGHT)

        combined_view = cv2.hconcat([view_side, view_front])
        status_side = "Pose:OK" if data_side["has_pose"] else "No detection"

        status_front = "Pose:OK" if data_front["has_pose"] else "No detection"

        angle_text_side = ""
        angle_text_front = ""
        if elbow_angles_side:
            angle_text_side = f" | Elbow L:{elbow_angles_side['left']:.0f} R:{elbow_angles_side['right']:.0f}"
        if elbow_angles_front:
            angle_text_front = f" | Elbow L:{elbow_angles_front['left']:.0f} R:{elbow_angles_front['right']:.0f}"

        label_side = f"SIDE | {status_side}{angle_text_side}"
        label_front = f"FRONT | {status_front}{angle_text_front}"

        cv2.putText(
            combined_view,
            label_side,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )

        width_side = view_side.shape[1]
        cv2.putText(
            combined_view,
            label_front,
            (width_side + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )

        cv2.putText(
            combined_view,
            f"Sync: {time_diff_ms:.1f}ms | Frame: {frame_count}",
            (10, combined_view.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.imshow("CyberTrener - Pose Detection", combined_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total out-of-sync frames: {out_of_sync_count}")

    # Cleanup
    detector_front.close()
    detector_side.close()
    cam_front.stop()
    cam_side.stop()
    cv2.destroyAllWindows()
