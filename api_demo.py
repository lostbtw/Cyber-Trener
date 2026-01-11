import cv2
from cyber_trener_api import CyberTrener, get_elbow_angles
from calibrated_camera import resize_to_height

# ================= CONFIGURATION =================
SRC_FRONT = 0  # Front camera ID
SRC_SIDE = 1  # Side camera ID
MAX_SYNC_DIFF_MS = 40
ROTATE_SIDE_CAMERA = True
DISPLAY_HEIGHT = 480

PATH_CALIB_FRONT = "tools/calibration_results/camera_params_calibration_laptop.mp4.npz"
PATH_CALIB_SIDE = "tools/calibration_results/camera_params_calibration_mobile.mp4.npz"

if __name__ == "__main__":
    trainer = CyberTrener(
        camera_front=SRC_FRONT,
        camera_side=SRC_SIDE,
        calibration_front=PATH_CALIB_FRONT,
        calibration_side=PATH_CALIB_SIDE,
        rotate_side=ROTATE_SIDE_CAMERA,
        max_sync_diff_ms=MAX_SYNC_DIFF_MS,
    )

    trainer.start()
    print("Press 'q' to exit.")

    frame_count = 0
    out_of_sync_count = 0

    while True:
        frame = trainer.get_frame(skip_out_of_sync=False)

        if frame is None:
            continue

        if not frame.is_synced:
            out_of_sync_count += 1
            print(
                f"OUT OF SYNC: {frame.sync_diff_ms:.1f}ms [Total: {out_of_sync_count}]"
            )
            continue

        frame_count += 1
        elbow_front = get_elbow_angles(frame.skeleton_front_world)
        elbow_side = get_elbow_angles(frame.skeleton_side_world)
        view_front = frame.frame_front.copy()
        view_side = frame.frame_side.copy()
        trainer.draw_skeleton(view_front, frame.skeleton_front)
        trainer.draw_skeleton(view_side, frame.skeleton_side)
        view_front = resize_to_height(view_front, DISPLAY_HEIGHT)
        view_side = resize_to_height(view_side, DISPLAY_HEIGHT)
        combined = cv2.hconcat([view_side, view_front])
        status_front = "Pose:OK" if frame.skeleton_front else "No detection"
        status_side = "Pose:OK" if frame.skeleton_side else "No detection"

        angle_front = ""
        angle_side = ""
        if elbow_front:
            angle_front = (
                f" | Elbow L:{elbow_front['left']:.0f} R:{elbow_front['right']:.0f}"
            )
        if elbow_side:
            angle_side = (
                f" | Elbow L:{elbow_side['left']:.0f} R:{elbow_side['right']:.0f}"
            )

        cv2.putText(
            combined,
            f"SIDE | {status_side}{angle_side}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )
        cv2.putText(
            combined,
            f"FRONT | {status_front}{angle_front}",
            (view_side.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )
        cv2.putText(
            combined,
            f"Sync: {frame.sync_diff_ms:.1f}ms | Frame: {frame_count}",
            (10, combined.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            1,
        )

        cv2.imshow("CyberTrener", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nFrames: {frame_count}, Out of sync: {out_of_sync_count}")
    trainer.stop()
