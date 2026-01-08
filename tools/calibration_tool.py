import cv2
import numpy as np
import os

VIDEO_PATH = "example_video.mp4"  # path to the file with calibration video
# use calibration_pattern.png in tool directory to create calibration video
CHESSBOARD_SIZE = (9, 6)
# square size in mm
SQUARE_SIZE = 36


def run_calibration():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(
        -1, 2
    )
    objp = objp * SQUARE_SIZE

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: can't open videofile {VIDEO_PATH}")
        return

    frame_count = 0
    success_count = 0

    print("Analyzing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
            imgpoints.append(corners2)
            success_count += 1

            # Visualization (optional - slows down processing)
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret)
            cv2.imshow("Calibration Preview", frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if success_count < 10:
        print(f"Not enough frames ({success_count}). Record a longer video.")
        return

    print(f"\n{success_count} frames collected. Analyzing results...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n" + "=" * 40)
    print("SUCCESS!")
    print("=" * 40)
    print(f"Reprojection error (average error in pixels): {ret:.4f}")
    print("-" * 40)
    print("COPY THE DATA BELOW TO YOUR MAIN PROGRAM:")
    print("-" * 40)

    # Format output for copy-paste into Python
    print(
        f"CAMERA_MATRIX = np.array({np.array2string(mtx, separator=', ')}, dtype=np.float32)"
    )
    print(
        f"\nDIST_COEFFS = np.array({np.array2string(dist, separator=', ')}, dtype=np.float32)"
    )
    print("=" * 40)

    # Optional: Save to .npz file
    output_dir = "calibration_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/camera_params_{os.path.basename(VIDEO_PATH)}.npz"
    np.savez(output_file, mtx=mtx, dist=dist)
    print(f"Also saved to file: {output_file}")


if __name__ == "__main__":
    run_calibration()
