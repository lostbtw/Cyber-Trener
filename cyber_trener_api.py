import time
import cv2
import numpy as np

from utils.calibrated_camera import CalibratedCamera
from utils.camera_synchronizer import CameraSynchronizer
from utils.pose_detector import PoseDetector

# Landmark ID to name mapping
LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


class ProcessedFrame:
    """
    Single synchronized frame from both cameras with skeleton data.

    Attributes:
        timestamp - capture time in seconds
        frame_front - front camera image (BGR numpy array)
        frame_side - side camera image (BGR numpy array)
        skeleton_front - list of landmarks from front camera, or None
        skeleton_side - list of landmarks from side camera, or None
        skeleton_front_world - world coordinates in meters (for angles)
        skeleton_side_world - world coordinates in meters (for angles)
        is_synced - True if cameras are synchronized
        sync_diff_ms - time difference between cameras in ms

    Each landmark is a dict: {id, name, x, y, z, visibility}
    """

    def __init__(
        self,
        timestamp,
        frame_front,
        frame_side,
        skeleton_front,
        skeleton_side,
        skeleton_front_world,
        skeleton_side_world,
        is_synced,
        sync_diff_ms,
    ):
        self.timestamp = timestamp
        self.frame_front = frame_front
        self.frame_side = frame_side
        self.skeleton_front = skeleton_front
        self.skeleton_side = skeleton_side
        self.skeleton_front_world = skeleton_front_world
        self.skeleton_side_world = skeleton_side_world
        self.is_synced = is_synced
        self.sync_diff_ms = sync_diff_ms


class CyberTrener:
    """
    Main API class for dual-camera pose detection.

    Usage:
        trainer = CyberTrener(camera_front=0, camera_side=1)
        trainer.start()

        while True:
            frame = trainer.get_frame()
            if frame is None:
                continue

            # Use frame.skeleton_front, frame.skeleton_side, etc.

            if cv2.waitKey(1) == ord('q'):
                break

        trainer.stop()
    """

    def __init__(
        self,
        camera_front=0,
        camera_side=1,
        calibration_front=None,
        calibration_side=None,
        rotate_side=True,
        max_sync_diff_ms=40.0,
    ):
        """
        Initialize the trainer.

        Args:
            camera_front: Front camera device ID
            camera_side: Side camera device ID
            calibration_front: Path to front camera calibration .npz file
            calibration_side: Path to side camera calibration .npz file
            rotate_side: Rotate side camera 90° clockwise (default: True)
            max_sync_diff_ms: Max allowed time difference between cameras (default: 40ms)
        """
        self._camera_front_id = camera_front
        self._camera_side_id = camera_side
        self._calibration_front = calibration_front
        self._calibration_side = calibration_side
        self._rotate_side = rotate_side
        self._max_sync_diff_ms = max_sync_diff_ms

        self._cam_front = None
        self._cam_side = None
        self._synchronizer = None
        self._detector_front = None
        self._detector_side = None
        self._running = False

    def start(self):
        """Start cameras and pose detection. Call this before get_frame()."""
        self._cam_front = CalibratedCamera(
            self._camera_front_id, self._calibration_front, "Front"
        )
        self._cam_side = CalibratedCamera(
            self._camera_side_id, self._calibration_side, "Side"
        )
        self._synchronizer = CameraSynchronizer(
            self._cam_front, self._cam_side, self._max_sync_diff_ms
        )
        self._detector_front = PoseDetector()
        self._detector_side = PoseDetector()

        self._cam_front.start()
        self._cam_side.start()
        self._running = True

        time.sleep(0.5)
        print("Started")

    def stop(self):
        """Stop cameras and release resources."""
        self._running = False
        if self._detector_front:
            self._detector_front.close()
        if self._detector_side:
            self._detector_side.close()
        if self._cam_front:
            self._cam_front.stop()
        if self._cam_side:
            self._cam_side.stop()
        cv2.destroyAllWindows()
        print("Stopped")

    def get_frame(self, skip_out_of_sync=True):
        """
        Get next synchronized frame with skeleton data.

        Args:
            skip_out_of_sync: If True, returns None when cameras are out of sync

        Returns:
            ProcessedFrame with all data, or None if no frame available
        """
        if not self._running:
            return None

        frame_front, frame_side, synced, diff_ms = self._synchronizer.get_synced_pair()

        if frame_front is None or frame_side is None:
            return None

        if self._rotate_side:
            frame_side = cv2.rotate(frame_side, cv2.ROTATE_90_CLOCKWISE)

        if not synced and skip_out_of_sync:
            return None

        results_front = self._detector_front.detect(frame_front)
        results_side = self._detector_side.detect(frame_side)

        skeleton_front = self._extract_skeleton(results_front, frame_front.shape)
        skeleton_side = self._extract_skeleton(results_side, frame_side.shape)
        skeleton_front_world = self._extract_skeleton_world(results_front)
        skeleton_side_world = self._extract_skeleton_world(results_side)

        return ProcessedFrame(
            timestamp=time.time(),
            frame_front=frame_front,
            frame_side=frame_side,
            skeleton_front=skeleton_front,
            skeleton_side=skeleton_side,
            skeleton_front_world=skeleton_front_world,
            skeleton_side_world=skeleton_side_world,
            is_synced=synced,
            sync_diff_ms=diff_ms,
        )

    def _extract_skeleton(self, results, frame_shape):
        """Convert pose results to list of landmark dicts with pixel coordinates"""
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        h, w = frame_shape[:2]
        landmarks = []

        for idx, lm in enumerate(results.pose_landmarks[0]):
            landmarks.append(
                {
                    "id": idx,
                    "name": LANDMARK_NAMES.get(idx, f"point_{idx}"),
                    "x": int(lm.x * w),
                    "y": int(lm.y * h),
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
                }
            )

        return landmarks

    def _extract_skeleton_world(self, results):
        """Convert pose results to list of landmark dicts with world coordinates"""
        if not results.pose_world_landmarks or len(results.pose_world_landmarks) == 0:
            return None

        landmarks = []

        for idx, lm in enumerate(results.pose_world_landmarks[0]):
            landmarks.append(
                {
                    "id": idx,
                    "name": LANDMARK_NAMES.get(idx, f"point_{idx}"),
                    "x": lm.x,  # meters
                    "y": lm.y,  # meters
                    "z": lm.z,  # meters
                    "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
                }
            )

        return landmarks

    def draw_skeleton(self, frame, skeleton):
        """Draw skeleton on frame. Returns frame with skeleton drawn."""
        if skeleton is None:
            return frame

        # Connections between landmarks
        connections = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
            (15, 17),
            (15, 19),
            (15, 21),
            (16, 18),
            (16, 20),
            (16, 22),
        ]

        for start_id, end_id in connections:
            start = next((p for p in skeleton if p["id"] == start_id), None)
            end = next((p for p in skeleton if p["id"] == end_id), None)
            if start and end and start["visibility"] > 0.5 and end["visibility"] > 0.5:
                cv2.line(
                    frame,
                    (start["x"], start["y"]),
                    (end["x"], end["y"]),
                    (0, 255, 0),
                    2,
                )

        for point in skeleton:
            if point["visibility"] > 0.5:
                cv2.circle(frame, (point["x"], point["y"]), 5, (0, 0, 255), -1)

        return frame

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3. Returns angle in degrees."""
    a = np.array([p1["x"], p1["y"], p1["z"]])
    b = np.array([p2["x"], p2["y"], p2["z"]])
    c = np.array([p3["x"], p3["y"], p3["z"]])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(angle))


def get_landmark(skeleton, landmark_id):
    """Get landmark by ID (0-32) from skeleton list. Returns dict or None."""
    if skeleton is None:
        return None
    return next((p for p in skeleton if p["id"] == landmark_id), None)


def get_landmark_by_name(skeleton, name):
    """Get landmark by name (e.g. 'left_elbow') from skeleton. Returns dict or None."""
    if skeleton is None:
        return None
    return next((p for p in skeleton if p["name"] == name), None)


def get_elbow_angles(skeleton_world):
    """Get elbow angles for bicep curls. Returns {'left': deg, 'right': deg} or None."""
    if skeleton_world is None:
        return None

    try:
        left = calculate_angle(
            get_landmark(skeleton_world, 11),
            get_landmark(skeleton_world, 13),
            get_landmark(skeleton_world, 15),
        )
        right = calculate_angle(
            get_landmark(skeleton_world, 12),
            get_landmark(skeleton_world, 14),
            get_landmark(skeleton_world, 16),
        )
        return {"left": left, "right": right}
    except:
        return None


def get_shoulder_angles(skeleton_world):
    """Get shoulder angles (arm raise). Returns {'left': deg, 'right': deg} or None."""
    if skeleton_world is None:
        return None

    try:
        left = calculate_angle(
            get_landmark(skeleton_world, 23),
            get_landmark(skeleton_world, 11),
            get_landmark(skeleton_world, 13),
        )
        right = calculate_angle(
            get_landmark(skeleton_world, 24),
            get_landmark(skeleton_world, 12),
            get_landmark(skeleton_world, 14),
        )
        return {"left": left, "right": right}
    except:
        return None
