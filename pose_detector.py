import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:
    # Pose landmark indices (33 total)
    POSE_LANDMARKS = {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    POSE_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
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
        (27, 29),
        (29, 31),
        (28, 30),
        (30, 32),
        (15, 17),
        (15, 19),
        (15, 21),
        (16, 18),
        (16, 20),
        (16, 22),
    ]

    def __init__(
        self,
        model_path="pose_landmarker_full.task",
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        num_poses=1,
    ):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        print(
            f"[PoseDetector] Initialized with min_detection_confidence={min_detection_confidence}"
        )

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)

        return results

    def get_exercise_data(self, results, frame_shape):
        has_pose = len(results.pose_landmarks) > 0 if results.pose_landmarks else False

        return {
            "pose": (
                self.extract_pose_landmarks(results, frame_shape) if has_pose else None
            ),
            "pose_world": self.extract_world_landmarks(results) if has_pose else None,
            "has_pose": has_pose,
        }

    def extract_pose_landmarks(self, results, frame_shape):
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        h, w = frame_shape[:2]
        landmarks = {}
        pose_landmarks = results.pose_landmarks[0]

        for name, idx in self.POSE_LANDMARKS.items():
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                landmarks[name] = {
                    "x": int(lm.x * w),
                    "y": int(lm.y * h),
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
                }

        return landmarks

    def extract_world_landmarks(self, results):
        if not results.pose_world_landmarks or len(results.pose_world_landmarks) == 0:
            return None

        landmarks = {}
        pose_world = results.pose_world_landmarks[0]

        for name, idx in self.POSE_LANDMARKS.items():
            if idx < len(pose_world):
                lm = pose_world[idx]
                landmarks[name] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
                }

        return landmarks

    def draw_landmarks(self, frame, results, draw_pose=True):
        if (
            not draw_pose
            or not results.pose_landmarks
            or len(results.pose_landmarks) == 0
        ):
            return frame

        h, w = frame.shape[:2]
        pose_landmarks = results.pose_landmarks[0]

        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        for landmark in pose_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return frame

    def calculate_angle(self, p1, p2, p3):
        a = np.array([p1["x"], p1["y"], p1["z"]])
        b = np.array([p2["x"], p2["y"], p2["z"]])
        c = np.array([p3["x"], p3["y"], p3["z"]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def get_elbow_angles(self, world_landmarks):
        if world_landmarks is None:
            return None

        try:
            left_angle = self.calculate_angle(
                world_landmarks["left_shoulder"],
                world_landmarks["left_elbow"],
                world_landmarks["left_wrist"],
            )
            right_angle = self.calculate_angle(
                world_landmarks["right_shoulder"],
                world_landmarks["right_elbow"],
                world_landmarks["right_wrist"],
            )
            return {"left": left_angle, "right": right_angle}
        except KeyError:
            return None

    def get_shoulder_angles(self, world_landmarks):
        if world_landmarks is None:
            return None

        try:
            left_angle = self.calculate_angle(
                world_landmarks["left_hip"],
                world_landmarks["left_shoulder"],
                world_landmarks["left_elbow"],
            )
            right_angle = self.calculate_angle(
                world_landmarks["right_hip"],
                world_landmarks["right_shoulder"],
                world_landmarks["right_elbow"],
            )
            return {"left": left_angle, "right": right_angle}
        except KeyError:
            return None

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
