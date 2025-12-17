import cv2
import numpy as np
from mediapipe.tasks.python import vision


def draw_landmarks(
    image,
    landmarks,
    connections=None,
    landmark_color=(0, 255, 0),
    connection_color=(255, 0, 0),
    thickness=2,
):
    """
    Draws landmarks and connections on the image.
    Args:
        image: The image to draw on (BGR).
        landmarks: A list of NormalizedLandmark objects.
        connections: A list of connection tuples or Connection objects.
    """
    if not landmarks:
        return

    h, w, _ = image.shape

    # Draw connections
    if connections:
        for connection in connections:
            start_idx = connection.start
            end_idx = connection.end

            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]

            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))

            cv2.line(image, start_point, end_point, connection_color, thickness)

    # Draw landmarks
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), thickness * 2, landmark_color, -1)
