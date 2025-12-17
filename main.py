import cv2
import numpy as np
import time
import mediapipe as mp

# --- PATCH START ---
# Fix for Python 3.13 Windows: "function 'free' not found" in MediaPipe
print("Attempting to patch MediaPipe via ctypes...")
try:
    import ctypes
    import os

    # We patch ctypes.CDLL to inject 'free' into libmediapipe.dll
    # This avoids importing internal MediaPipe modules that might fail.

    class PatchedCDLL(ctypes.CDLL):
        def __init__(self, name, *args, **kwargs):
            super().__init__(name, *args, **kwargs)
            # Check if this is the mediapipe library
            if name and (
                "libmediapipe.dll" in name
                or "libmediapipe.so" in name
                or "libmediapipe.dylib" in name
            ):
                print(f"Intercepted loading of: {name}")
                if not hasattr(self, "free"):
                    print("Injecting 'free' function...")
                    if os.name == "nt":
                        try:
                            self.free = ctypes.cdll.msvcrt.free
                        except Exception as e:
                            print(f"Failed to inject msvcrt.free: {e}")

    # Apply the patch
    ctypes.CDLL = PatchedCDLL
    print("ctypes.CDLL patched successfully.")

except Exception as e:
    print(f"Warning: Failed to patch ctypes: {e}")
# --- PATCH END ---

from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import drawing_utils

# Konfiguracja kamer (0 - zazwyczaj wbudowana, 1 - zazwyczaj USB/DroidCam)
FRONT_CAM_INDEX = 0
SIDE_CAM_INDEX = 1

# Ścieżka do modelu
MODEL_PATH = "pose_landmarker_full.task"


def create_pose_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


# Inicjalizacja modeli MediaPipe
try:
    pose_front = create_pose_landmarker()
    pose_side = create_pose_landmarker()
    print("Pomyślnie zainicjalizowano modele MediaPipe (Tasks API).")
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"Błąd inicjalizacji MediaPipe: {e}")
    print(f"Upewnij się, że plik '{MODEL_PATH}' znajduje się w katalogu roboczym.")
    exit()


def process_pose(frame, pose_landmarker, timestamp_ms):
    """
    Funkcja przyjmuje klatkę obrazu i model MediaPipe.
    Zwraca klatkę z narysowanym szkieletem.
    """
    if frame is None:
        return None

    # Konwersja BGR (OpenCV) -> RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Utworzenie obiektu mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detekcja pozy
    try:
        detection_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as e:
        print(f"Błąd detekcji: {e}")
        return frame

    # Rysowanie szkieletu na oryginalnej klatce (frame)
    if detection_result.pose_landmarks:
        # Tasks API zwraca listę list landmarków (dla każdej wykrytej osoby)
        # Zakładamy jedną osobę
        for landmarks in detection_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_color=(0, 255, 0),
                connection_color=(255, 0, 0),
            )

    return frame


def resize_frame(frame, target_height):
    """Skaluje obraz zachowując proporcje"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_height))


def main():
    # Uruchomienie strumieni wideo
    cap_front = cv2.VideoCapture(FRONT_CAM_INDEX)
    cap_side = cv2.VideoCapture(SIDE_CAM_INDEX)

    # Ustawienie stałej wysokości okna (dla estetycznego połączenia)
    TARGET_HEIGHT = 480

    print("Uruchamianie Cyber Trenera... Wciśnij 'q', aby wyjść.")

    start_time = time.time()

    while True:
        # Odczyt klatek
        ret_front, frame_front = cap_front.read()
        ret_side, frame_side = cap_side.read()

        if not ret_front and not ret_side:
            print("Błąd: Nie wykryto żadnej kamery.")
            break

        # Obliczenie timestampu (w ms)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Obsługa sytuacji, gdy działa tylko jedna kamera (żeby program się nie wywalił)
        if not ret_front:
            frame_front = np.zeros((480, 640, 3), dtype=np.uint8)  # Czarny ekran
            cv2.putText(
                frame_front,
                "Brak kamery przod",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if not ret_side:
            frame_side = np.zeros((480, 640, 3), dtype=np.uint8)  # Czarny ekran
            cv2.putText(
                frame_side,
                "Brak kamery bok",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # --- PRZETWARZANIE MEDIAPIPE ---
        # POPRAWKA: Teraz przechwytujemy zwracane klatki
        if ret_front:
            frame_front = process_pose(frame_front, pose_front, timestamp_ms)
        if ret_side:
            frame_side = process_pose(frame_side, pose_side, timestamp_ms)

        # --- SKŁADANIE WIDOKU ---
        # 1. Skalowanie do równej wysokości
        view_front = resize_frame(frame_front, TARGET_HEIGHT)
        view_side = resize_frame(frame_side, TARGET_HEIGHT)

        # 2. Dodanie opisów
        cv2.putText(
            view_front,
            "Front Camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            view_side,
            "Side Camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # 3. Połączenie obrazów (horyzontalnie)
        combined_view = np.hstack((view_front, view_side))

        # Wyświetlenie
        cv2.imshow("Cyber Trener v1.0 (Python 3.13 Tasks API)", combined_view)

        # Wyjście klawiszem 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Sprzątanie po wyjściu
    cap_front.release()
    cap_side.release()
    pose_front.close()
    pose_side.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
