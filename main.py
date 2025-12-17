import cv2
import numpy as np

# 1. Konfiguracja źródeł wideo
# Zgodnie z projektem, mogą to być strumienie z telefonów [cite: 27]
# Wpisz '0' i '1' dla kamer USB lub adresy URL (np. 'http://192.168.0.x:8080/video')
cap_front = cv2.VideoCapture(0)
cap_side = cv2.VideoCapture(1)

# Ustawienie docelowej wysokości okna (dla estetyki)
TARGET_HEIGHT = 480

while True:
    # 2. Pobranie klatek
    ret_front, frame_front = cap_front.read()
    ret_side, frame_side = cap_side.read()

    # Sprawdzenie czy kamery działają
    if not ret_front or not ret_side:
        print("Błąd odczytu z kamer. Sprawdź połączenie.")
        break

    # 3. Skalowanie do tej samej wysokości
    # Jest to konieczne, aby połączyć obrazy funkcją hstack
    def resize_to_height(img, height):
        ratio = height / img.shape[0]
        width = int(img.shape[1] * ratio)
        return cv2.resize(img, (width, height))

    frame_front = resize_to_height(frame_front, TARGET_HEIGHT)
    frame_side = resize_to_height(frame_side, TARGET_HEIGHT)

    # 4. Dodanie napisów (jak na zdjęciu z dokumentacji )
    # Kolor (255, 0, 0) to niebieski w formacie BGR
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame_front, "Front Camera", (30, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame_side, "Side Camera", (30, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA
    )

    # ---------------------------------------------------------
    # TU BĘDZIE MIEJSCE NA TWOJE ALGORYTMY (MediaPipe / YOLO)
    # Nanoszenie szkieletu musi odbyć się PRZED sklejeniem obrazów.
    # ---------------------------------------------------------

    # 5. Łączenie obrazów (horyzontalnie)
    # Tworzymy jeden szeroki obraz z dwóch mniejszych
    combined_window = np.hstack((frame_front, frame_side))

    # 6. Wyświetlenie wyniku
    cv2.imshow("Cyber Trener - Podglad", combined_window)

    # Wyjście po wciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Sprzątanie
cap_front.release()
cap_side.release()
cv2.destroyAllWindows()
