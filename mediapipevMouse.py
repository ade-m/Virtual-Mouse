import cv2
import mediapipe as mp
import pyautogui
import time

# Inisialisasi MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Untuk menggambar landmark tangan
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Ambil resolusi layar
screen_w, screen_h = pyautogui.size()

# Simpan posisi sebelumnya
prev_x, prev_y = None, None  
mouse_held = False  # Status klik tahan
hold_timer = 0  # Timer untuk klik tahan
hold_threshold = 0.5  # Waktu minimum (detik) klik harus bertahan

# Buka webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Konversi ke RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        num_hands = len(result.multi_hand_landmarks)  # Hitung jumlah tangan
        
        # Jika lebih dari satu tangan, mouse berhenti
        if num_hands > 1:
            print("Dua tangan terdeteksi! Mouse berhenti.")
            prev_x, prev_y = None, None
            if mouse_held:
                pyautogui.mouseUp()
                mouse_held = False
        else:
            for hand_landmarks in result.multi_hand_landmarks:
                # Gambar landmark tangan
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ambil posisi wrist (pergelangan tangan)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Konversi koordinat relatif kamera ke layar
                x = int(wrist.x * w)
                y = int(wrist.y * h)

                if prev_x is not None and prev_y is not None:
                    # Hitung delta movement
                    dx = x - prev_x
                    dy = y - prev_y

                    # Sesuaikan sensitivitas
                    sensitivity = 2  
                    pyautogui.moveRel(dx * sensitivity, dy * sensitivity, duration=0.1)

                # Update posisi sebelumnya
                prev_x, prev_y = x, y

                # **Deteksi Kepalan & Genggaman (Menekan Mouse)**
                finger_tips = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                ]
                
                finger_mcp = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                ]

                # **1. Mengepal (Semua Jari Turun)**
                kepalan = all(finger_tips[i].y > finger_mcp[i].y for i in range(1, 5))  

                # **2. Genggaman (Ibu Jari Naik, Jari Lain Turun)**
                genggaman = (
                    finger_tips[0].y < finger_mcp[0].y and  # Ibu jari naik
                    all(finger_tips[i].y > finger_mcp[i].y for i in range(1, 5))  # Jari lain turun
                )

                # **Klik dan tahan jika mengepal atau menggenggam**
                if kepalan or genggaman:
                    if not mouse_held:
                        pyautogui.mouseDown()
                        mouse_held = True
                        hold_timer = time.time()  # Set waktu awal klik
                        print("Mouse Click Held")

                    # Cek apakah klik sudah bertahan lebih dari threshold
                    if time.time() - hold_timer >= hold_threshold:
                        text_x, text_y = int(wrist.x * w), int(wrist.y * h) + 30
                        cv2.putText(frame, "Klik Tahan", (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Jika genggaman hilang dan waktu tahan sudah cukup, baru lepaskan klik
                    if mouse_held and (time.time() - hold_timer >= hold_threshold):
                        pyautogui.mouseUp()
                        mouse_held = False
                        print("Mouse Released")

    else:
        # Jika tidak ada tangan terdeteksi, pastikan klik dilepas
        if mouse_held:
            pyautogui.mouseUp()
            mouse_held = False
            print("Mouse Released (No Hand Detected)")

    # Tampilkan video
    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
