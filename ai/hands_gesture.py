import cv2
import mediapipe as mp
from pythonosc import udp_client

# Configuração do OSC
OSC_IP = "192.168.0.14"  # IP do destino
OSC_PORT = 8001         # Porta do destino
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# Configuração do MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Abrir câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Índices dos dedos
finger_tips = [4, 8, 12, 16, 20]
finger_pips = [2, 6, 10, 14, 18]  # articulações proximais

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # --- Cálculo de dedos estendidos ---
            fingers_extended = 0
            distances = []
            for tip, pip in zip(finger_tips, finger_pips):
                tip_y = hand_landmarks.landmark[tip].y
                pip_y = hand_landmarks.landmark[pip].y
                tip_x = hand_landmarks.landmark[tip].x
                pip_x = hand_landmarks.landmark[pip].x

                # Verifica se dedo está estendido
                if tip_y < pip_y:  # palma voltada para câmera
                    fingers_extended += 1

                # Distância normalizada entre ponta e articulação
                dist = ((tip_x - pip_x) ** 2 + (tip_y - pip_y) ** 2) ** 0.5
                distances.append(dist)

            # --- Classificação do gesto ---
            if fingers_extended == 0:
                gesture = "Fechada"
                print("fechada")
            elif fingers_extended == 5:
                gesture = "Aberta"
            else:
                gesture = f"{fingers_extended} dedos"
                if fingers_extended == 1:
                    client.send_message("/zero", 0)


            # --- Cálculo de "valor de aperto" (0 a 1) ---
            avg_dist = sum(distances) / len(distances)
            grip_strength = max(0.0, min(1.0, 1 - (avg_dist * 4)))  # ajuste do fator

            # --- Enviar dados via OSC ---
            client.send_message("/hand/gesture", gesture)
            client.send_message("/hand/grip", grip_strength)

            # --- Desenho na tela ---
            h, w, _ = frame.shape
            cv2.putText(frame, f"Gesto: {gesture}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(frame, f"Aperto: {grip_strength:.2f}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        cv2.putText(frame, "No hand detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        client.send_message("/hand/exit", 0)

    # Mostrar imagem
    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
