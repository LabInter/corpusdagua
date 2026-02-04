import cv2
import mediapipe as mp
import socket
import json

# Comunicação com Unreal via socket UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]  # dedo indicador
            x, y = index_tip.x, index_tip.y
            print(x,y)
            data = json.dumps({"x": x, "y": y})
            sock.sendto(data.encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("Hand Tracking", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
