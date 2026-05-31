import cv2
import mediapipe as mp
from pythonosc import udp_client

OSC_IP = "10.0.0.131"
OSC_PORT = 7400
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

center_indices = [0, 5, 9, 13, 17]  
center_indices_size = len(center_indices)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(image_rgb)
    

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            avg_x = sum([hand_landmarks.landmark[i].x for i in center_indices]) / center_indices_size
            avg_y = sum([hand_landmarks.landmark[i].y for i in center_indices]) / center_indices_size


            client.send_message("/center", [avg_x, avg_y])
            print([avg_x, avg_y])

            h, w, _ = frame.shape
            cx, cy = int(avg_x * w), int(avg_y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        client.send_message(f"/hand/exit", 0)

    # No final, antes do imshow
    scale_percent = 150  # aumenta 150% o tamanho da janela
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Hand Center Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()