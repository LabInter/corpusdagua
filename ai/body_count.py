import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from pythonosc import udp_client

OSC_IP = "172.21.22.222"
OSC_PORT = 8001
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# === Carregar modelo multipessoa ===
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    min_tracking_confidence=0.2,
    num_poses=10  # máximo de pessoas detectadas
)
detector = vision.PoseLandmarker.create_from_options(options)

# === Abrir câmera ===
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# contador de frames para timestamps
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para formato mediapipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Timestamp artificial (em microssegundos)
    timestamp = frame_counter * 33_000  # ~30fps
    frame_counter += 1

    # Processar detecção
    results = detector.detect_for_video(mp_image, timestamp)

    # Contar pessoas detectadas
    num_people = len(results.pose_landmarks)

    client.send_message("/body/count", float(num_people+1))

    # === Desenhar landmarks de cada pessoa ===
    for pose_landmarks in results.pose_landmarks:
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
            ) for lm in pose_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            landmark_list,
            mp.solutions.pose.POSE_CONNECTIONS
        )

    # Mostrar número de pessoas na tela
    cv2.putText(frame, f"Pessoas detectadas: {num_people}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Deteccao de Pessoas", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
