import cv2
import numpy as np
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient


# ---------------- OSC ----------------

OSC_IP = "127.0.0.1"
OSC_PORT = 8000

client = SimpleUDPClient(OSC_IP, OSC_PORT)

# ---------------- CONFIG ----------------

MAX_PEOPLE = 5
LERP_SPEED = 0.05

LARGURA_EXIBICAO = 1920
ALTURA_EXIBICAO = 1080

valor_atual = 0.0
modo_teste = False

# ---------------- MODELO MEDIAPIPE ----------------

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Baixando modelo hand_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo baixado.")

# ---------------- YOLO ----------------

model_yolo = YOLO("yolov8n.pt")

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a câmera.")

# ---------------- WINDOW ----------------

WINDOW_NAME = "Detector Pessoas + Maos"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, LARGURA_EXIBICAO, ALTURA_EXIBICAO)

# ---------------- DESENHO DE LANDMARKS (sem solutions) ----------------

# 21 conexões da mão (índices dos landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # indicador
    (0, 9), (9, 10), (10, 11), (11, 12),   # médio
    (0, 13), (13, 14), (14, 15), (15, 16), # anelar
    (0, 17), (17, 18), (18, 19), (19, 20), # mínimo
    (5, 9), (9, 13), (13, 17),             # palma
]


def draw_hand_landmarks(frame, landmarks, cor_ponto, cor_linha):
    h, w = frame.shape[:2]

    # Converte coordenadas normalizadas para pixels
    pontos = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ]

    # Desenha conexões
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pontos[a], pontos[b], cor_linha, 2)

    # Desenha pontos
    for px, py in pontos:
        cv2.circle(frame, (px, py), 5, cor_ponto, -1)


# ---------------- MEDIAPIPE HAND LANDMARKER ----------------

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

print("q = sair")
print("t = modo teste")

frame_timestamp_ms = 0

with mp_vision.HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ====================================
        # YOLO - PESSOAS
        # ====================================

        current_frame_people = 0
        yolo_results = model_yolo(frame, verbose=False)

        for result in yolo_results:
            for box in result.boxes:

                cls = int(box.cls[0])
                if cls != 0:
                    continue

                conf = float(box.conf[0])
                if conf < 0.4:
                    continue

                current_frame_people += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ====================================
        # MEDIAPIPE HAND LANDMARKER
        # ====================================

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33  # ~30 fps

        hand_result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        if hand_result.hand_landmarks:

            for idx, landmarks in enumerate(hand_result.hand_landmarks):

                handedness = hand_result.handedness[idx][0].display_name

                # Cores: azul = esquerda, laranja = direita
                if handedness == "Left":
                    cor_ponto = (255, 80, 80)
                    cor_linha = (200, 50, 50)
                else:
                    cor_ponto = (80, 165, 255)
                    cor_linha = (50, 130, 200)

                draw_hand_landmarks(frame, landmarks, cor_ponto, cor_linha)

                # Pulso = landmark 0
                wrist = landmarks[0]
                mao_x = float((wrist.x * 2.0) - 1.0)
                mao_y = float(((1.0 - wrist.y) * 2.0) - 1.0)

                px = int(wrist.x * w)
                py = int(wrist.y * h)

                cv2.circle(frame, (px, py), 12, cor_ponto, -1)

                # Label da mão
                cv2.putText(
                    frame,
                    handedness,
                    (px + 15, py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cor_ponto,
                    2
                )

                if handedness == "Left":
                    client.send_message(
                        "/posicao_mao_esquerda",
                        [mao_x, mao_y]
                    )
                else:
                    client.send_message(
                        "/posicao_mao_direita",
                        [mao_x, mao_y]
                    )

        # ====================================
        # LERP CHUVA
        # ====================================

        pessoas = 5 if modo_teste else current_frame_people

        valor_alvo = np.clip(pessoas / MAX_PEOPLE, 0.0, 1.0)
        valor_atual += (valor_alvo - valor_atual) * LERP_SPEED

        client.send_message("/construcao", float(valor_atual))

        # ====================================
        # UI
        # ====================================

        cv2.putText(
            frame, f"Pessoas: {pessoas}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Chuva: {int(valor_atual * 100)}%",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2
        )

        if modo_teste:
            cv2.putText(
                frame, "MODO TESTE",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )

        frame_show = cv2.resize(frame, (LARGURA_EXIBICAO, ALTURA_EXIBICAO))
        cv2.imshow(WINDOW_NAME, frame_show)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key in (ord("t"), ord("T")):
            modo_teste = not modo_teste

cap.release()
cv2.destroyAllWindows()