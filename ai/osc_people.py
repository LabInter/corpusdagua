"""Detecção de pessoas (YOLO) e mãos (MediaPipe), com envio de dados via OSC."""

import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO


# ==========================================================
# CONFIGURAÇÃO
# ==========================================================
@dataclass(frozen=True)
class Config:
    osc_ip: str = "127.0.0.1"
    osc_port: int = 8000

    max_people: int = 5
    lerp_speed: float = 0.05

    display_width: int = 1920
    display_height: int = 1080
    capture_width: int = 1280
    capture_height: int = 720

    yolo_model_path: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.4
    yolo_person_class_id: int = 0

    hand_model_path: str = "hand_landmarker.task"
    hand_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    max_hands: int = 10

    # Calibração de abertura da mão. Ative debug_openness=True para calibrar:
    # 1) feche o punho e anote o raw_ratio mínimo -> ratio_closed
    # 2) abra a mão totalmente e anote o raw_ratio máximo -> ratio_open
    ratio_closed: float = 1.0
    ratio_open: float = 1.83
    debug_openness: bool = False

    window_name: str = "Detector Pessoas + Maos"


CONFIG = Config()

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),     # Médio
    (0, 13), (13, 14), (14, 15), (15, 16),   # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20),   # Mínimo
    (5, 9), (9, 13), (13, 17),               # Palma
]

FINGER_TIP_IDS = (8, 12, 16, 20)
WRIST_ID = 0
MIDDLE_MCP_ID = 9  # referência de escala estável (não depende da flexão dos dedos)

LEFT_HAND_COLOR_POINT = (255, 80, 80)
LEFT_HAND_COLOR_LINE = (200, 50, 50)
RIGHT_HAND_COLOR_POINT = (80, 165, 255)
RIGHT_HAND_COLOR_LINE = (50, 130, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class HandInfo:
    x: float
    y: float
    openness: float


@dataclass
class Person:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    left_hand: Optional[HandInfo] = None
    right_hand: Optional[HandInfo] = None


@dataclass
class AppState:
    rain_value: float = 0.0
    test_mode: bool = False


# ==========================================================
# SETUP
# ==========================================================
def ensure_hand_model(config: Config) -> None:
    if os.path.exists(config.hand_model_path):
        return
    print("Baixando modelo hand_landmarker.task...")
    urllib.request.urlretrieve(config.hand_model_url, config.hand_model_path)
    print("Modelo baixado.")


def create_yolo_model(config: Config) -> tuple[YOLO, str, bool]:
    model = YOLO(config.yolo_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half_precision = device == "cuda"
    return model, device, use_half_precision


def create_camera(config: Config) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")
    return cap


def create_hand_landmarker(config: Config) -> mp_vision.HandLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=config.hand_model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def create_window(config: Config) -> None:
    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.window_name, config.display_width, config.display_height)


# ==========================================================
# UTILITÁRIOS
# ==========================================================
def normalize_point(px: float, py: float, w: int, h: int) -> tuple[float, float]:
    """Converte coordenadas em pixels para o intervalo [-1, 1]."""
    norm_x = (px / w) * 2 - 1
    norm_y = ((1 - py / h) * 2) - 1
    return norm_x, norm_y


# ==========================================================
# DETECÇÃO DE PESSOAS (YOLO)
# ==========================================================
def detect_people(
    frame: np.ndarray, model: YOLO, device: str, use_half: bool, config: Config
) -> list[Person]:
    """Detecta pessoas no frame e retorna cada uma com bbox e centro normalizado.

    Filtrar por classe (`classes=`) e confiança (`conf=`) direto na chamada do
    YOLO evita rodar NMS/pós-processamento para classes que não usamos e
    remove o laço de filtragem manual que existia antes em Python.
    """
    h, w = frame.shape[:2]
    results = model(
        frame,
        verbose=False,
        conf=config.yolo_conf_threshold,
        classes=[config.yolo_person_class_id],
        device=device,
        half=use_half,
    )

    people = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            people.append(
                Person(
                    bbox=(x1, y1, x2, y2),
                    center=normalize_point(center_x, center_y, w, h),
                )
            )
    return people


def draw_people(frame: np.ndarray, people: list[Person]) -> None:
    for person in people:
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)


# ==========================================================
# DETECÇÃO DE MÃOS (MEDIAPIPE)
# ==========================================================
def calculate_hand_openness(landmarks, config: Config) -> float:
    """Calcula quão aberta está a mão (0.0 = Fechada / Punho, 1.0 = Totalmente Aberta).

    A referência de escala (hand_size) usa o MCP do dedo médio (landmark 9),
    que fica praticamente fixo independente da mão estar aberta ou fechada —
    diferente de usar a ponta de um dedo, que também entraria na média abaixo
    e distorceria o resultado.
    """
    wrist = landmarks[WRIST_ID]
    hand_size = math.hypot(
        landmarks[MIDDLE_MCP_ID].x - wrist.x, landmarks[MIDDLE_MCP_ID].y - wrist.y
    )
    if hand_size == 0:
        return 0.0

    avg_tip_dist = sum(
        math.hypot(landmarks[idx].x - wrist.x, landmarks[idx].y - wrist.y)
        for idx in FINGER_TIP_IDS
    ) / len(FINGER_TIP_IDS)

    raw_ratio = avg_tip_dist / hand_size

    if config.debug_openness:
        print(f"raw_ratio={raw_ratio:.3f}")

    openness = (raw_ratio - config.ratio_closed) / (config.ratio_open - config.ratio_closed)
    return float(np.clip(openness, 0.0, 1.0))


def draw_hand_skeleton(frame, points, color_point, color_line) -> None:
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], color_line, 2)
    for px, py in points:
        cv2.circle(frame, (px, py), 4, color_point, -1)


def get_handedness(raw_handedness: str) -> str:
    """Inverte a lateralidade pois o frame é espelhado antes da detecção."""
    return "Right" if raw_handedness == "Left" else "Left"


def find_owner_person(people: list[Person], wrist_x: int, wrist_y: int) -> Optional[Person]:
    for person in people:
        x1, y1, x2, y2 = person.bbox
        if x1 <= wrist_x <= x2 and y1 <= wrist_y <= y2:
            return person
    return None


def process_hands(frame, hand_result, people: list[Person], config: Config) -> None:
    if not hand_result or not hand_result.hand_landmarks:
        return

    h, w = frame.shape[:2]

    for idx, landmarks in enumerate(hand_result.hand_landmarks):
        raw_handedness = hand_result.handedness[idx][0].display_name
        handedness = get_handedness(raw_handedness)

        wrist_x = int(landmarks[WRIST_ID].x * w)
        wrist_y = int(landmarks[WRIST_ID].y * h)
        norm_x, norm_y = normalize_point(wrist_x, wrist_y, w, h)

        openness = calculate_hand_openness(landmarks, config)

        is_left = handedness == "Left"
        point_color = LEFT_HAND_COLOR_POINT if is_left else RIGHT_HAND_COLOR_POINT
        line_color = LEFT_HAND_COLOR_LINE if is_left else RIGHT_HAND_COLOR_LINE

        pt_coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        draw_hand_skeleton(frame, pt_coords, point_color, line_color)

        cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"{int(openness * 100)}%",
            (wrist_x - 20, wrist_y - 15),
            FONT,
            0.5,
            (0, 255, 255),
            2,
        )

        owner = find_owner_person(people, wrist_x, wrist_y)
        if owner is None:
            continue

        hand_info = HandInfo(norm_x, norm_y, openness)
        if is_left:
            owner.left_hand = hand_info
        else:
            owner.right_hand = hand_info


# ==========================================================
# OSC
# ==========================================================
def _hand_or_default(hand: Optional[HandInfo]) -> tuple[float, float, float]:
    if hand is None:
        return -2.0, -2.0, -1.0
    return hand.x, hand.y, hand.openness


def send_osc_data(client: SimpleUDPClient, people: list[Person]) -> None:
    client.send_message("/people/count", len(people))

    osc_people = []
    for person in people:
        px, py = person.center
        lx, ly, lo = _hand_or_default(person.left_hand)
        rx, ry, ro = _hand_or_default(person.right_hand)
        osc_people.extend([px, py, lx, ly, lo, rx, ry, ro])

    client.send_message("/people", osc_people)


def update_rain_value(state: AppState, person_count: int, config: Config) -> None:
    target = np.clip(person_count / config.max_people, 0.0, 1.0)
    state.rain_value += (target - state.rain_value) * config.lerp_speed


# ==========================================================
# HUD
# ==========================================================
def draw_hud(frame, people: list[Person], person_count: int, state: AppState) -> None:
    cv2.putText(frame, f"Pessoas: {person_count}", (20, 40), FONT, 1, (0, 255, 0), 2)
    cv2.putText(
        frame, f"Chuva: {int(state.rain_value * 100)}%", (20, 80), FONT, 1, (255, 255, 255), 2
    )

    if state.test_mode:
        cv2.putText(frame, "MODO TESTE", (20, 120), FONT, 1, (0, 0, 255), 2)

    y_text = 170
    cv2.putText(frame, "Posicoes & Gestos:", (20, y_text), FONT, 0.7, (255, 255, 0), 2)
    y_text += 30

    for i, person in enumerate(people):
        px, py = person.center
        lh_str = f"L:{int(person.left_hand.openness * 100)}%" if person.left_hand else "L:Nao"
        rh_str = f"R:{int(person.right_hand.openness * 100)}%" if person.right_hand else "R:Nao"

        cv2.putText(
            frame,
            f"P{i + 1}: ({px:.2f}, {py:.2f}) | {lh_str} | {rh_str}",
            (20, y_text),
            FONT,
            0.5,
            (255, 255, 0),
            1,
        )
        y_text += 25


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
def main() -> None:
    config = CONFIG
    ensure_hand_model(config)

    client = SimpleUDPClient(config.osc_ip, config.osc_port)
    yolo_model, device, use_half = create_yolo_model(config)
    cap = create_camera(config)
    create_window(config)
    state = AppState()

    print("Pressione 'q' para sair")
    print("Pressione 't' para alternar modo teste")

    start_time = time.time()

    with create_hand_landmarker(config) as landmarker:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                people = detect_people(frame, yolo_model, device, use_half, config)
                draw_people(frame, people)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((time.time() - start_time) * 1000)
                hand_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                process_hands(frame, hand_result, people, config)

                send_osc_data(client, people)

                person_count = config.max_people if state.test_mode else len(people)
                update_rain_value(state, person_count, config)
                client.send_message("/construcao", float(state.rain_value))

                draw_hud(frame, people, person_count, state)

                frame_show = cv2.resize(frame, (config.display_width, config.display_height))
                cv2.imshow(config.window_name, frame_show)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("t"), ord("T")):
                    state.test_mode = not state.test_mode
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()