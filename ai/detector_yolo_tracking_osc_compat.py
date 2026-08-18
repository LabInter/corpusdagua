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
    """Configurações da aplicação."""

    # OSC
    osc_ip: str = "127.0.0.1"
    osc_port: int = 8000

    # Pessoas
    max_people: int = 5
    lerp_speed: float = 0.05

    # Câmera / janela
    display_width: int = 1920
    display_height: int = 1080
    capture_width: int = 1280
    capture_height: int = 720
    window_name: str = "Detector Pessoas + Maos"

    # YOLO
    yolo_model_path: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.4
    yolo_person_class_id: int = 0
    yolo_tracker: str = "bytetrack.yaml"

    # Pessoa ativa
    closest_person_switch_ratio: float = 1.15

    # MediaPipe
    hand_model_path: str = "hand_landmarker.task"
    hand_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    max_hands: int = 10

    # Abertura da mão
    ratio_closed: float = 1.0
    ratio_open: float = 1.83
    hand_open_threshold: float = 0.5
    debug_openness: bool = False


CONFIG = Config()


# ==========================================================
# CONSTANTES
# ==========================================================

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

FINGER_TIP_IDS = (8, 12, 16, 20)
WRIST_ID = 0
MIDDLE_MCP_ID = 9

LEFT_HAND_COLOR_POINT = (255, 80, 80)
LEFT_HAND_COLOR_LINE = (200, 50, 50)
RIGHT_HAND_COLOR_POINT = (80, 165, 255)
RIGHT_HAND_COLOR_LINE = (50, 130, 200)

NO_HAND_POSITION = -2.0
NO_HAND_STATE = -1

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ==========================================================
# MODELOS DE DADOS
# ==========================================================

@dataclass
class HandInfo:
    """Informações normalizadas de uma mão."""

    x: float
    y: float
    openness: float


@dataclass
class Person:
    """Informações de uma pessoa detectada."""

    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    track_id: Optional[int] = None
    left_hand: Optional[HandInfo] = None
    right_hand: Optional[HandInfo] = None

    @property
    def area(self) -> int:
        """Calcula a área da bounding box."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class AppState:
    """Estado mutável da aplicação."""

    rain_value: float = 0.0
    test_mode: bool = False
    active_person_id: Optional[int] = None


# ==========================================================
# SETUP
# ==========================================================

def ensure_hand_model(config: Config) -> None:
    """Garante que o modelo do MediaPipe esteja disponível."""
    if os.path.exists(config.hand_model_path):
        return

    print("Baixando modelo hand_landmarker.task...")
    urllib.request.urlretrieve(
        config.hand_model_url,
        config.hand_model_path,
    )
    print("Modelo baixado.")


def create_yolo_model(config: Config) -> tuple[YOLO, str, bool]:
    """Cria o modelo YOLO e configura CPU/GPU."""
    model = YOLO(config.yolo_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"

    return model, device, use_half


def create_camera(config: Config) -> cv2.VideoCapture:
    """Inicializa a câmera."""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)

    if not camera.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    return camera


def create_hand_landmarker(
    config: Config,
) -> mp_vision.HandLandmarker:
    """Inicializa o Hand Landmarker do MediaPipe."""
    base_options = mp_python.BaseOptions(
        model_asset_path=config.hand_model_path,
    )

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
    """Cria a janela de visualização."""
    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        config.window_name,
        config.display_width,
        config.display_height,
    )


# ==========================================================
# UTILITÁRIOS
# ==========================================================

def normalize_point(
    px: float,
    py: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Converte coordenadas de pixels para o intervalo [-1, 1]."""
    x = (px / width) * 2 - 1
    y = ((1 - py / height) * 2) - 1
    return x, y


def get_person_by_id(
    people: list[Person],
    track_id: Optional[int],
) -> Optional[Person]:
    """Localiza uma pessoa pelo ID do tracker."""
    if track_id is None:
        return None

    return next(
        (
            person
            for person in people
            if person.track_id == track_id
        ),
        None,
    )


def find_person_at_point(
    people: list[Person],
    point_x: int,
    point_y: int,
) -> Optional[Person]:
    """Localiza a pessoa que contém determinado ponto."""
    for person in people:
        x1, y1, x2, y2 = person.bbox

        if x1 <= point_x <= x2 and y1 <= point_y <= y2:
            return person

    return None


def get_hand_label(
    hand: Optional[HandInfo],
    threshold: float,
) -> str:
    """Converte a abertura da mão para um rótulo."""
    if hand is None:
        return "Nao"

    return "Aberta" if hand.openness >= threshold else "Fechada"


def get_hand_osc_data(
    hand: Optional[HandInfo],
    threshold: float,
) -> tuple[float, float, int]:
    """
    Retorna os dados da mão no formato OSC.

    Sem mão:
        (-2.0, -2.0, -1)

    Mão fechada:
        (x, y, 0)

    Mão aberta:
        (x, y, 1)
    """
    if hand is None:
        return NO_HAND_POSITION, NO_HAND_POSITION, NO_HAND_STATE

    state = int(hand.openness >= threshold)
    return hand.x, hand.y, state


# ==========================================================
# YOLO — DETECÇÃO E TRACKING
# ==========================================================

def detect_people(
    frame: np.ndarray,
    model: YOLO,
    device: str,
    use_half: bool,
    config: Config,
) -> list[Person]:
    """Detecta pessoas e atribui IDs persistentes via YOLO Tracking."""
    height, width = frame.shape[:2]

    results = model.track(
        frame,
        verbose=False,
        conf=config.yolo_conf_threshold,
        classes=[config.yolo_person_class_id],
        device=device,
        half=use_half,
        persist=True,
        tracker=config.yolo_tracker,
    )

    people: list[Person] = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            people.append(
                create_person_from_box(
                    box,
                    width,
                    height,
                )
            )

    return people


def create_person_from_box(
    box,
    frame_width: int,
    frame_height: int,
) -> Person:
    """Converte uma detecção YOLO em Person."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    center = normalize_point(
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        frame_width,
        frame_height,
    )

    track_id = None

    if box.id is not None:
        track_id = int(box.id[0].item())

    return Person(
        bbox=(x1, y1, x2, y2),
        center=center,
        track_id=track_id,
    )


def select_active_person(
    people: list[Person],
    state: AppState,
    config: Config,
) -> Optional[Person]:
    """
    Seleciona persistentemente a pessoa que controla as mãos.

    A maior bounding box representa a pessoa mais próxima.
    A troca só ocorre quando a nova pessoa é suficientemente maior.
    """
    if not people:
        state.active_person_id = None
        return None

    closest_person = max(people, key=lambda person: person.area)
    active_person = get_person_by_id(
        people,
        state.active_person_id,
    )

    if active_person is None:
        state.active_person_id = closest_person.track_id
        return closest_person

    if closest_person.track_id == active_person.track_id:
        return active_person

    if active_person.area <= 0:
        state.active_person_id = closest_person.track_id
        return closest_person

    area_ratio = closest_person.area / active_person.area

    if area_ratio >= config.closest_person_switch_ratio:
        state.active_person_id = closest_person.track_id
        return closest_person

    return active_person


def draw_people(
    frame: np.ndarray,
    people: list[Person],
    active_person_id: Optional[int],
) -> None:
    """Desenha todas as pessoas e destaca a pessoa ativa."""
    for person in people:
        x1, y1, x2, y2 = person.bbox
        is_active = person.track_id == active_person_id

        color = (0, 255, 255) if is_active else (0, 255, 0)
        thickness = 4 if is_active else 2

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(frame, center, 6, (0, 0, 255), -1)

        track_label = (
            f"ID {person.track_id}"
            if person.track_id is not None
            else "ID ?"
        )

        cv2.putText(
            frame,
            track_label,
            (x1, max(y1 - 10, 20)),
            FONT,
            0.6,
            color,
            2,
        )

        if is_active:
            cv2.putText(
                frame,
                "MAIS PROXIMA",
                (x1, y2 + 25),
                FONT,
                0.6,
                color,
                2,
            )


# ==========================================================
# MEDIAPIPE — DETECÇÃO E PROCESSAMENTO DAS MÃOS
# ==========================================================

def calculate_hand_openness(
    landmarks,
    config: Config,
) -> float:
    """Calcula a abertura normalizada da mão entre 0 e 1."""
    wrist = landmarks[WRIST_ID]

    hand_size = math.hypot(
        landmarks[MIDDLE_MCP_ID].x - wrist.x,
        landmarks[MIDDLE_MCP_ID].y - wrist.y,
    )

    if hand_size == 0:
        return 0.0

    average_tip_distance = sum(
        math.hypot(
            landmarks[index].x - wrist.x,
            landmarks[index].y - wrist.y,
        )
        for index in FINGER_TIP_IDS
    ) / len(FINGER_TIP_IDS)

    raw_ratio = average_tip_distance / hand_size

    if config.debug_openness:
        print(f"raw_ratio={raw_ratio:.3f}")

    calibration_range = config.ratio_open - config.ratio_closed

    if calibration_range == 0:
        return 0.0

    openness = (
        raw_ratio - config.ratio_closed
    ) / calibration_range

    return float(np.clip(openness, 0.0, 1.0))


def get_handedness(raw_handedness: str) -> str:
    """Corrige a lateralidade considerando o frame espelhado."""
    return "Right" if raw_handedness == "Left" else "Left"


def get_hand_colors(
    handedness: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Retorna as cores de desenho da mão."""
    if handedness == "Left":
        return LEFT_HAND_COLOR_POINT, LEFT_HAND_COLOR_LINE

    return RIGHT_HAND_COLOR_POINT, RIGHT_HAND_COLOR_LINE


def draw_hand_skeleton(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    point_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
) -> None:
    """Desenha conexões e pontos do esqueleto da mão."""
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            line_color,
            2,
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            point_color,
            -1,
        )


def draw_hand_status(
    frame: np.ndarray,
    wrist: tuple[int, int],
    openness: float,
    config: Config,
) -> None:
    """Desenha o estado atual da mão."""
    is_open = openness >= config.hand_open_threshold
    label = "Aberta" if is_open else "Fechada"

    cv2.circle(
        frame,
        wrist,
        8,
        (0, 255, 255),
        -1,
    )

    cv2.putText(
        frame,
        f"{label} ({int(openness * 100)}%)",
        (wrist[0] - 40, wrist[1] - 15),
        FONT,
        0.5,
        (0, 255, 255),
        2,
    )


def process_hands(
    frame: np.ndarray,
    hand_result,
    people: list[Person],
    active_person: Optional[Person],
    config: Config,
) -> None:
    """
    Processa somente as mãos pertencentes à pessoa ativa.

    As outras pessoas continuam sendo detectadas e enviadas
    pelo OSC, mas suas mãos não são associadas aos dados OSC.
    """
    if (
        active_person is None
        or not hand_result
        or not hand_result.hand_landmarks
    ):
        return

    height, width = frame.shape[:2]

    for index, landmarks in enumerate(hand_result.hand_landmarks):
        wrist = (
            int(landmarks[WRIST_ID].x * width),
            int(landmarks[WRIST_ID].y * height),
        )

        owner = find_person_at_point(
            people,
            wrist[0],
            wrist[1],
        )

        if owner is None:
            continue

        if owner.track_id != active_person.track_id:
            continue

        handedness = get_handedness(
            hand_result.handedness[index][0].display_name,
        )

        openness = calculate_hand_openness(
            landmarks,
            config,
        )

        normalized_x, normalized_y = normalize_point(
            wrist[0],
            wrist[1],
            width,
            height,
        )

        points = [
            (
                int(landmark.x * width),
                int(landmark.y * height),
            )
            for landmark in landmarks
        ]

        point_color, line_color = get_hand_colors(handedness)

        draw_hand_skeleton(
            frame,
            points,
            point_color,
            line_color,
        )

        draw_hand_status(
            frame,
            wrist,
            openness,
            config,
        )

        hand_info = HandInfo(
            x=normalized_x,
            y=normalized_y,
            openness=openness,
        )

        if handedness == "Left":
            active_person.left_hand = hand_info
        else:
            active_person.right_hand = hand_info


# ==========================================================
# OSC
# ==========================================================

def send_hand_event(
    client: SimpleUDPClient,
    person_index: int,
    hand: Optional[HandInfo],
    side: str,
    config: Config,
) -> None:
    """
    Envia as rotas de evento da mão.

    Rotas mantidas:
        /people/{i}/mao_esquerda_aberta
        /people/{i}/mao_esquerda_fechada
        /people/{i}/mao_direita_aberta
        /people/{i}/mao_direita_fechada
    """
    if hand is None:
        return

    state = (
        "aberta"
        if hand.openness >= config.hand_open_threshold
        else "fechada"
    )

    route = f"/people/{person_index}/mao_{side}_{state}"
    client.send_message(route, 1)


def send_osc_data(
    client: SimpleUDPClient,
    people: list[Person],
    config: Config,
) -> None:
    """
    Envia os dados mantendo o contrato OSC do Unreal.

    Rotas:
        /people/count
        /people/{i}
        /people/{i}/mao_esquerda_aberta
        /people/{i}/mao_esquerda_fechada
        /people/{i}/mao_direita_aberta
        /people/{i}/mao_direita_fechada
        /people
    """
    ordered_people = sorted(
        people,
        key=lambda person: person.center[0],
    )

    # ------------------------------------------------------
    # /people/count
    # ------------------------------------------------------
    client.send_message(
        "/people/count",
        len(ordered_people),
    )

    # ------------------------------------------------------
    # /people/{i} e /people
    # ------------------------------------------------------
    osc_people: list[float | int] = []

    for index, person in enumerate(ordered_people, start=1):
        px, py = person.center

        left_data = get_hand_osc_data(
            person.left_hand,
            config.hand_open_threshold,
        )

        right_data = get_hand_osc_data(
            person.right_hand,
            config.hand_open_threshold,
        )

        left_x, left_y, left_state = left_data
        right_x, right_y, right_state = right_data

        # Formato original da rota /people:
        #
        # px, py,
        # left_x, left_y, left_state,
        # right_x, right_y, right_state
        osc_people.extend(
            [
                float(px),
                float(py),
                float(left_x),
                float(left_y),
                int(left_state),
                float(right_x),
                float(right_y),
                int(right_state),
            ]
        )

        # Rota individual da pessoa.
        client.send_message(
            f"/people/{index}",
            [float(px), float(py)],
        )

        # Eventos das mãos.
        send_hand_event(
            client,
            index,
            person.left_hand,
            "esquerda",
            config,
        )

        send_hand_event(
            client,
            index,
            person.right_hand,
            "direita",
            config,
        )

    # Payload agregado.
    client.send_message(
        "/people",
        osc_people,
    )


# ==========================================================
# ESTADO DA APLICAÇÃO
# ==========================================================

def update_rain_value(
    state: AppState,
    person_count: int,
    config: Config,
) -> None:
    """Atualiza suavemente o valor da chuva."""
    target = float(
        np.clip(
            person_count / config.max_people,
            0.0,
            1.0,
        )
    )

    state.rain_value += (
        target - state.rain_value
    ) * config.lerp_speed


# ==========================================================
# HUD
# ==========================================================

def draw_hud(
    frame: np.ndarray,
    people: list[Person],
    person_count: int,
    state: AppState,
    config: Config,
) -> None:
    """Exibe informações de depuração."""
    draw_basic_hud(
        frame,
        person_count,
        state,
    )

    draw_active_person_hud(
        frame,
        state,
    )

    draw_people_hud(
        frame,
        people,
        state,
        config,
    )


def draw_basic_hud(
    frame: np.ndarray,
    person_count: int,
    state: AppState,
) -> None:
    """Desenha contagem, chuva e modo de teste."""
    cv2.putText(
        frame,
        f"Pessoas: {person_count}",
        (20, 40),
        FONT,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Chuva: {int(state.rain_value * 100)}%",
        (20, 80),
        FONT,
        1,
        (255, 255, 255),
        2,
    )

    if state.test_mode:
        cv2.putText(
            frame,
            "MODO TESTE",
            (20, 120),
            FONT,
            1,
            (0, 0, 255),
            2,
        )


def draw_active_person_hud(
    frame: np.ndarray,
    state: AppState,
) -> None:
    """Exibe a pessoa atualmente responsável pelas mãos."""
    if state.active_person_id is None:
        text = "Pessoa ativa: nenhuma"
    else:
        text = f"Pessoa ativa: ID {state.active_person_id}"

    cv2.putText(
        frame,
        text,
        (20, 155),
        FONT,
        0.7,
        (0, 255, 255),
        2,
    )


def draw_people_hud(
    frame: np.ndarray,
    people: list[Person],
    state: AppState,
    config: Config,
) -> None:
    """Exibe posição e estado das mãos de cada pessoa."""
    y_text = 195

    cv2.putText(
        frame,
        "Posicoes & Gestos:",
        (20, y_text),
        FONT,
        0.7,
        (255, 255, 0),
        2,
    )

    y_text += 30

    for index, person in enumerate(people, start=1):
        left_label = get_hand_label(
            person.left_hand,
            config.hand_open_threshold,
        )

        right_label = get_hand_label(
            person.right_hand,
            config.hand_open_threshold,
        )

        active_marker = (
            " [ATIVA]"
            if person.track_id == state.active_person_id
            else ""
        )

        track_label = (
            f"ID:{person.track_id}"
            if person.track_id is not None
            else "ID:?"
        )

        text = (
            f"P{index} ({track_label}){active_marker}: "
            f"({person.center[0]:.2f}, {person.center[1]:.2f}) | "
            f"L:{left_label} | R:{right_label}"
        )

        cv2.putText(
            frame,
            text,
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

def process_frame(
    frame: np.ndarray,
    yolo_model: YOLO,
    device: str,
    use_half: bool,
    landmarker: mp_vision.HandLandmarker,
    state: AppState,
    config: Config,
    start_time: float,
) -> list[Person]:
    """Executa todo o processamento de um frame."""
    people = detect_people(
        frame,
        yolo_model,
        device,
        use_half,
        config,
    )

    active_person = select_active_person(
        people,
        state,
        config,
    )

    draw_people(
        frame,
        people,
        state.active_person_id,
    )

    hand_result = detect_hands(
        frame,
        landmarker,
        start_time,
    )

    process_hands(
        frame,
        hand_result,
        people,
        active_person,
        config,
    )

    return people


def detect_hands(
    frame: np.ndarray,
    landmarker: mp_vision.HandLandmarker,
    start_time: float,
):
    """Executa a detecção de mãos no frame atual."""
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    return landmarker.detect_for_video(
        mp_image,
        timestamp_ms,
    )


def handle_keyboard(
    state: AppState,
    key: int,
) -> bool:
    """
    Processa comandos do teclado.

    Retorna True quando o programa deve finalizar.
    """
    if key in (ord("q"), ord("Q")):
        return True

    if key in (ord("t"), ord("T")):
        state.test_mode = not state.test_mode

    return False


def main() -> None:
    """Inicializa e executa a aplicação."""
    config = CONFIG

    ensure_hand_model(config)

    osc_client = SimpleUDPClient(
        config.osc_ip,
        config.osc_port,
    )

    yolo_model, device, use_half = create_yolo_model(config)
    camera = create_camera(config)

    create_window(config)

    state = AppState()
    start_time = time.time()

    print("Pressione 'q' para sair")
    print("Pressione 't' para alternar modo teste")
    print(f"YOLO Tracking: {config.yolo_tracker}")
    print(
        "Fator de troca da pessoa ativa: "
        f"{config.closest_person_switch_ratio}"
    )
    print("Rotas OSC mantidas para compatibilidade com o Unreal.")

    with create_hand_landmarker(config) as landmarker:
        try:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Não foi possível capturar o frame.")
                    break

                frame = cv2.flip(frame, 1)

                people = process_frame(
                    frame,
                    yolo_model,
                    device,
                    use_half,
                    landmarker,
                    state,
                    config,
                    start_time,
                )

                person_count = (
                    config.max_people
                    if state.test_mode
                    else len(people)
                )

                send_osc_data(
                    osc_client,
                    people,
                    config,
                )

                update_rain_value(
                    state,
                    person_count,
                    config,
                )

                draw_hud(
                    frame,
                    people,
                    person_count,
                    state,
                    config,
                )

                display_frame = cv2.resize(
                    frame,
                    (
                        config.display_width,
                        config.display_height,
                    ),
                )

                cv2.imshow(
                    config.window_name,
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if handle_keyboard(state, key):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()