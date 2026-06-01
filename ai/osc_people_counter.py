import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
import mediapipe as mp

# --- CONFIGURAÇÕES OSC ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
client = SimpleUDPClient(OSC_IP, OSC_PORT)

# --- CONFIGURAÇÕES DE LÓGICA ---
MAX_PEOPLE = 5      
LERP_SPEED = 0.05   

# --- CONFIGURAÇÃO DE TAMANHO DA JANELA DA TELA ---
# Altere aqui para o tamanho que deseja visualizar no monitor
LARGURA_EXIBICAO = 1920  
ALTURA_EXIBICAO = 1080

# --- ESTADO ---
valor_atual = 0.0
valor_alvo = 0.0
modo_teste = False  

# --- CONFIGURAR MEDIAPIPE (MÃOS) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,        
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- CARREGAR MODELO DE PESSOAS ---
try:
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
except:
    print("ERRO: Arquivos do modelo MobileNetSSD não encontrados!")
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# --- CONFIGURAR ENTRADA DA CÂMERA ---
cap = cv2.VideoCapture(0)
# Tenta definir a captura nativa para HD (o OpenCV ajusta para o máximo suportado pela webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

# --- CONFIGURAR JANELA INTERATIVA ---
NOME_JANELA = "Detector HD - Correcao Lados"
cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL) # Permite maximizar e redimensionar livremente
cv2.resizeWindow(NOME_JANELA, LARGURA_EXIBICAO, ALTURA_EXIBICAO)

print("COMANDOS:")
print("'q' - Sair")
print("'t' - Alternar Modo Teste (Simular 5 pessoas)")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Espelha o frame para que o movimento seja natural (comportamento de espelho)
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]
    
    # --- 1. DETECÇÃO DE PESSOAS (MobileNetSSD) ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    current_frame_people = 0

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                current_frame_people += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # --- 2. DETECÇÃO E CLASSIFICAÇÃO DAS MÃOS (MediaPipe) ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    maos_detectadas = {"Left": False, "Right": False}

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lado_da_mao = results.multi_handedness[idx].classification[0].label
            
            # Correção de espelhamento: Left mapeia para Left do usuário na tela
            if lado_da_mao == "Left":
                lado_real = "Left"
            else:
                lado_real = "Right"
                
            maos_detectadas[lado_real] = True
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            ponto_central = hand_landmarks.landmark[0]
            
            # Normaliza para o plano cartesiano (-1.0 a 1.0)
            mao_x = float((ponto_central.x * 2.0) - 1.0)
            mao_y = float(((1.0 - ponto_central.y) * 2.0) - 1.0)
            
            # Envia os dados via OSC
            if lado_real == "Left":
                client.send_message("/posicao_mao_esquerda", [mao_x, mao_y])
                cv2.putText(frame, f"Esq X: {mao_x:.2f} Y: {mao_y:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif lado_real == "Right":
                client.send_message("/posicao_mao_direita", [mao_x, mao_y])
                cv2.putText(frame, f"Dir X: {mao_x:.2f} Y: {mao_y:.2f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            px = int(ponto_central.x * w)
            py = int(ponto_central.y * h)
            cor = (255, 0, 0) if lado_real == "Left" else (0, 165, 255)
            cv2.circle(frame, (px, py), 12, cor, -1)

    # --- LÓGICA DE TESTE DA CHUVA ---
    pessoas_para_calculo = 5 if modo_teste else current_frame_people
    valor_alvo = float(np.clip(pessoas_para_calculo / MAX_PEOPLE, 0.0, 1.0))
    valor_atual += (valor_alvo - valor_atual) * LERP_SPEED
    if abs(valor_atual - valor_alvo) < 0.005: valor_atual = valor_alvo

    client.send_message("/construcao", float(valor_atual))

    # --- UI DO DETECTOR ---
    status = "TESTE ATIVO" if modo_teste else "Monitorando Real"
    cv2.putText(frame, f"Modo: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Chuva: {int(valor_atual*100)}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Pessoas: {pessoas_para_calculo}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if not maos_detectadas["Left"]:
        cv2.putText(frame, "Mao Esq: Ausente", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
    if not maos_detectadas["Right"]:
        cv2.putText(frame, "Mao Dir: Ausente", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
        
    # Redimensiona o frame final para o tamanho gigante escolhido antes de exibir
    frame_gigante = cv2.resize(frame, (LARGURA_EXIBICAO, ALTURA_EXIBICAO))
    cv2.imshow(NOME_JANELA, frame_gigante)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'): break
    elif key == ord('t') or key == ord('T'): modo_teste = not modo_teste

cap.release()
cv2.destroyAllWindows()