import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

# --- CONFIGURAÇÕES OSC ---
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
client = SimpleUDPClient(OSC_IP, OSC_PORT)

# --- CONFIGURAÇÕES DE LÓGICA ---
MAX_PEOPLE = 5      # 1 pessoa = 0.2 (pouca chuva) | 5 pessoas = 1.0 (chuva intensa)
LERP_SPEED = 0.03   # Transição suave

# --- ESTADO ---
valor_atual = 0.0
valor_alvo = 0.0
modo_teste = False  # Variável para forçar 5 pessoas

# --- CARREGAR MODELO ---
try:
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
except:
    print("ERRO: Arquivos do modelo não encontrados!")
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)

print("COMANDOS:")
print("'q' - Sair")
print("'t' - Alternar Modo Teste (Simular 5 pessoas)")

while True:
    ret, frame = cap.read()
    if not ret: break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    current_frame_people = 0

    # Contagem real pela câmera
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                current_frame_people += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # --- LÓGICA DE TESTE ---
    # Se o modo teste estiver ativo, ignoramos a câmera e forçamos 5 pessoas
    pessas_para_calculo = 5 if modo_teste else current_frame_people
    
    valor_alvo = min(pessas_para_calculo / MAX_PEOPLE, 1.0)

    # Suavização e envio
    if abs(valor_alvo - valor_atual) > 0.001:
        valor_atual += (valor_alvo - valor_atual) * LERP_SPEED
        client.send_message("/construcao", float(valor_atual))

    # --- UI ---
    status = "TESTE ATIVO (Simulando 5)" if modo_teste else "Monitorando Real"
    cv2.putText(frame, f"Modo: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Chuva: {int(valor_atual*100)}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Detector para Unreal", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("Saindo...")
        break
    elif key == ord('t') or key == ord('T'):
        print("Alternando Modo Teste...")
        modo_teste = not modo_teste # Liga/Desliga o teste ao apertar 'T'

cap.release()
cv2.destroyAllWindows()