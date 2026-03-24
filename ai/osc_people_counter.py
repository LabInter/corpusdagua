import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

# --- CONFIGURAÇÕES OSC ---
OSC_IP = "10.0.0.42"
OSC_PORT = 8001
client = SimpleUDPClient(OSC_IP, OSC_PORT)

# --- CONFIGURAÇÕES DE CONTAGEM ---
MAX_PEOPLE = 40

# --- CONFIGURAÇÕES DE SUAVIZAÇÃO ---
# Velocidade da interpolação: 0.0 (parado) → 1.0 (instantâneo)
# Valores baixos = transição mais lenta e suave (ex: 0.02 = ~2% por frame)
LERP_SPEED = 0.03

# --- ESTADO ---
total_people_passed = 0
last_person_count = 0
valor_atual = 0.0   # valor que está sendo enviado agora (suavizado)
valor_alvo = 0.0    # valor que queremos atingir

# --- MODELO ---
try:
    net = cv2.dnn.readNetFromCaffe(
        'MobileNetSSD_deploy.prototxt.txt',
        'MobileNetSSD_deploy.caffemodel'
    )
except cv2.error:
    print("Erro ao carregar o modelo. Verifique se os arquivos estão na mesma pasta.")
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Iniciando contagem de pessoas (modo suave). Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar quadro.")
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_count = 0

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"Pessoa {person_count + 1}: {confidence * 100:.1f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                person_count += 1

    # --- CONTAGEM DE PASSAGEM ---
    if person_count > last_person_count:
        total_people_passed += person_count - last_person_count
        valor_alvo = min(total_people_passed / MAX_PEOPLE, 1.0)
        print(f"Total passaram: {total_people_passed} → alvo /construcao = {valor_alvo:.2f}")

    last_person_count = person_count

    # --- SUAVIZAÇÃO (lerp por frame) ---
    # Interpola o valor atual em direção ao alvo a cada frame
    if abs(valor_alvo - valor_atual) > 0.001:
        valor_atual += (valor_alvo - valor_atual) * LERP_SPEED
        client.send_message("/construcao", float(valor_atual))

    # --- VISUALIZAÇÃO ---
    cv2.putText(frame, f"Agora: {person_count} pessoa(s)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Total passaram: {total_people_passed}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"/construcao: {valor_atual:.3f} → {valor_alvo:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

    cv2.imshow("Contador de Pessoas - OSC (Suave)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Finalizado. Total de pessoas que passaram: {total_people_passed}")
