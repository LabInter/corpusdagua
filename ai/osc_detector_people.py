import cv2
import numpy as np
from pythonosc import udp_client

# --- CONFIGURAÇÕES OSC ---
OSC_IP = "127.0.0.1" 
OSC_PORT = 8001  
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# --- VARIÁVEIS DE CONTAGEM ---
total_people_passed = 0 
last_person_count = 0 

# --- CARREGAMENTO DO MODELO ---
try:
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
except cv2.error:
    print("Erro: Arquivos do modelo não encontrados.")
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"Iniciando Teste de Contagem. Enviando para {OSC_IP}:{OSC_PORT}")
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret: break

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
                person_count += 1
                
                # Coordenadas e cálculo da posição
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Normalização (0.0 a 1.0)
                normalizedX = 1.0 - ((startX + endX) / 2 / w)
                
                # ENVIO OSC: Posição X (Aciona o CASE /x na Unreal)
                # Forçamos o tipo float para o nó 'Get OSC Message Float' da Unreal não falhar
                if person_count == 1:
                    print(f"Enviado para Unreal: X Normalizado = {normalizedX:.2f}")
                    client.send_message("/x", float(normalizedX))

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Lógica de Contagem Acumulada
    if person_count > last_person_count:
        total_people_passed += (person_count - last_person_count)
        
        # ENVIO OSC: Total (Aciona o CASE /people/total na Unreal)
        # Enviamos como int para o nó 'Get OSC Message Integer'
        client.send_message("/people/total", int(total_people_passed))
        print(f"Enviado para Unreal: Total = {total_people_passed}")

    last_person_count = person_count 

    # UI do OpenCV
    cv2.putText(frame, f"Pessoas Agora: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Geral: {total_people_passed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    cv2.imshow("Teste OSC - Apenas Pessoas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()