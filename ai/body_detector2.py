import cv2
import numpy as np
from pythonosc import udp_client
import mediapipe as mp # --- ### NOVO: Importa a biblioteca MediaPipe ### ---

# --- CONFIGURAÇÕES OSC ---
OSC_IP = "10.0.0.42" # IP do computador rodando a Unreal. Mantenha 127.0.0.1 se for o mesmo PC.
OSC_PORT = 8001  # Porta que a Unreal estará escutando.
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
client2 = udp_client.SimpleUDPClient("10.0.0.42", 8002)

# --- ### NOVO: CONTAGEM DE PASSAGEM DE PESSOAS ### ---
total_people_passed = 0 # Variável para a contagem total acumulada
last_person_count = 0     # Variável para armazenar a contagem do quadro anterior
# --- ### FIM DA CONTAGEM DE PASSAGEM DE PESSOAS ### ---

# --- ### NOVO: INICIALIZAÇÃO DO MEDIAPIPE HANDS ### ---
mp_hands = mp.solutions.hands
# Configura o detector de mãos. max_num_hands=2 para detectar até duas mãos.
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # Utilitário para desenhar os landmarks
# --- ### FIM DA INICIALIZAÇÃO DO MEDIAPIPE ### ---

# Carrega o modelo de detecção de objetos pré-treinado (MobileNet SSD)
try:
     net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
except cv2.error as e:
     print("Erro ao carregar o modelo. Verifique se os arquivos 'MobileNetSSD_deploy.prototxt.txt' e 'MobileNetSSD_deploy.caffemodel' estão na mesma pasta do script.")
     exit()

# Lista de classes que o modelo pode detectar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

# Inicia a captura de vídeo da câmera
cap = cv2.VideoCapture(0)

# Define a resolução da câmera para 1280x720 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Iniciando detecção. Pressione 'q' para sair.")

while True:
     # Lê um quadro da câmera
     ret, frame = cap.read()
     if not ret:
          print("Erro: Não foi possível capturar o quadro da câmera.")
          break

     # Obtém as dimensões do quadro
     (h, w) = frame.shape[:2]

     # --- INÍCIO DA DETECÇÃO DE PESSOAS (CÓDIGO ORIGINAL) ---
     # Cria um blob a partir do quadro para a rede neural
     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

     # Passa o blob pela rede e obtém as detecções
     net.setInput(blob)
     detections = net.forward()

     # Inicializa o contador de pessoas para este quadro
     person_count = 0

     # Itera sobre as detecções
     for i in np.arange(0, detections.shape[2]):
          confidence = detections[0, 0, i, 2]

          if confidence > 0.5:
               idx = int(detections[0, 0, i, 1])

               if CLASSES[idx] == "person":
                    # Extrai as coordenadas da caixa delimitadora
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Calcula o centro da pessoa
                    centerX = (startX + endX) / 2
                    centerY = (startY + endY) / 2

                    # Normaliza as coordenadas (de 0.0 a 1.0)
                    normalizedX = 1.0 - (centerX / w)
                    normalizedY = centerY / h

                    osc_address = f"/person/{person_count}/position"
                    client.send_message(osc_address, [normalizedX, normalizedY])
                    # print(f"/person/{person_count}/position", [normalizedX, normalizedY]) # Comentado para evitar poluir o console

                    # --- ### NOVO: CÁLCULO DO Z (PROFUNDIDADE ESTIMADA) ### ---
                    person_height_pixels = endY - startY
                    MAX_HEIGHT_PIXELS = 650
                    MIN_HEIGHT_PIXELS = 100
                    z_inverse = (person_height_pixels - MIN_HEIGHT_PIXELS) / (MAX_HEIGHT_PIXELS - MIN_HEIGHT_PIXELS)
                    z_inverse = np.clip(z_inverse, 0.0, 1.0)
                    z = 1.0 - z_inverse
                    # --- ### FIM DO CÁLCULO DO Z ### ---

                    if person_count == 0:
                         client2.send_message("/x", normalizedX)
                         client2.send_message("/y", normalizedY)
                         client2.send_message("/z", z)

                    # Incrementa o contador para a próxima pessoa e para o total
                    person_count += 1

                    # --- Lógica de visualização (não afeta o OSC) ---
                    label = "Pessoa {}: {:.2f}%".format(person_count, confidence * 100) # Mostra ID 1, 2, 3...
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (int(centerX), int(centerY)), 5, (0, 0, 255), -1)

     # --- ### NOVO: CONTAGEM DE PASSAGEM DE PESSOAS ### ---
     # A lógica de contagem é: se o número atual de pessoas for maior 
     # que o número anterior (indicando que uma nova pessoa entrou na cena), 
     # incrementamos a contagem total.

     if person_count > last_person_count:
          # Uma nova pessoa foi detectada neste quadro que não estava no anterior
          total_people_passed += person_count - last_person_count
          client.send_message("/people/total", total_people_passed)
          print(f"Nova pessoa detectada! Total de pessoas que passaram: {total_people_passed}")

     # Atualiza a contagem anterior para o próximo loop
     last_person_count = person_count 

     # Exibe a contagem total de pessoas no quadro (agora em duas linhas)
     cv2.putText(frame, f"Pessoas Detectadas AGORA: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
     cv2.putText(frame, f"TOTAL que Passaram: {total_people_passed}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
     # --- ### FIM DA CONTAGEM DE PASSAGEM DE PESSOAS ### ---

     # --- FIM DA DETECÇÃO DE PESSOAS ---


     # --- ### NOVO: DETECÇÃO DE MÃOS COM MEDIAPIPE ### ---
     # 1. Converte a imagem de BGR (OpenCV) para RGB (MediaPipe)
     # --- ### NOVO: DETECÇÃO DE MÃOS COM MEDIAPIPE (COM ID FIXO) ### ---
     # 1. Converte a imagem de BGR (OpenCV) para RGB (MediaPipe)
     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # 2. Processa a imagem e detecta as mãos
     results = hands.process(rgb_frame)
     
     # 3. Se mãos forem detectadas, itera sobre elas
     if results.multi_hand_landmarks:
          
          # --- ### ALTERAÇÃO PRINCIPAL ### ---
          # Iteramos usando um índice para poder acessar 
          # tanto os landmarks (coordenadas) quanto o handedness (esquerda/direita)
          for i in range(len(results.multi_hand_landmarks)):
               
               # Pega os landmarks (pontos da mão)
               hand_landmarks = results.multi_hand_landmarks[i]
               # Pega a classificação (Left/Right)
               hand_handedness = results.multi_handedness[i]
               
               # Extrai a classificação (string 'Left' ou 'Right')
               # O [0] é para pegar a classificação principal (a de maior score)
               label = hand_handedness.classification[0].label
               
               # --- Lógica para encontrar o centro da mão (igual ao seu código) ---
               x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
               y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
               
               x_min, x_max = int(min(x_coords)), int(max(x_coords))
               y_min, y_max = int(min(y_coords)), int(max(y_coords))
               
               hand_center_x = (x_min + x_max) // 2
               hand_center_y = (y_min + y_max) // 2
               
               normalized_hand_x = hand_center_x / w
               normalized_hand_y = hand_center_y / h
               
               # --- ### AQUI ESTÁ A LÓGICA DE ID FIXO ### ---
               # Atribui um ID fixo: 0 para Esquerda, 1 para Direita
               if label == 'Left':
                    hand_id = 0
               elif label == 'Right':
                    hand_id = 1
               else:
                    # Se não for nem left/right (improvável), pula esta mão
                    continue 

               # Envia a mensagem OSC com o ID fixo
               client.send_message(f"/hand/{hand_id}/position", [normalized_hand_x, normalized_hand_y])
               # print(f"/hand/{hand_id}/position ({label})", [normalized_hand_x, normalized_hand_y]) # Comentado para evitar poluir o console
               
               # --- Lógica de visualização (Atualizada para mostrar o ID correto) ---
               cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 255), 2)
               cv2.circle(frame, (hand_center_x, hand_center_y), 7, (255, 255, 0), -1)
               
               # Define o texto para mostrar "Mao Esquerda (0)" ou "Mao Direita (1)"
               display_text = f"Mao {label} ({hand_id})"
               cv2.putText(frame, display_text, (x_min - 20, y_min - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

     # --- ### FIM DA DETECÇÃO DE MÃOS ### ---
    

     # Mostra o quadro resultante com ambas as detecções
     cv2.imshow("Detector de Pessoas e Maos", frame)

     # Aguarda a tecla 'q' ser pressionada para sair
     if cv2.waitKey(1) & 0xFF == ord('q'):
          break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
print("Detecção finalizada.")