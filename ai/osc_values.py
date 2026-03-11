# Exemplo Python com python-osc
from pythonosc.udp_client import SimpleUDPClient
import time

client = SimpleUDPClient("192.168.0.8", 8001)

# Envia valores de 0 até 1.0 e depois de 1.0 até 0 em loop contínuo
while True:
    # Sobe de 0 a 1.0
    for i in range(0, 101):
        valor = i / 100.0  # 0.00, 0.01, ..., 1.00
        client.send_message("/construcao", valor)
        print(valor)
        time.sleep(0.05)  # ajuste o intervalo conforme necessário

    # Desce de 1.0 a 0
    for i in range(100, -1, -1):
        valor = i / 100.0  # 1.00, 0.99, ..., 0.00
        client.send_message("/construcao", valor)
        print(valor)
        time.sleep(0.05)
