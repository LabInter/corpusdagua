from pythonosc import dispatcher
from pythonosc import osc_server

# Função callback para tratar partículas
def particle_handler(address, *args):
    print(f"Recebido de {address}: {args}")

# Configuração do servidor OSC
IP = "0.0.0.0"   # escuta em todas as interfaces
PORT = 8000

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/particle/*", particle_handler)

server = osc_server.ThreadingOSCUDPServer((IP, PORT), dispatcher)
print(f"Servidor OSC escutando em {IP}:{PORT}")
server.serve_forever()
