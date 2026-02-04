import random
import time
from pythonosc import udp_client

OSC_IP = "10.0.0.131"  
OSC_PORT = 7400
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

NUM_PARTICLES = 50

minX = 0
maxX = 127

while True:
    for i in range(NUM_PARTICLES):
        x = random.uniform(minX, maxX)
        y = random.uniform(minX, maxX)
        z = random.uniform(-1.0, 1.0)
        client.send_message(f"/x", x)
        client.send_message(f"/y", y)
    
    time.sleep(0.9)
