
from pythonosc import udp_client
import random
import time

OSC_IP = "10.0.0.42"
OSC_PORT = 8001
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
count_list = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

while True:
    count = float(input('Digite a quantidade de pessoas:'))
    client.send_message("/body/count", count)
    print("/body/count", count)
    #time.sleep(2)