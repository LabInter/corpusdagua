from pythonosc.udp_client import SimpleUDPClient
import tkinter as tk


OSC_IP = "192.168.0.8"
OSC_PORT = 8001
OSC_ADDRESS = "/construcao"


client = SimpleUDPClient(OSC_IP, OSC_PORT)


def on_slider_change(value: str):
    """Callback do slider: envia valor normalizado (0–1) via OSC."""
    v = float(value) / 100.0
    value_label.config(text=f"Valor: {v:.2f}")
    client.send_message(OSC_ADDRESS, v)


root = tk.Tk()
root.title("Controle OSC 0–1")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

title_label = tk.Label(frame, text="Slider OSC (/construcao)", font=("Segoe UI", 12, "bold"))
title_label.pack(pady=(0, 10))

slider = tk.Scale(
    frame,
    from_=0,
    to=100,
    orient=tk.HORIZONTAL,
    length=300,
    command=on_slider_change,
)
slider.pack()

value_label = tk.Label(frame, text="Valor: 0.00", font=("Segoe UI", 10))
value_label.pack(pady=(10, 0))

root.mainloop()
