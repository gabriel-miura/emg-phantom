import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
import threading
import csv
import time
import numpy as np
import serial.tools.list_ports



SERIAL_PORT = [p.device for p in serial.tools.list_ports.comports()][0]
BAUD_RATE = 115200
N = 10000

plot_y = np.zeros(N)
is_recording = False
recorded_data = []
current_label = "1"
t0 = time.time()

# ── Thread Serial ──
def serial_worker():
    global is_recording, plot_y
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.flushInput()
        print(f"[OK] Conectado em {SERIAL_PORT}")
        while True:
            raw = ser.readline()
            if raw:
                try:
                    val = float(raw.decode('utf-8', errors='ignore').strip())
                    plot_y = np.roll(plot_y, -1)
                    plot_y[-1] = val
                    if is_recording:
                        recorded_data.append([time.time() - t0, val, current_label])
                except ValueError:
                    pass
    except Exception as e:
        print(f"[ERRO] Serial: {e}")

threading.Thread(target=serial_worker, daemon=True).start()

# ── Interface ──
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
fig.subplots_adjust(bottom=0.25)
ax.set_facecolor('#0d1117')
ax.set_xlim(0, N)
ax.set_ylim(-0.001, 0.005)
ax.grid(True, alpha=0.1)

line, = ax.plot(np.arange(N), plot_y, color='#00FFCC', lw=1, animated=True)
status_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='gray', fontfamily='monospace')

btn_rec = Button(fig.add_axes([0.1, 0.05, 0.15, 0.07]), 'GRAVAR', color='#161b22', hovercolor='#1c2d22')
btn_rec.label.set_color('#3fb950')
text_box = TextBox(fig.add_axes([0.35, 0.05, 0.15, 0.07]), 'Gesto: ', initial=current_label, color='#161b22')
btn_save = Button(fig.add_axes([0.6, 0.05, 0.15, 0.07]), 'SALVAR CSV', color='#161b22', hovercolor='#141d2b')

def toggle_rec(event):
    global is_recording
    is_recording = not is_recording
    btn_rec.label.set_text('PARAR' if is_recording else 'GRAVAR')
    btn_rec.label.set_color('#f85149' if is_recording else '#3fb950')

def save_csv(event):
    if not recorded_data: return
    fname = f"1_DADOS.csv"
    with open(fname, 'w', newline='') as f:
        csv.writer(f).writerows([['timestamp', 'valor', 'label']] + recorded_data)
    print(f"[OK] Salvo: {fname}")

btn_rec.on_clicked(toggle_rec)
btn_save.on_clicked(save_csv)
text_box.on_submit(lambda t: globals().update(current_label=t))

def update(frame):
    line.set_ydata(plot_y)
    state = "● REC" if is_recording else "○ IDLE"
    status_text.set_text(f"{state} | Gesto: {current_label} | Amostras: {len(recorded_data)}")
    return line, status_text

ani = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
plt.show()
