import numpy as np
import tensorflow as tf
import joblib

def load_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def predict_function(data):
        input_data = np.array(data, dtype=np.float32) # formato float32
        if len(input_data.shape) == 2: # shape=(1, WINDOW_SIZE, 1)
            input_data = np.expand_dims(input_data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data) # define tensor de entrada
        interpreter.invoke() # inferência
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    return predict_function

loaded_scalers = joblib.load('2_SCALERS.pkl')
model_predict = load_tflite("2_MODELO.tflite")
TEST_PATIENT = 1



import pandas as pd
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections
import numpy as np
import threading
import time
import serial.tools.list_ports

# --- Configurações de Engenharia ---
WINDOW_SIZE_MS = 150
SERIAL_PORT = [p.device for p in serial.tools.list_ports.comports()][0]
BAUD_RATE = 115200
CLASSES = [0, 1]

# Buffers de Dados
df_full = pd.read_csv('1_DADOS.csv').dropna()
SAMPLING_RATE = int(1/df_full['timestamp'].diff().mean())
WINDOW_SIZE = int((WINDOW_SIZE_MS * SAMPLING_RATE) / 1000)
BUFFER_SIZE_RAW = WINDOW_SIZE_MS * SAMPLING_RATE // 1000
BUFFER_MEDIA_MOVEL_MS = 10
BUFFER_MEDIA_MOVEL = BUFFER_MEDIA_MOVEL_MS * SAMPLING_RATE // 1000
data_raw = collections.deque([0.0] * BUFFER_SIZE_RAW, maxlen=BUFFER_SIZE_RAW)
data_10s = collections.deque([0.0] * (SAMPLING_RATE * 10), maxlen=SAMPLING_RATE * 10)
data_rms = collections.deque([0.0] * 50, maxlen=50)
prob_buffer = collections.deque(maxlen=BUFFER_MEDIA_MOVEL) 

# Novos Buffers para a 3ª Coluna
data_latency = collections.deque([0.0] * 50, maxlen=50) # Latência em ms
data_class_hist = collections.deque([0] * 100, maxlen=100) # Histórico de 0 ou 1

latest_prediction = {"label": "INIT", "conf": 0, "probs": [0.5, 0.5]}
latest_metrics = {"rms": 0.0, "latency": 0.0}

def serial_and_model_worker():
    global latest_prediction, latest_metrics
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        while True:
            if ser.in_waiting:
                line_data = ser.readline().decode('utf-8').strip()
                if line_data:
                    try:
                        start_time = time.perf_counter() # Início do cronômetro
                        
                        val = float(line_data)
                        data_raw.append(val)
                        data_10s.append(val)
                        
                        # RMS
                        window = np.array(data_raw)
                        rms_val = np.sqrt(np.mean(window**2))
                        data_rms.append(rms_val)
                        
                        # IA com Média Móvel
                        input_data = window.reshape(-1, 1)
                        scaled_data = loaded_scalers[TEST_PATIENT].transform(input_data)
                        raw_probs = model_predict(scaled_data)[0]
                        prob_buffer.append(raw_probs)
                        
                        avg_probs = np.mean(list(prob_buffer), axis=0)
                        idx = np.argmax(avg_probs)
                        
                        # Cálculo de Latência
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        data_latency.append(latency_ms)
                        data_class_hist.append(idx)
                        
                        latest_prediction = {"label": idx, "conf": avg_probs[idx] * 100, "probs": avg_probs}
                        latest_metrics = {"rms": rms_val, "latency": latency_ms}
                    except: continue
    except Exception as e: print(e)

t = threading.Thread(target=serial_and_model_worker, daemon=True)
t.start()
# --- Layout Supervisório Ajustado ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9))

# Aumentei o espaço do cabeçalho para evitar sobreposição (0.6) 
# e deixei os outros com o mesmo peso (1, 1, 1)
gs = fig.add_gridspec(4, 1, height_ratios=[0.6, 1, 1, 1])

# Painel de Status (Topo - Ajustado para não encavalar)
ax_txt = fig.add_subplot(gs[0, :])
ax_txt.axis('off')
txt_main = ax_txt.text(0.5, 0.75, "", ha='center', va='center', fontsize=28, fontweight='bold', color='white', animated=True)
txt_sub = ax_txt.text(0.5, 0.20, "", ha='center', va='center', fontsize=11, family='monospace', color='#CCCCCC', animated=True)

# COLUNA 1
ax1 = fig.add_subplot(gs[1, 0])
line_raw, = ax1.plot(data_raw, color='#00FFCC', lw=1, animated=True)
ax1.set_title("EMG RAW SIGNAL", fontsize=8, color='#AAAAAA')
ax1.set_ylim(-0.0001, 0.0011) # Margem leve

ax2 = fig.add_subplot(gs[2:, 0]) 
# SINAL 10S: Cor Azul Cobalto e descolado das margens
line_10s, = ax2.plot(data_10s, color='#4169E1', lw=0.7, animated=True)
ax2.set_title("10s TELEMETRY (BLUE WAVE)", fontsize=8, color='#AAAAAA')
ax2.set_ylim(-0.0001, 0.0011) # Descolado da margem 0.0 e 0.001

def init():
    return line_raw, line_10s, txt_main, txt_sub

def update(frame):
    line_raw.set_ydata(list(data_raw))
    line_10s.set_ydata(list(data_10s))

    txt_main.set_text(latest_prediction['label'])
    status = f"CONF: {latest_prediction['conf']:.1f}% | RMS: {latest_metrics['rms']:.6f}V | LATENCY: {latest_metrics['latency']:.2f}ms"
    txt_sub.set_text(status)

    return line_raw, line_10s, txt_main, txt_sub

ani = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
plt.tight_layout(pad=3.0) # Aumentei o padding para evitar overlap
plt.show()
