import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

CSV_CHANNEL = 'valor'
CSV_CLUSTER = 'cluster'
N_CLUSTERS = 2
WINDOW_SIZE = 150

def build_model(input_length, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)),
        layers.Conv1D(8, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(8),
        layers.Dense(8, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def sliding_window(data, labels, window_size, stride):
    """Aplica o método de janela deslizante."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    X_windows = sliding_window_view(data, (window_size, data.shape[1]))[::stride].squeeze(axis=1)
    y_windows = sliding_window_view(labels, window_size)[::stride]
    is_stable = np.all(y_windows == y_windows[:, [0]], axis=1)
    X_clean = X_windows[is_stable]
    y_clean = y_windows[is_stable, -1]
    return X_clean, y_clean

def process_data(data, csv_channel, csv_class, window_size, stride):
    """Processa o conjunto completo, aplica escalonamento global e janela deslizante."""
    scaler = StandardScaler()
    reshaped_data = data[csv_channel].values.reshape(-1, 1)
    channel_data = scaler.fit_transform(reshaped_data).flatten()
    label_data = data[csv_class].values
    X, y = sliding_window(channel_data, label_data, window_size, stride)
    return X, y, scaler

def augment_emg(X, y, noise_factor=0.05, scale_range=(0.8, 1.2), n_augments=2):
    """Aplica ruído gaussiano e alteração de escala para data augmentation."""
    X_aug, y_aug = [X.copy()], [y.copy()]
    for _ in range(n_augments):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
        scale = np.random.uniform(scale_range[0], scale_range[1], size=(X.shape[0], 1, 1)) 
        X_aug.append(X * scale)
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)

def plot_cm(y_test, y_pred, CSV_CLASSES):
    _, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=CSV_CLASSES, ax=ax, cmap="Blues")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Real')
    plt.title(f"Confusion Matrix {accuracy_score(y_test, y_pred):.2%}")
    plt.show()

def plot_interactive_gmm_window(df, window_size, sampling_rate):
    """
    Cria uma janela interativa com um slider para navegar por todo o sinal
    e visualizar o zoom da janela atual e o cluster GMM correspondente.
    """
    fig, (ax_full, ax_window) = plt.subplots(2, 1, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25)

    time_axis = np.arange(len(df)) / sampling_rate
    signal = df['valor'].values
    clusters = df['cluster'].fillna(-1).values

    # Gráfico Superior: Sinal completo com coloração por cluster
    ax_full.set_title("Sinal Completo com Segmentação GMM via RMS (Arraste o slider para navegar)")
    ax_full.set_ylabel("Amplitude")
    ax_full.grid(True, alpha=0.3)

    ax_full.scatter(time_axis, signal, c=clusters, cmap='coolwarm', s=10, alpha=0.6)

    window_duration = window_size / sampling_rate
    initial_start = 0
    rect = Rectangle((initial_start, signal.min()), window_duration, signal.max() - signal.min(), 
                     color='yellow', alpha=0.3, label='Janela Deslizante')
    ax_full.add_patch(rect)
    ax_full.legend(loc='upper right')

    # Gráfico Inferior: Zoom da janela atual
    ax_window.set_title("Zoom da Janela Atual")
    ax_window.set_xlabel("Amostras na Janela")
    ax_window.set_ylabel("Amplitude")
    ax_window.grid(True, alpha=0.3)

    initial_window_data = signal[0:window_size]
    initial_cluster = clusters[window_size // 2]
    
    line_window, = ax_window.plot(initial_window_data, color='tab:blue', lw=1.5, marker='o')
    
    title_cluster = ax_window.text(0.02, 0.85, f"Cluster GMM: {initial_cluster}", 
                                   transform=ax_window.transAxes, fontsize=12, 
                                   fontweight='bold', bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax_window.set_xlim(0, window_size)
    ax_window.set_ylim(signal.min() * 1.1, (signal.max() if signal.max() > 0 else 1) * 1.1)

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    max_slider_val = max(0, len(signal) - window_size)
    slider = Slider(ax_slider, 'Posição (Amostra)', 0, max_slider_val, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        start_time = idx / sampling_rate
        rect.set_x(start_time)

        end_idx = min(idx + window_size, len(signal))
        w_data = signal[idx : end_idx]
        
        if len(w_data) < window_size:
            w_data = np.pad(w_data, (0, window_size - len(w_data)), 'edge')

        line_window.set_ydata(w_data)
        line_window.set_xdata(np.arange(len(w_data)))

        mid_idx = idx + window_size // 2
        if mid_idx < len(clusters):
            cl = clusters[mid_idx]
            title_cluster.set_text(f"Cluster GMM: {int(cl)} (Amostra Inicial: {idx})")

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def plot_interactive_emg_window(
    df, window_size, sampling_rate, col_name='valor_filtrado'
):
  """Cria uma janela interativa com um slider para navegar pelo sinal de EMG.

  O gráfico superior mostra o sinal completo e o retângulo da janela.
  O gráfico inferior mostra o zoom da janela atual com os pontos conectados.
  """
  fig, (ax_full, ax_window) = plt.subplots(2, 1, figsize=(14, 8))
  plt.subplots_adjust(bottom=0.25)

  time_axis = np.arange(len(df)) / sampling_rate
  signal = df[col_name].values

  # Gráfico Superior: Sinal EMG Completo
  ax_full.set_title(
      'Sinal EMG Completo (Arraste o slider para navegar pela janela)'
  )
  ax_full.set_ylabel('Amplitude')
  ax_full.grid(True, alpha=0.3)

  ax_full.plot(time_axis, signal, color='tab:blue', lw=0.8, alpha=0.7)

  window_duration = window_size / sampling_rate
  initial_start = 0
  
  # Tratamento para altura do retângulo caso o sinal seja constante
  sig_min, sig_max = signal.min(), signal.max()
  rect_height = (sig_max - sig_min) if sig_max != sig_min else 1.0

  rect = Rectangle(
      (initial_start, sig_min),
      window_duration,
      rect_height,
      color='yellow',
      alpha=0.3,
      label='Janela Deslizante',
  )
  ax_full.add_patch(rect)
  ax_full.legend(loc='upper right')

  # Gráfico Inferior: Zoom da janela atual com pontos conectados por linhas
  ax_window.set_title('Zoom da Janela Atual (Pontos Conectados)')
  ax_window.set_xlabel('Amostras na Janela')
  ax_window.set_ylabel('Amplitude')
  ax_window.grid(True, alpha=0.3)

  initial_window_data = signal[0:window_size]
  
  # Linha com marcadores 'o' para conectar os pontos claramente
  (line_window,) = ax_window.plot(
      np.arange(len(initial_window_data)),
      initial_window_data,
      color='tab:blue',
      lw=1.5,
      marker='o',
      markersize=4,
  )

  ax_window.set_xlim(0, window_size)
  ax_window.set_ylim(sig_min * 1.1, (sig_max if sig_max > 0 else 1) * 1.1)

  # Slider de navegação
  ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
  max_slider_val = max(0, len(signal) - window_size)
  slider = Slider(
      ax_slider, 'Posição (Amostra)', 0, max_slider_val, valinit=0, valstep=1
  )

  def update(val):
    idx = int(slider.val)
    start_time = idx / sampling_rate
    rect.set_x(start_time)

    end_idx = min(idx + window_size, len(signal))
    w_data = signal[idx:end_idx]

    if len(w_data) < window_size:
      w_data = np.pad(w_data, (0, window_size - len(w_data)), 'edge')

    line_window.set_ydata(w_data)
    line_window.set_xdata(np.arange(len(w_data)))

    fig.canvas.draw_idle()

  slider.on_changed(update)
  plt.show()

# --- CARREGAMENTO E PROCESSAMENTO ---
df_full = pd.read_csv('1_DADOS.csv').dropna()
SAMPLING_RATE = 1000

# 1. Configuração do Filtro Passa-Faixa Butterworth (Ordem 4, 20 a 150 Hz)
# Usamos output='sos' (Second-Order Sections) para maior estabilidade numérica
lowcut = 20.0
highcut = 150.0
sos_bp = signal.butter(N=4, Wn=[lowcut, highcut], btype='band', fs=SAMPLING_RATE, output='sos')

# 2. Configuração do Filtro Rejeita-Faixa (Notch em 60 Hz para rede elétrica)
f0 = 60.0
Q = 30.0  # Fator de qualidade (quanto maior, mais estreito e seletivo é o corte)
b_notch, a_notch = signal.iirnotch(f0, Q, SAMPLING_RATE)

# Aplicando os filtros aos dados (utilizamos filtfilt/sosfiltfilt para zero-phase, evitando atraso de fase)
sinal_original = df_full['valor'].values

# Passo A: Aplica o passa-faixa
sinal_filtrado = signal.sosfiltfilt(sos_bp, sinal_original)

# Passo B: Aplica o notch em 60 Hz no sinal já filtrado
sinal_filtrado = signal.filtfilt(b_notch, a_notch, sinal_filtrado)

# Armazena o sinal filtrado (opcional)
df_full['valor_filtrado'] = sinal_filtrado

plot_interactive_emg_window(df_full, window_size=WINDOW_SIZE, sampling_rate=SAMPLING_RATE, col_name='valor')
plot_interactive_emg_window(df_full, window_size=WINDOW_SIZE, sampling_rate=SAMPLING_RATE, col_name='valor_filtrado')

# 3. Cálculo do RMS sobre o sinal limpo/filtrado
df_full['rms'] = np.sqrt(
    pd.Series(sinal_filtrado).pow(2).rolling(
        window=SAMPLING_RATE, center=True, min_periods=1
    ).mean()
)

# 2. Aplicação do GMM no RMS
scaler_gmm = StandardScaler()
X_scaled = scaler_gmm.fit_transform(df_full[['rms']])
labels = GaussianMixture(
    n_components=N_CLUSTERS, 
    covariance_type='full', 
    tol=1e-4, 
    reg_covar=1e-5, 
    max_iter=200, 
    n_init=10, 
    random_state=42
).fit_predict(X_scaled)

df_full['cluster_id'] = labels
# Suavização temporal utilizando a moda em janela centrada
df_full['cluster_id'] = df_full['cluster_id'].rolling(window=int(SAMPLING_RATE), center=True).apply(lambda x: x.mode()[0]).fillna(df_full['cluster_id'])
df_full['cluster'] = df_full['cluster_id'].astype(int)

# Exibe a janela interativa com os clusters identificados via RMS + GMM
plot_interactive_gmm_window(df_full, WINDOW_SIZE, SAMPLING_RATE)

# --- PIPELINE DE TREINAMENTO DA CNN ---
X_all, y_all, scaler_signal = process_data(df_full, CSV_CHANNEL, CSV_CLUSTER, WINDOW_SIZE, WINDOW_SIZE // 2)
X_all = X_all.reshape(-1, WINDOW_SIZE, 1) 

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all) 
X_train_aug, y_train_aug = augment_emg(X_train, y_train) 

joblib.dump(scaler_signal, '2_SCALER.pkl') 

model = build_model(WINDOW_SIZE, N_CLUSTERS)
model.summary()
history = model.fit(X_train_aug, y_train_aug, validation_data=(X_test, y_test), epochs=10, batch_size=32, shuffle=True)
model.save('2_MODELO.keras')

loaded_model = models.load_model('2_MODELO.keras')
y_pred_raw = []
for i in range(len(X_test)):
    sample = np.expand_dims(X_test[i], axis=0) 
    resultado = loaded_model(sample, training=False) 
    y_pred_raw.append(resultado.numpy()[0])
    
y_pred = np.argmax(y_pred_raw, axis=1)
plot_cm(y_test, y_pred, df_full[CSV_CLUSTER].unique())
