import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import models
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- CONFIGURAÇÕES DA IA ---
MODEL_PATH = '2_MODELO.keras'
SCALER_PATH = '2_SCALER.pkl'
WINDOW_SIZE = 150  # Tamanho da janela usado no treinamento

class SupervisorioCSV(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Supervisório EMG - Reprodução CSV & IA Otimizada")
        self.geometry("1400x850")

        self.dados_csv = []
        self.indice_atual = 0
        self.reproduzindo = False
        self.thread_reproducao = None
        self.fila_dados = queue.Queue()

        # Carrega os artefatos da IA com tratamento de erro
        try:
            self.model = models.load_model(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("Modelo e Scaler carregados com sucesso!")
        except Exception as e:
            messagebox.showwarning("Aviso de IA", f"Não foi possível carregar os artefatos da IA:\n{e}\nA inferência ficará desativada.")
            self.model = None
            self.scaler = None

        self.max_pontos = 1500
        self.x_data = list(range(self.max_pontos))
        self.y_bruto_buffer = [float('nan')] * self.max_pontos
        self.ponteiro = 0
        
        # Buffer específico para alimentar a janela da IA
        self.buffer_ia = []
        self.contador_inferencia = 0  # Controla a frequência da IA para não travar a UI

        self.criar_interfaces()

    def criar_interfaces(self):
        painel_esquerdo = ttk.Frame(self)
        painel_esquerdo.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Configurações de Ficheiro CSV
        painel_controle = ttk.LabelFrame(painel_esquerdo, text=" Fonte de Dados (CSV) ", padding=10)
        painel_controle.pack(fill=tk.X, pady=(0, 10))

        self.btn_abrir = ttk.Button(painel_controle, text="📁 Abrir Ficheiro CSV", command=self.carregar_csv)
        self.btn_abrir.pack(fill=tk.X, pady=5)

        self.lbl_info_csv = ttk.Label(painel_controle, text="Nenhum ficheiro carregado", font=("Arial", 9, "italic"))
        self.lbl_info_csv.pack(anchor=tk.W, pady=2)

        self.btn_play = ttk.Button(painel_controle, text="▶ Reproduzir", command=self.alternar_reproducao, state=tk.DISABLED)
        self.btn_play.pack(fill=tk.X, pady=10)

        ttk.Separator(painel_controle, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(painel_controle, text="Máx. Y (Escala):").pack(anchor=tk.W, pady=2)
        self.entry_ymax = ttk.Entry(painel_controle, width=15)
        self.entry_ymax.insert(0, "65535")
        self.entry_ymax.pack(anchor=tk.W, pady=2)

        self.btn_escalas = ttk.Button(painel_controle, text="Fixar Escala", command=self.atualizar_escala)
        self.btn_escalas.pack(fill=tk.X, pady=10)

        # Painel de Status da IA (Tempo Real)
        painel_ia = ttk.LabelFrame(painel_esquerdo, text=" Inferência IA (Tempo Real) ", padding=10)
        painel_ia.pack(fill=tk.X, pady=(0, 10))

        self.lbl_cluster_pred = ttk.Label(painel_ia, text="Cluster: --", font=("Arial", 14, "bold"), foreground="blue")
        self.lbl_cluster_pred.pack(anchor=tk.W, pady=4)
        
        self.lbl_confianca = ttk.Label(painel_ia, text="Confiança: --%", font=("Arial", 11))
        self.lbl_confianca.pack(anchor=tk.W, pady=2)

        # Painel do Gráfico
        painel_graficos = ttk.Frame(self)
        painel_graficos.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.figura = Figure(figsize=(9, 7), dpi=100)
        self.figura.patch.set_facecolor('#1e1e1e')

        self.ax = self.figura.add_subplot(111)
        self.linha_bruto, = self.ax.plot(self.x_data, self.y_bruto_buffer, color='cyan', lw=1.2)
        
        self.ax.set_title("Sinal EMG Bruto (Reprodução CSV) com Inferência da IA", fontsize=11, color='white', pad=12)
        self.ax.set_xlabel("Amostras", color='white')
        self.ax.set_ylabel("Valor ADC", color='white')
        self.ax.set_xlim(0, self.max_pontos)
        self.ax.set_ylim(0, 65535)  
        
        self.estilizar_eixo(self.ax)
        self.figura.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figura, master=painel_graficos)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def estilizar_eixo(self, ax):
        ax.set_facecolor('black')
        ax.grid(True, color='gray', linestyle='--', alpha=0.3)
        ax.tick_params(colors='white', which='both', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('gray')

    def carregar_csv(self):
        ficheiro = filedialog.askopenfilename(
            filetypes=[("Ficheiros CSV", "*.csv"), ("Todos os ficheiros", "*.*")],
            title="Selecionar Ficheiro CSV de Dados"
        )
        if ficheiro:
            try:
                df = pd.read_csv(ficheiro).dropna()
                coluna_alvo = 'valor' if 'valor' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                self.dados_csv = df[coluna_alvo].values.tolist()
                
                self.indice_atual = 0
                self.lbl_info_csv.config(text=f"Carregado: {len(self.dados_csv)} amostras")
                self.btn_play.config(state=tk.NORMAL)
                messagebox.showinfo("Sucesso", f"Ficheiro carregado com {len(self.dados_csv)} linhas.")
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível ler o ficheiro CSV:\n{e}")

    def atualizar_escala(self):
        try:
            ymax = float(self.entry_ymax.get())
            self.ax.set_ylim(0, ymax)
            self.canvas.draw()
        except ValueError:
            messagebox.showerror("Erro", "Insira um valor numérico válido.")

    def alternar_reproducao(self):
        if not self.reproduzindo:
            if not self.dados_csv:
                return
            self.reproduzindo = True
            self.btn_play.config(text="⏸ Pausar")
            self.btn_abrir.config(state=tk.DISABLED)
            
            # Inicia thread de simulação do fluxo de dados
            self.thread_reproducao = threading.Thread(target=self.loop_reproducao, daemon=True)
            self.thread_reproducao.start()
            
            self.after(30, self.processar_dados)
        else:
            self.reproduzindo = False
            self.btn_play.config(text="▶ Reproduzir")
            self.btn_abrir.config(state=tk.NORMAL)

    def loop_reproducao(self):
        # Envia os dados para a fila controlando o ritmo
        while self.reproduzindo and self.indice_atual < len(self.dados_csv):
            valor = self.dados_csv[self.indice_atual]
            self.fila_dados.put(valor)
            self.indice_atual += 1
            time.sleep(0.002)  # Leve pausa para simular taxa de amostragem sem sufocar a fila
            
            if self.indice_atual >= len(self.dados_csv):
                self.reproduzindo = False
                break

    def processar_dados(self):
        atualizou = False
        ultima_predicao = None
        ultima_confianca = None

        # Processa em lote tudo o que estiver acumulado na fila para aliviar o processamento
        while not self.fila_dados.empty():
            valor = float(self.fila_dados.get())
            
            # Atualiza o buffer circular do gráfico bruto
            self.y_bruto_buffer[self.ponteiro] = valor
            self.ponteiro = (self.ponteiro + 1) % self.max_pontos

            # Alimenta o buffer da IA
            if self.model and self.scaler:
                self.buffer_ia.append(valor)
                if len(self.buffer_ia) > WINDOW_SIZE * 5:
                    self.buffer_ia = self.buffer_ia[-WINDOW_SIZE * 5:]

            atualizou = True

        # Executa a IA de forma cadenciada (a cada N ciclos ou quando houver dados suficientes)
        if self.model and self.scaler and len(self.buffer_ia) >= WINDOW_SIZE:
            self.contador_inferencia += 1
            # Roda a inferência a cada 10 pontos novos inseridos para evitar travamentos na UI
            if self.contador_inferencia >= 10:
                self.contador_inferencia = 0
                janela_atual = np.array(self.buffer_ia[-WINDOW_SIZE:])
                
                try:
                    janela_scaled = self.scaler.transform(janela_atual.reshape(-1, 1)).flatten()
                    sample = janela_scaled.reshape(1, WINDOW_SIZE, 1)
                    
                    resultado = self.model(sample, training=False)
                    prob = resultado.numpy()[0]
                    pred_class = int(np.argmax(prob))
                    confianca = float(np.max(prob)) * 100

                    # Atualiza os labels da interface
                    self.lbl_cluster_pred.config(text=f"Cluster: {pred_class}")
                    self.lbl_confianca.config(text=f"Confiança: {confianca:.1f}%")
                except Exception:
                    pass

        if atualizou:
            self.linha_bruto.set_ydata(self.y_bruto_buffer)
            self.canvas.draw_idle()

        if self.reproduzindo:
            self.after(30, self.processar_dados)  # Atualiza a tela a cada 30ms de forma fluida
        else:
            self.btn_play.config(text="▶ Reproduzir")
            self.btn_abrir.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = SupervisorioCSV()
    app.mainloop()
