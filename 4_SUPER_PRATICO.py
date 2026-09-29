import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import serial
import numpy as np
import time
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
WINDOW_SIZE = 10  # Deve ser o mesmo tamanho usado no treinamento

class SupervisorioComIA(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Supervisório EMG - IA em Tempo Real & Retorno Serial")
        self.geometry("1400x850")

        self.porta_serial = None
        self.conectado = False
        self.thread_leitura = None
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

        self.gravando = False
        self.dados_gravados = []  

        self.criar_interfaces()

    def criar_interfaces(self):
        painel_esquerdo = ttk.Frame(self)
        painel_esquerdo.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Configurações de Conexão
        painel_controle = ttk.LabelFrame(painel_esquerdo, text=" Conexão Serial ", padding=10)
        painel_controle.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(painel_controle, text="Porta Serial:").pack(anchor=tk.W, pady=2)
        self.entry_porta = ttk.Entry(painel_controle, width=15)
        self.entry_porta.insert(0, "COM4")
        self.entry_porta.pack(anchor=tk.W, pady=2)

        ttk.Label(painel_controle, text="Baudrate:").pack(anchor=tk.W, pady=2)
        self.entry_baud = ttk.Entry(painel_controle, width=15)
        self.entry_baud.insert(0, "115200")
        self.entry_baud.pack(anchor=tk.W, pady=2)

        self.btn_conectar = ttk.Button(painel_controle, text="Conectar", command=self.alternar_conexao)
        self.btn_conectar.pack(fill=tk.X, pady=10)

        ttk.Separator(painel_controle, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(painel_controle, text="Máx. Y (ADC 0-65535):").pack(anchor=tk.W, pady=2)
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

        # Painel de Gravação CSV
        painel_gravacao = ttk.LabelFrame(painel_esquerdo, text=" Gravação de Dados ", padding=10)
        painel_gravacao.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(painel_gravacao, text="Label / ID:").pack(anchor=tk.W, pady=2)
        self.entry_id_paciente = ttk.Entry(painel_gravacao, width=18)
        self.entry_id_paciente.insert(0, "Paciente_01")
        self.entry_id_paciente.pack(anchor=tk.W, pady=2)

        self.btn_gravar = ttk.Button(painel_gravacao, text="▶ Iniciar Gravação", command=self.alternar_gravacao)
        self.btn_gravar.pack(fill=tk.X, pady=6)

        self.btn_salvar_csv = ttk.Button(painel_gravacao, text="💾 Salvar em .CSV", command=self.salvar_csv)
        self.btn_salvar_csv.pack(fill=tk.X, pady=2)

        self.lbl_status_gravacao = ttk.Label(painel_gravacao, text="Status: Parado (0 amostras)", font=("Arial", 9, "italic"))
        self.lbl_status_gravacao.pack(anchor=tk.W, pady=4)

        # Painel do Gráfico
        painel_graficos = ttk.Frame(self)
        painel_graficos.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.figura = Figure(figsize=(9, 7), dpi=100)
        self.figura.patch.set_facecolor('#1e1e1e')

        self.ax = self.figura.add_subplot(111)
        self.linha_bruto, = self.ax.plot(self.x_data, self.y_bruto_buffer, color='cyan', lw=1.2)
        
        self.ax.set_title("Sinal EMG Bruto (Pico 2 W) com Inferência da IA", fontsize=11, color='white', pad=12)
        self.ax.set_xlabel("Amostras", color='white')
        self.ax.set_ylabel("Valor ADC (0 - 65535)", color='white')
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

    def alternar_conexao(self):
        if not self.conectado:
            porta = self.entry_porta.get()
            try:
                baud = int(self.entry_baud.get())
                self.porta_serial = serial.Serial(porta, baud, timeout=0.05)
                self.conectado = True
                self.btn_conectar.config(text="Desconectar")
                
                self.thread_leitura = threading.Thread(target=self.ler_serial, daemon=True)
                self.thread_leitura.start()
                
                self.after(30, self.processar_dados)
            except Exception as e:
                messagebox.showerror("Erro de Conexão", f"Não foi possível abrir a {porta}:\n{e}")
        else:
            self.conectado = False
            if self.porta_serial and self.porta_serial.is_open:
                self.porta_serial.close()
            self.btn_conectar.config(text="Conectar")

    def atualizar_escala(self):
        try:
            ymax = float(self.entry_ymax.get())
            self.ax.set_ylim(0, ymax)
            self.canvas.draw()
        except ValueError:
            messagebox.showerror("Erro", "Insira um valor numérico válido.")

    def alternar_gravacao(self):
        if not self.gravando:
            self.gravando = True
            self.btn_gravar.config(text="⏸ Pausar Gravação")
            self.entry_id_paciente.config(state="disabled")
        else:
            self.gravando = False
            self.btn_gravar.config(text="▶ Iniciar Gravação")
            self.entry_id_paciente.config(state="normal")

    def salvar_csv(self):
        if not self.dados_gravados:
            messagebox.showwarning("Aviso", "Não há dados gravados para salvar.")
            return

        arquivo = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*")],
            title="Salvar Dados EMG Brutos"
        )
        
        if arquivo:
            try:
                with open(arquivo, "w", encoding="utf-8") as f:
                    f.write("timestamp,valor,label\n")
                    for ts, val, lbl in self.dados_gravados:
                        f.write(f"{ts},{val},{lbl}\n")
                messagebox.showinfo("Sucesso", f"Dados salvos com sucesso em:\n{arquivo}")
                self.dados_gravados.clear()
                self.lbl_status_gravacao.config(text="Status: Salvo (0 amostras)")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar arquivo:\n{e}")

    def ler_serial(self):
        while self.conectado:
            try:
                if self.porta_serial.in_waiting > 0:
                    linha = self.porta_serial.readline().decode('utf-8', errors='ignore').strip()
                    if linha:
                        valor = float(linha)
                        self.fila_dados.put(valor)
            except Exception:
                pass

    def processar_dados(self):
        if not self.conectado:
            return

        atualizou = False
        ts_atual = time.time()
        label_atual = self.entry_id_paciente.get().strip() or "Desconhecido"

        while not self.fila_dados.empty():
            valor = float(self.fila_dados.get())
            
            # Atualiza o buffer circular do gráfico bruto
            self.y_bruto_buffer[self.ponteiro] = valor
            self.ponteiro = (self.ponteiro + 1) % self.max_pontos
            
            # Salva se a gravação estiver ativa
            if self.gravando:
                self.dados_gravados.append((ts_atual, valor, label_atual))
                self.lbl_status_gravacao.config(text=f"Gravando... ({len(self.dados_gravados)} amostras)")

            # --- LÓGICA DE INFERÊNCIA DA IA E RETORNO SERIAL ---
            if self.model and self.scaler:
                self.buffer_ia.append(valor)
                if len(self.buffer_ia) >= WINDOW_SIZE:
                    # Pega apenas os últimos WINDOW_SIZE pontos
                    janela_atual = np.array(self.buffer_ia[-WINDOW_SIZE:])
                    
                    # Mantém o buffer com tamanho controlado (ex: desliza de 1 em 1 ou com stride)
                    # Removemos alguns elementos antigos para evitar crescimento excessivo da lista
                    if len(self.buffer_ia) > WINDOW_SIZE * 5:
                        self.buffer_ia = self.buffer_ia[-WINDOW_SIZE:]

                    # Normaliza com o scaler carregado (.pkl)
                    janela_scaled = self.scaler.transform(janela_atual.reshape(-1, 1)).flatten()
                    
                    # Formata o tensor para a entrada da CNN: (1, WINDOW_SIZE, 1)
                    sample = janela_scaled.reshape(1, WINDOW_SIZE, 1)
                    
                    # Executa a predição
                    resultado = self.model(sample, training=False)
                    prob = resultado.numpy()[0]
                    pred_class = int(np.argmax(prob))
                    confianca = float(np.max(prob)) * 100

                    # Atualiza a interface gráfica do supervisório
                    self.lbl_cluster_pred.config(text=f"Cluster: {pred_class}")
                    self.lbl_confianca.config(text=f"Confiança: {confianca:.1f}%")

                    # DEVOLVE O VALOR PREDITO PARA A PORTA SERIAL (para o Pico W ler)
                    if self.porta_serial and self.porta_serial.is_open:
                        try:
                            self.porta_serial.write(f"{pred_class}\n".encode('utf-8'))
                        except Exception:
                            pass

            atualizou = True

        if atualizou:
            self.linha_bruto.set_ydata(self.y_bruto_buffer)
            self.canvas.draw_idle()

        self.after(20, self.processar_dados)

if __name__ == "__main__":
    app = SupervisorioComIA()
    app.mainloop()
