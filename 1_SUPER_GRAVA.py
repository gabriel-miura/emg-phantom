import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import serial
import numpy as np
import time
from scipy.signal import butter, lfilter, iirnotch
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class SupervisorioEMGComGravacao:

    def __init__(self, root):
        self.root = root
        self.root.title("Supervisório EMG - Sinal Bruto (Raspberry Pi Pico 2 W) & Gravação CSV")
        self.root.geometry("1300x800")

        self.porta_serial = None
        self.conectado = False
        self.thread_leitura = None
        self.fila_dados = queue.Queue()

        self.max_pontos = 1500
        self.taxa_amostragem = 1000  # 1kHz configurado no Pico W
        
        self.x_data = list(range(self.max_pontos))
        self.y_bruto_buffer = [float('nan')] * self.max_pontos
        self.ponteiro = 0
        self.ultimo_ponteiro = 0

        self.gravando = False
        self.dados_gravados = []  

        self.criar_interfaces()

    def criar_interfaces(self):
        painel_esquerdo = ttk.Frame(self.root)
        painel_esquerdo.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Painel de Conexão
        painel_controle = ttk.LabelFrame(painel_esquerdo, text=" Configurações e Escalas ", padding=10)
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

        # Painel de Gravação
        painel_gravacao = ttk.LabelFrame(painel_esquerdo, text=" Gravação de Dados (CSV Bruto) ", padding=10)
        painel_gravacao.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(painel_gravacao, text="ID do Paciente / Label:").pack(anchor=tk.W, pady=2)
        self.entry_id_paciente = ttk.Entry(painel_gravacao, width=18)
        self.entry_id_paciente.insert(0, "Paciente_01")
        self.entry_id_paciente.pack(anchor=tk.W, pady=2)

        self.btn_gravar = ttk.Button(painel_gravacao, text="▶ Iniciar Gravação", command=self.alternar_gravacao)
        self.btn_gravar.pack(fill=tk.X, pady=6)

        self.btn_salvar_csv = ttk.Button(painel_gravacao, text="💾 Salvar em .CSV", command=self.salvar_csv)
        self.btn_salvar_csv.pack(fill=tk.X, pady=2)

        self.lbl_status_gravacao = ttk.Label(painel_gravacao, text="Status: Parado (0 amostras)", font=("Arial", 9, "italic"))
        self.lbl_status_gravacao.pack(anchor=tk.W, pady=4)

        # Painel de Métricas
        painel_metricas = ttk.LabelFrame(painel_esquerdo, text=" Métricas do Sinal ", padding=10)
        painel_metricas.pack(fill=tk.BOTH, expand=True)

        self.lbl_rms = ttk.Label(painel_metricas, text="RMS: --", font=("Arial", 10, "bold"))
        self.lbl_rms.pack(anchor=tk.W, pady=6)
        
        self.lbl_mav = ttk.Label(painel_metricas, text="MAV: --", font=("Arial", 10))
        self.lbl_mav.pack(anchor=tk.W, pady=6)
        
        self.lbl_vpp = ttk.Label(painel_metricas, text="Peak-to-Peak: --", font=("Arial", 10))
        self.lbl_vpp.pack(anchor=tk.W, pady=6)

        # Painel de Gráficos
        painel_graficos = ttk.Frame(self.root)
        painel_graficos.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.figura = Figure(figsize=(9, 7), dpi=100)
        self.figura.patch.set_facecolor('#1e1e1e')

        self.ax = self.figura.add_subplot(111)
        self.linha_bruto, = self.ax.plot(self.x_data, self.y_bruto_buffer, color='cyan', lw=1.2)
        
        self.ax.set_title("Sinal EMG Bruto (Raspberry Pi Pico 2 W - read_u16)", fontsize=11, color='white', pad=12)
        self.ax.set_xlabel("Amostras / Tempo", color='white')
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
                
                self.root.after(30, self.processar_dados)
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
                    # Cabeçalho idêntico ao exigido pelo pipeline de treino
                    f.write("timestamp,valor,label\n")
                    for ts, val, lbl in self.dados_gravados:
                        f.write(f"{ts},{val},{lbl}\n")
                
                messagebox.showinfo("Sucesso", f"Dados brutos salvos com sucesso em:\n{arquivo}")
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
                        valor = float(linha)  # Lê o valor bruto read_u16() do Pico W
                        self.fila_dados.put(valor)
            except Exception:
                pass

    def calcular_metricas(self, dados_brutos):
        dados = np.array([d for d in dados_brutos if not np.isnan(d)])
        if len(dados) < 30:
            return

        # Métricas calculadas em cima da flutuação em relação à média local
        sinal_centrado = dados - np.mean(dados)
        rms = np.sqrt(np.mean(sinal_centrado**2))
        mav = np.mean(np.abs(sinal_centrado))
        vpp = np.ptp(dados)

        self.lbl_rms.config(text=f"RMS: {rms:.2f}")
        self.lbl_mav.config(text=f"MAV: {mav:.2f}")
        self.lbl_vpp.config(text=f"Peak-to-Peak: {vpp:.2f}")

    def processar_dados(self):
        if not self.conectado:
            return

        atualizou = False
        ts_atual = time.time()
        label_atual = self.entry_id_paciente.get().strip() or "Desconhecido"

        while not self.fila_dados.empty():
            valor = float(self.fila_dados.get())
            
            # Insere o dado 100% bruto diretamente no buffer de plotagem
            self.y_bruto_buffer[self.ponteiro] = valor
            
            # Se a gravação estiver ativa, salva o valor BRUTO puro
            if self.gravando:
                self.dados_gravados.append((ts_atual, valor, label_atual))
                self.lbl_status_gravacao.config(text=f"Gravando... ({len(self.dados_gravados)} amostras)")

            self.ponteiro = (self.ponteiro + 1) % self.max_pontos
            atualizou = True

        if atualizou:
            # Atualiza o gráfico com o sinal totalmente bruto
            self.linha_bruto.set_ydata(self.y_bruto_buffer)
            self.calcular_metricas(self.y_bruto_buffer)
            self.canvas.draw_idle()

        self.root.after(20, self.processar_dados)

if __name__ == "__main__":
    root = tk.Tk()
    app = SupervisorioEMGComGravacao(root)
    root.mainloop()
