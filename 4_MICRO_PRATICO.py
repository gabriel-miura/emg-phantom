import machine
import utime
import sys
import uselect

# Configuração do pino ADC (ex: pino 28 / ADC2)
emg_pin = machine.ADC(28)
INTERVALO_US = 1000  # 1 ms (1 kHz)
tempo_anterior = utime.ticks_us()

# Configura o polling não-bloqueante para ler a serial (stdin) vinda do PC
spoll = uselect.poll()
spoll.register(sys.stdin, uselect.POLLIN)

# Opcional: pino de feedback (ex: LED integrado ou pino digital para acionamento)
# led = machine.Pin("LED", machine.Pin.OUT)

while True:
    tempo_atual = utime.ticks_us()
    
    # 1. Envio do sinal bruto a cada 1ms
    if utime.ticks_diff(tempo_atual, tempo_anterior) >= INTERVALO_US:
        tempo_anterior += INTERVALO_US
        valor_bruto = emg_pin.read_u16()
        sys.stdout.write(f"{valor_bruto}\n")
        
    # 2. Leitura da resposta da IA enviada pelo Supervisório (sem travar o loop)
    if spoll.poll(0):
        linha_ia = sys.stdin.readline().strip()
        if linha_ia:
            try:
                cluster_predito = int(linha_ia)
                
                # Exemplo de ação no hardware com base no cluster da IA:
                if cluster_predito == 1:
                    # led.on()
                    pass
                else:
                    # led.off()
                    pass
                    
            except ValueError:
                pass
