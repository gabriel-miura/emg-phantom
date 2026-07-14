# Importação da biblioteca para medir o tempo de execução (latência)
import time
# Biblioteca fundamental para cálculos matemáticos e manipulação de arrays/matrizes
import numpy as np
# Biblioteca para manipulação e análise de dados tabulares (DataFrames)
import pandas as pd
# Biblioteca para criação de gráficos e visualizações de dados
import matplotlib.pyplot as plt
# Ferramentas matemáticas para aplicar filtros digitais em sinais (sinal EMG)
from scipy.signal import butter, filtfilt
# Ferramenta para criar janelas deslizantes sobre um array de forma eficiente
from numpy.lib.stride_tricks import sliding_window_view

# Importação do modelo de mistura Gaussiana para agrupar (clusterizar) os dados automaticamente
from sklearn.mixture import GaussianMixture
# Ferramentas para avaliar o desempenho do modelo (Matriz de Confusão, Acurácia, etc.)
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
# Função crucial para dividir os dados entre treino e teste de forma estratificada (80/20 por classe)
from sklearn.model_selection import train_test_split

# TensorFlow e Keras para construção e treinamento da Rede Neural (Deep Learning)
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

# Define sementes (seeds) fixas para que os resultados sejam reproduzíveis sempre que rodar o código
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# HIPERPARÂMETROS E CONFIGURAÇÕES GERAIS
# ==========================================
NOISE_FACTOR = 0.05       # Intensidade do ruído gaussiano adicionado no Data Augmentation
SCALE_RANGE = (0.8, 1.2)  # Variação de escala de amplitude (entre 80% e 120%) para Augmentation
N_AUGMENTS = 2            # Número de vezes que os dados originais serão duplicados e alterados

BATCH_SIZE = 64           # Quantidade de amostras processadas antes de atualizar os pesos da rede
EPOCHS = 100              # Quantidade de vezes que a rede verá o conjunto de dados inteiro no treino

# ==========================================
# FUNÇÕES DO PIPELINE
# ==========================================

def build_model(input_length, n_classes):
    """Constrói a arquitetura da Rede Neural Convolucional (CNN) Simples."""
    # Inicializa um modelo sequencial (camadas empilhadas uma após a outra)
    model = models.Sequential([
        # Define o formato de entrada: (Tamanho da Janela, 1 canal de dados)
        layers.Input(shape=(input_length, 1)),
        
        # Camada Convolucional 1D para extrair características do sinal no tempo
        layers.Conv1D(8, kernel_size=15, activation='relu', padding='same'),
        # Normaliza as ativações da camada anterior, acelerando o treino e ajudando a estabilizar
        layers.BatchNormalization(),
        # Reduz a dimensionalidade, pegando o valor máximo a cada 4 pontos (resume a informação)
        layers.MaxPooling1D(pool_size=4),
        
        # Segunda Camada Convolucional para extrair padrões mais complexos
        layers.Conv1D(16, kernel_size=7, activation='relu', padding='same'),
        # Nova redução de dimensionalidade
        layers.MaxPooling1D(pool_size=2),
        
        # "Achata" as matrizes 2D geradas pelas convoluções em um único vetor 1D
        layers.Flatten(), 
        
        # "Desliga" aleatoriamente 40% dos neurônios durante o treino para forçar a rede a aprender de forma generalista (Evita Overfitting)
        layers.Dropout(0.4), 
        
        # Camada totalmente conectada com 16 neurônios para processar os padrões encontrados
        layers.Dense(16, activation='relu'),
        
        # Camada de saída. O número de neurônios é igual ao número de classes.
        # Softmax converte a saída em probabilidades (soma = 100%)
        layers.Dense(n_classes, activation='softmax')
    ])
    
    # Compila o modelo definindo o otimizador (Adam), a função de erro e as métricas a serem acompanhadas
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4), # Taxa de aprendizado baixa para evitar pulos grandes
        loss='sparse_categorical_crossentropy',        # Ideal para classes inteiras (0, 1, 2...)
        metrics=['accuracy']                           # Acompanharemos a acurácia
    )
    # Retorna o modelo pronto
    return model

def sliding_window(data, labels, window_size, stride):
    """Recorta o sinal contínuo em várias janelas (blocos) menores."""
    # Se os dados forem 1D, transforma em coluna (necessário para a função seguinte)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Cria janelas deslizantes nos sinais (Ex: pega do ponto 0 ao 150, depois do 75 ao 225...)
    X_windows = sliding_window_view(data, (window_size, data.shape[1]))[::stride].squeeze(axis=1)
    # Faz a mesma janelação para as classes (rótulos)
    y_windows = sliding_window_view(labels, window_size)[::stride]

    # Verifica se a janela inteira pertence a uma única classe (evita janelas de transição entre movimentos)
    is_stable = np.all(y_windows == y_windows[:, [0]], axis=1)
    
    # Mantém apenas as janelas cujos sinais e classes são estáveis
    X_clean = X_windows[is_stable]
    # O rótulo da janela será o último valor da janela (como são todos iguais, qualquer um serviria)
    y_clean = y_windows[is_stable, -1]

    # Retorna os dados fragmentados em janelas e seus respectivos rótulos
    return X_clean, y_clean

def process_data(data, csv_channel, csv_class, window_size, stride):
    """Extrai os dados do DataFrame e passa para a função de janelas deslizantes."""
    # Se não houver dados suficientes para formar nem uma janela, retorna vazio
    if len(data) < window_size:
        return np.array([]), np.array([])
        
    # Extrai a coluna do sinal como um array limpo
    channel_data = data[csv_channel].values.flatten()
    # Extrai a coluna das classes
    label_data = data[csv_class].values
    
    # Executa a fragmentação e retorna
    X, y = sliding_window(channel_data, label_data, window_size, stride)
    return X, y

def augment_emg(X, y, noise_factor=NOISE_FACTOR, scale_range=SCALE_RANGE, n_augments=N_AUGMENTS):
    """Gera variações artificiais dos dados para treinar o modelo de forma mais robusta."""
    # Inicializa listas com os dados originais
    X_aug, y_aug = [X.copy()], [y.copy()]
    
    # Repete o processo pelo número de aumentos definido
    for _ in range(n_augments):
        # Gera ruído branco gaussiano (simula interferências elétricas do sensor)
        noise = np.random.normal(0, noise_factor, X.shape)
        # Adiciona o ruído ao dado original e guarda na lista
        X_aug.append(X + noise)
        y_aug.append(y) # O rótulo continua o mesmo

        # Gera multiplicadores aleatórios de escala (simula variação de força do músculo ou contato da pele)
        scale = np.random.uniform(
            scale_range[0], scale_range[1],
            size=(X.shape[0], 1, 1)
        )
        # Multiplica o dado original pela escala e guarda
        X_aug.append(X * scale)
        y_aug.append(y)

    # Junta todas as listas em um grande array e retorna
    return np.concatenate(X_aug), np.concatenate(y_aug)

def plot_history(history):
    """Plota gráficos lado a lado mostrando a evolução do erro (loss) e acerto (accuracy) no treino e teste."""
    # Cria uma figura com 2 gráficos (1 linha, 2 colunas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Configura o gráfico de Loss (Erro)
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Configura o gráfico de Accuracy (Acurácia/Acertos)
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Acurácia')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Ajusta os espaços e exibe a imagem
    plt.tight_layout()
    plt.show()

def plot_cm(y_test, y_pred, csv_classes):
    """Plota a Matriz de Confusão e o Relatório de Métricas."""
    # Cria a figura para a Matriz de Confusão
    _, ax = plt.subplots(figsize=(8, 6))
    # Gera a matriz baseada nas predições reais vs esperadas
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=csv_classes, ax=ax, cmap="Blues")
    ax.set_xlabel('Predicted') # Rótulo Eixo X
    ax.set_ylabel('Real')      # Rótulo Eixo Y
    # Adiciona a % de acerto no título
    plt.title(f"Confusion Matrix {accuracy_score(y_test, y_pred):.2%}")
    plt.show()

    # Imprime no terminal as métricas precisão, recall e f1-score
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

def load_tflite(model_path):
    """Carrega o modelo exportado para TFLite e cria uma função pronta para uso simulando embarcado."""
    # Inicia o interpretador do arquivo .tflite
    interpreter = tf.lite.Interpreter(model_path)
    # Aloca memória para os pesos e variáveis (obrigatório no tflite)
    interpreter.allocate_tensors()
    # Pega os metadados de entrada e saída (shapes e index)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Função interna que recebe um dado cru e retorna a predição
    def predict_function(data):
        # Garante que a entrada seja float32 (formato exigido pelo TFLite)
        input_data = np.array(data, dtype=np.float32)
        # Se a entrada estiver 2D, adiciona a dimensão de "batch" no início para virar 3D
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Alimenta os dados de entrada na camada inicial do modelo
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # Roda a rede neural (Processa)
        interpreter.invoke()
        # Lê o resultado na camada de saída
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Retorna as probabilidades de cada classe
        return output_data
    
    # Retorna a função utilizável
    return predict_function


# ==========================================
# 1. CONFIGURAÇÃO E CARREGAMENTO
# ==========================================
CSV_CHANNEL = 'valor'      # Nome da coluna que tem os dados elétricos
CSV_CLUSTER = 'cluster'    # Nome da coluna que vai receber os labels criados
WINDOW_SIZE_MS = 150       # Tamanho da janela de recorte em milissegundos

# Lê o arquivo CSV ignorando linhas em branco
df_full = pd.read_csv('1_DADOS.csv').dropna()

# Calcula dinamicamente a taxa de amostragem (Hz) com base na diferença de tempo entre as amostras
SAMPLING_RATE = int(1/df_full['timestamp'].diff().mean())
# Converte o tempo da janela em milissegundos para número de amostras reais (linhas no dataframe)
WINDOW_SIZE = int((WINDOW_SIZE_MS * SAMPLING_RATE) / 1000)

def emg_lowpass(data, cutoff=5, fs=1000, order=4):
    """Filtra ruídos de alta frequência do sinal EMG, alisando a onda."""
    nyq = 0.5 * fs # Calcula a frequência de Nyquist (metade da amostragem)
    normal_cutoff = cutoff / nyq # Normaliza a frequência de corte
    # Cria os coeficientes matemáticos do filtro Butterworth
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Aplica o filtro aos dados indo e voltando para não ter atraso de fase
    return filtfilt(b, a, data)

# ==========================================
# 2. TRATAMENTO DO SINAL E CLUSTERING
# ==========================================
# Retificação: Transforma todos os valores negativos do sinal em positivos (Módulo)
df_full['rect'] = df_full['valor'].abs()
# Envoltória: Aplica o filtro passa-baixa no sinal retificado para pegar apenas o "contorno" de força (suave)
df_full['env_smooth'] = emg_lowpass(df_full['rect'], cutoff=5, fs=SAMPLING_RATE)
# Derivada: Calcula a taxa de variação da envoltória (se a força está subindo ou descendo) e preenche os NaN com 0
df_full['derivada'] = df_full['env_smooth'].diff().fillna(0)

# Cria um DataFrame secundário removendo eventuais linhas inúteis do cálculo de envoltória
df_valid = df_full.dropna(subset=['env_smooth']).copy()

# Pega apenas as colunas úteis para definir se o músculo tá ativo ou em repouso
X_features = df_valid[['env_smooth', 'derivada']].values

# Define o número desejado de classes (Ex: 2 = Repouso e Contração)
N_CLUSTERS = 2
# Configura o algoritmo Gaussian Mixture Model para analisar os dados cegamente e agrupá-los em 2 clusters
lgmm = GaussianMixture(
    n_components=N_CLUSTERS, 
    covariance_type='full', 
    tol=1e-4, 
    reg_covar=1e-5,
    max_iter=200, 
    n_init=10, 
    random_state=42
)
# Executa o algoritmo nos dados e gera um array de IDs de classe (ex: 0, 0, 0, 1, 1, 1...)
labels = lgmm.fit_predict(X_features)

# Coloca a previsão dos clusters de volta no DataFrame
df_valid['cluster_id'] = pd.Series(labels, index=df_valid.index)

# Pós processamento (Mode Filter): Analisa blocos de amostras e força pontos divergentes rápidos a virarem o mais comum
# (Exemplo: 0 0 1 0 0 -> O 1 vira 0, removendo "piscas" na classificação)
df_valid['cluster_id'] = df_valid['cluster_id'].rolling(
    window=int(SAMPLING_RATE/2),
    center=True
).apply(lambda x: x.mode()[0]).fillna(df_valid['cluster_id'])

# Passa a coluna corrigida para o DataFrame original
df_full['cluster'] = df_valid['cluster_id']

# ==========================================
# 3. EXTRAÇÃO DE JANELAS E DIVISÃO (TRAIN_TEST_SPLIT ESTRATIFICADO)
# ==========================================
# Ao invés de cortar o CSV cronologicamente, extraímos TODAS as janelas de uma vez
X_all, y_all = process_data(df_full, CSV_CHANNEL, CSV_CLUSTER, WINDOW_SIZE, WINDOW_SIZE // 2)

# Adiciona uma dimensão extra necessária para as Camadas Conv1D (amostras, tamanho, canal)
X_all = X_all.reshape(-1, WINDOW_SIZE, 1)

# A MÁGICA DA DIVISÃO ESTRATIFICADA:
# Pega todas as janelas geradas e divide aleatoriamente 80% para treino e 20% para teste.
# O parâmetro 'stratify=y_all' OBRIGA a manter a proporção exata de classes de y_all.
# Ou seja: 80% dos dados da classe 0 estarão no treino e 80% da classe 1 também.
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, 
    test_size=0.2,    # 20% pro teste
    stratify=y_all,   # Estratificar com base nos rótulos
    random_state=42   # Semente para dar sempre o mesmo embaralhamento
)

# Faz o Data Augmentation (ruído + ganho) SOMENTE nos dados de treino
# (Nunca aplique Augmentation no teste, pois o teste deve simular a vida real crua)
X_train_aug, y_train_aug = augment_emg(X_train, y_train)

# ==========================================
# 4. PLOTAGEM DOS CLUSTERS (REPRESENTAÇÃO VISUAL)
# ==========================================
sample_size = 500 # Pega 500 pontinhos para mostrar no gráfico de exemplo
cmap = plt.get_cmap('tab10') # Usa paleta de cores padrão

# Cria os gráficos para visualizar como ficou o visual de cada cluster gerado
fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(15, 3 * N_CLUSTERS), sharex=True)

for i in range(N_CLUSTERS):
    # Pega onde no CSV a classe 'i' começou a aparecer
    indices = df_full[df_full['cluster'] == i].index
    if len(indices) == 0:
        continue
    
    start_idx = indices[0]
    # Extrai o bloco de sinal bruto dessa classe para desenhar
    data_window = df_full['valor'].iloc[start_idx : start_idx + sample_size].values
    
    # Plota a linha
    axes[i].plot(data_window, color=cmap(i % 10), lw=1.2)
    axes[i].set_title(f"Assinatura Visual: Cluster {i}", fontsize=12, fontweight='bold')
    axes[i].set_ylabel("Amplitude")
    axes[i].grid(alpha=0.2)
    
    # Coloca uma caixinha com a amplitude média em cima do gráfico
    avg_amp = np.mean(np.abs(data_window))
    axes[i].annotate(f"Média Abs: {avg_amp:.6f}", xy=(0.01, 0.85), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.5))
    
axes[-1].set_xlabel("Pontos da Janela (Amostras)")
plt.tight_layout()
plt.show()

# ==========================================
# 5. TREINAMENTO DO MODELO
# ==========================================
# Chama a função que constrói a arquitetura da rede neural
model = build_model(WINDOW_SIZE, N_CLUSTERS)
# Mostra no console um resumo com a quantidade de parâmetros de cada camada
model.summary()

# Inicia de fato o treinamento
history = model.fit(
    X_train_aug, y_train_aug,            # Dados sintéticos e originais de treino
    validation_data=(X_test, y_test),    # Dados cru de validação (para acompanhar sem ensinar)
    epochs=EPOCHS,                       # Número de voltas completas
    batch_size=BATCH_SIZE,               # Quantas janelas envia de uma vez
    shuffle=True,                        # Mistura bem as amostras de treino a cada época (IMPORTANTE!)
    verbose=1                            # Mostra barra de progresso no terminal
)
# Ao final, mostra os gráficos de aprendizado (curvas subindo/descendo)
plot_history(history)

# ==========================================
# 6. EXPORTAÇÃO TFLITE E INFERÊNCIA
# ==========================================
# Prepara a conversão da rede pesada do keras (Tensorflow) para a versão lite (TFLite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS # Restringe para usar apenas operações leves nativas
]
# Salva o arquivo no disco em modo binário
with open('2_MODELO.tflite', 'wb') as f:
    f.write(converter.convert())

# Recarrega o modelo recém salvo para simular seu uso num microcontrolador (ESP32/Raspberry)
model_predict = load_tflite("2_MODELO.tflite")

# Inicia um cronômetro
inicio = time.perf_counter()
y_pred_raw = []

# Passa janela por janela do arquivo de teste na rede simulada
for i in range(len(X_test)):
    # Faz predição (retorna as porcentagens)
    resultado = model_predict(X_test[i])
    y_pred_raw.append(resultado[0])

# Usa np.argmax para transformar as porcentagens (Ex: [0.1, 0.9]) na resposta final de classe (Ex: Classe 1)
y_pred = np.argmax(y_pred_raw, axis=1)

# Para cronômetro
fim = time.perf_counter()

# Imprime o tempo médio que o processador demorou para dar UMA predição
print(f"LATÊNCIA MÉDIA: {(fim - inicio) / len(X_test):.6f}s")

# Chama a plotagem de acertos em formato de Matriz Quadriculada
plot_cm(y_test, y_pred, df_full[CSV_CLUSTER].unique())
