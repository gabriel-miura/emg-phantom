# Classificação de Sinais de Eletromiografia de Superfície via Arquitetura Híbrida CNN-LSTM e Aprendizado Semi-Supervisionado Baseado em LGMM
Software de Inteligência Artificial de Redes Neurais Artificiais do tipo CNN-LSTM para classificação de sinais de superfície de eletromiografia por meio de aprendizado semi-supervisionado por Laplace-Gaussian Mixture Models (LGMM). O framework destaca-se pela alta capacidade de generalização comprovada por validação Leave-One-Subject-Out (LOSO) e uma estratégia de Data Augmentation que simula variabilidades fisiológicas (como impedância da pele e ruído), o que garante um sistema independente de hardware e autocalibrável ao paciente através de uma plataforma integrada de coleta e processamento via microcontrolador.

## Destaques e Contribuições
- Zero-Hardware Dependency for Simulation: elimina necessidade de equipamentos complexos para a criação da base de dados;
- Integrated Pipeline: integra coleta via microcontrolador, treinamento e aplicação em tempo real em um único ecossistema;
- Automated Labeling and Segmentation: segmentação automática de padrões de movimento durante a coleta;
- Sensor Agnostic Framework: modelo independente da escala do hardware;
- Leave-One-Subject-Out (LOSO): Implementação de validação intersujeito (acurácia reflita a capacidade do modelo de generalizar para indivíduos nunca vistos durante o treinamento).

## Pré-requisitos
- Python 3.11.9: Certifique-se de ter esta versão instalada (disponível em [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)).
- Hardware: Microcontrolador Raspberry Pi Pico 2W.
OBS.: Irá aparecer uma mensagem destacada no link de instalação desta versão do python: ```Note: Python 3.11.9 has been superseded by Python 3.11.15```. Pode ignorar, já que a versão 3.11.9 é estável e seguira, cujo motivo de uso é a estabilidade. Instale a versão 3.11.9.


## Instalação de Dependências
No terminal, instale as bibliotecas necessárias para o projeto:
```bash
pip install pandas pyserial matplotlib numpy==1.26.4 tensorflow==2.19.1 joblib scikit-learn scipy seaborn statsmodels jupyter ipykernel
```

## Execução do Projeto
A execução do projeto é feita em etapas. Primeiro, deve-se programar o Raspberry Pi Pico 2W para enviar dados da leitura do sensor por meio da comunicação SERIAL por meio do modelo de configura;áo disponibilizado em 0_RASPBERRY.py.

Após isso, deve-se executar 1_DADOS.py, o qual reconhecerá automaticamente a porta COM pela qual o microcontrolador está conectada e iniciará a leitura do sinal, ao passo que a gravação é feita clicando em "GRAVAR" e "PARAR" o registro, considerando que é responsabilidade do usuário deste software editar o valor numérico dentro da caixa de texto, que indica o ID do paciente que está usando o equipamento de eletromiografia para validação LOSO (Leave One Subject Off, ou Deixe Um Sujeito Fora - verifica a capacidade do modelo em generalizar para novas pessoas que ele nunca viu durante o treinamento). Por fim, clica-se em "SALVAR CSV" para salvar os dados gravados em um arquivo de CSV. Perceba que surgirá um arquivo 1_DADOS.csv, que constitui um arquivo Comma-Separated Values (Valores Separados por Vírgula) com os dados de gravação (perceba que é um arquivo colossal, visto que a taxa de amostragem maiores ou iguais a 1kHz produzem 1000 pontos de gravação por segundo - salvos atrasos na comunicação e leitura da porta serial da interface deste código python e o envio pelo próprio microcontrolador, ficando entre 830Hz -, ou seja, matematicamente ele vai gerar muitos dados, portanto não invente de abrir o arquivo no editor de código ou aplicativos de planilhas, pois o computador irá travar.
OBS.: Deve-se clicar alternadamente entre "GRAVAR" e "PARAR" para efetuar a troca do equipamento entre diferentes pessoas para, futuramente, realizar a tal da validação LOSO.

Terminada a etapa de dados, deve-se abrir o arquivo "2_TREINO.ipynb". Para o treinamento, optou-se por um Treinamento Sem-Supervisionado em duas etapas. A primeira etapa consiste em um Treinamento Não-Supervisionado para rotulação dos dados por meio de uma técnica de Clustering (Agrupamento) por meio do algoritmo Laplacian-Gaussian Mixture Models (LGMM) aplicada sobre "1_DADOS.csv" para agrupar sinais de assinaturas e topologias semelhantes (sinais de músculo relaxado e sinais de músculo contraído) em grupos denominados "CLUSTERS"; veja, o modelo vai treinar baseado nos dados de entrada (os sinais recortados de 124 pontos de toda a gravação de milhões de pontos de gravação) e retornar qual grupo ele pertence, não importa o significa do grupo, uma vez que nós quem determinaremos o que cada grupo significa para nós. 
OBS.: arquivos .ipynb devem ser executados dentro de um ambiente de desenvolvimento Google Colab ou, como o nosso caso precisamos do modelo no computador e não na nuvem, Anaconda (pesado e mais chato de configurar) ou Notebook Jupyter no Visual Studio Code, necessidando apenas conectar o python instalado (3.11.9) com as bibliotecas já instaladas no campo "SELECT KERNEL" do VSCODE. O kernel é uma interface que conecta o código executado em blocos ao python instalado na máquina (para reduzir o tempo de modificações feitas sem precisar espera 5 minutos para carregar as bibliotecas e código não alterado, somente executando o que foi modificado e acessando os demais blocos de código já executados e armazenados na memória, coisa que o arquivo python puro não permite). 
```python
# LAPLACIAN-GAUSSIAN MIXTURE MODELS (LGMM)
lgmm = GaussianMixture(
    n_components=N_CLUSTERS, 
    covariance_type='full', 
    tol=1e-4, 
    reg_covar=1e-5,
    max_iter=200, 
    n_init=10, 
    random_state=42
)
```

A segunda parte do 2_TREINO.py consiste no treinamento do algoritmo de inferência acerca do sinal, portanto o Treinamento Supervisionado pelo Clusterign do algoritmo de LGMM. O tipo de Rede Neural Artificial utilizada encontra-se em uma subcategoria expecializada de Deep Learning (Aprendizado Profundo): DEEP LEARNING > UMA REDE CONVULOCIONAL JUNTO DE UM TIPO DE REDE RECURSIVA, A LONG-SHORT TERM MEMORY. Esta etapa condiciona, treina, otimiza e, por fim, salva o modelo CNN-LSTM em um arquivo 2_MODELO.tflite.
```python
# ARQUITETURA DO MODELO CNN-LSTM PARA EXTRAÇÃO DE CARACTERÍSTICAS E PROCESSAMENTO DO SINAL EMG + CONFIGURAÇÃO DE OTIMIZAÇÃO (COMPILE)
def build_model(input_length, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_length, 1)), # sinal bruto
        layers.Conv1D(16, kernel_size=7, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.02)), # extrair caracteristicas
        layers.BatchNormalization(), 
        layers.MaxPooling1D(pool_size=4), # "reduzir" a qtde de caracteristicas
        layers.SpatialDropout1D(0.3),
        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.02)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),
        layers.SpatialDropout1D(0.3),
        layers.LSTM(16, dropout=0.5),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.02)),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Nadam(learning_rate=5e-4), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model
```

Caso seja necessário alterar as entradas ou saídas do modelo, modifique os parâmetros abaixo. Edita-se abaixo e não no código do modelo pois o código acima apresenta a estrutura geral, a arquitetura como um molde, cujos tamanhos da entrada e saída são, dessa forma, baseadas na divisão de dados, portanto precisamoes mexer tanto no modelo quanto na divisão. Isso é feito apenas alterando estes parâmetros que o código ajusta automaticamente. 
```python
# CONFIGURAÇÃO DO MODELO
N_CLUSTERS = 2 # número de tipos de movimentos realizados durante a gravação (mão aberta e mão fechada). SERVE TAMANHO DA DAÍDA OU QUANTIDADE DE SAÍDAS
WINDOW_SIZE_MS = 150 # tamanho da entrada do modelo em milissegundos do sinal. SERVE COMO TAMANHO DA ENTRADA
```
OBS.: O ```WINDOW_SIZE_MS``` apresenta o tamanho da janela de sinal que ele usará para processamento em tempo real do movimento realizado em MILISSEGUNDOS. A conversão para a quantidade de pontos, taxa de amostragem e demais valores é feita automaticamente pelo código ao extrair os valores de TIMESTAMP registrados durante a gravação, portanto ajuste apenas o tamanho em MILISSEGUNDOS. Exemplo: para um sinal de taxa de amostragem (FS) desconhecida e queremos extrair 150ms de janela, a FS é obtida em função do poder de processamento do microcontrolador Raspberry Pi Pico 2W, então se o cálculo retornar uma FS=830Hz, o tamanho da janela em pontos é obtida realizando uma regra de três, cujo resultado é 124 pontos.

A última parte 3_SUPERVISORIO.py é o código do 1_DADOS.py adaptado para utilizar o modelo para classificação em tempo real com os dados envidados pelo Raspberry Pi Pico 2W.
```python
# FUNÇÃO DE CARREGAMENTO DO MODELO SALVO E OTIMIZADO PARA TFLITE
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
```

## Termos de Uso e Atribuição
**Este projeto está sob a licença Apache 2.0. É obrigatório incluir o nome do autor original (Gabriel Hideaki Miura) nos seguintes campos:**
- **Lista de Referências:** Miura, G. H. (2026). Classificação de Sinais de Eletromiografia de Superfície via Arquitetura Híbrida CNN-LSTM e Aprendizado Semi-Supervisionado Baseado em LGMM (Version 1.0.0) [Computer software]. Mogi das Cruzes: ETEC Presidente Vargas, 2026. Circulação Restrita. Licenca Apache 2.0.
- **Corpo do texto:** "Para o processamento de dados e tomada de decisão, utilizou-se a arquitetura de Inteligência Artificial desenvolvida por MIURA (2026), licenciada sob Apache 2.0, que integra o núcleo tecnológico deste projeto."
-  **Folha de Rosto:** Colaboração Técnica / Coautoria de Software: Gabriel Hideaki Miura 
