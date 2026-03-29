# EMG-PHANTOM PROJECT
Software de Inteligência Artificial de Redes Neurais Artificiais do tipo CNN-LSTM para classificação de sinais de superfície de eletromiografia por meio de aprendizado semi-supervisionado por Laplace-Gaussian Mixture Models (LGMM).

# Como Executar o Projeto
Siga os passos abaixo para configurar o ambiente e processar os sinais de EMG.

## 1. Pré-requisitos
- Python 3.11.9: Certifique-se de ter esta versão instalada (disponível em [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)).
- Hardware: Microcontrolador compatível com MicroPython (para a etapa final).

## 2. Instalação de Dependências
No terminal, instale as bibliotecas necessárias para o projeto:
```bash
pip install numpy scipy pandas matplotlib plotly tensorflow scikit-learn cloudpickle m2cgen (ATUALIZAR AS BIBLIOTECAS)
```

## 3. Preparação dos Dados
Prepare um arquivo .csv seguindo a estrutura de gravações temporais do sensor EMG. O arquivo deve conter:
- Sinal: Colunas correspondentes aos canais do sensor.
- Classe: Coluna identificando o movimento realizado.

Os sinais utilizados neste projeto foram obtidos a partir do dataset [Kaggle - EMG Signal for gesture recognition - Sojan Prajapati](https://www.kaggle.com/datasets/sojanprajapati/emg-signal-for-gesture-recognition)

## 4. Configuração e Execução
Navegue até o diretório src/ e ajuste os parâmetros no arquivo MAIN.py conforme sua necessidade:

```python
# CONFIGURAÇÃO
CSV_NAME = 'emg_signals.csv' # nome do arquivo CSV
CSV_CHANNEL = 'channel7' # nome da coluna do CSV que possui os dados do sensor EMG
CSV_CLASS = 'class' # nome da coluna do CSV que possui os dados da classe do sinal EMG
CSV_CLASSES = [1, 2] # valores que as classes do sinal do CSV podem assumir
SAMPLING_RATE = 200 # taxa de amostragem do sensor em hertz (velocidade com a qual os dados são adquiridos)
WINDOW_SIZE_MS = 150 # tamanho da entrada do modelo em milissegundos do sinal
SCALOGRAM_START_MS = 10000 # inicio da amostra de análise do sinal em milissegundos
SCALOGRAM_END_MS = 40000 # fim da amostra de análise do sinal em milissegundos
```

Após configurar, execute o script principal para treinar o modelo ou processar os dados:
```bash
python MAIN.py
```

## 5. Deploy no Microcontrolador
Para realizar a inferência na ponta (Edge AI), carregue o arquivo MICRO.py no seu dispositivo com suporte a MicroPython.

# Termos de Uso e Atribuição
**Este projeto está sob a licença Apache 2.0. É obrigatório incluir o nome do autor original (Gabriel Hideaki Miura) nos seguintes campos:**
- **Lista de Referências:** Miura, G. H. (2026). Sistema de Classificação de Sinais de Eletromiografia utilizando Técnicas de Machine Learning (Version 1.0.0) [Computer software]. Mogi das Cruzes: ETEC Presidente Vargas, 2026. Circulação Restrita. Licenca Apache 2.0.
- **Corpo do texto:** "Para o processamento de dados e tomada de decisão, utilizou-se a arquitetura de Inteligência Artificial desenvolvida por MIURA (2026), licenciada sob Apache 2.0, que integra o núcleo tecnológico deste projeto."
-  **Folha de Rosto:** Colaboração Técnica / Coautoria de Software: Gabriel Hideaki Miura 
