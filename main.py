import numpy as np
import pandas as pd
import streamlit as st
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Título do site
st.title('Previsão do Tempo com Redes Neurais')

# Centralizando a imagem usando colunas
col1, col2, col3 = st.columns([1, 2, 1])  # Definindo uma estrutura de colunas com proporções 1:2:1

with col2:
    st.image('img.png', width=350)

# Carregando os dados do arquivo CSV com pandas
data = pd.read_csv("tempo.csv")

# Pré-processamento dos dados
# Aqui você deve realizar o pré-processamento necessário, como codificação de variáveis categóricas, normalização, etc.

# Separando as features (X) e o target (y)
X = data.drop(columns=['Vai_Chover'])
y = data['Vai_Chover']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o modelo com camadas ocultas
modelo = Sequential([
    Dense(units=16, activation='relu', input_dim=X_train.shape[1]),  # Primeira camada oculta
    Dense(units=8, activation='relu'),  # Segunda camada oculta
    Dense(units=1, activation='sigmoid')  # Camada de saída
])

# Definindo o otimizador com taxa de aprendizado configurável
learning_rate = 0.01  # Taxa de aprendizado
sgd = SGD(learning_rate=learning_rate)  # Otimizador SGD

# Compilando o modelo com o otimizador personalizado
modelo.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# Definindo o
