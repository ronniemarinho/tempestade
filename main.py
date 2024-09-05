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
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image('img.png', width=350)

# Carregando os dados do arquivo CSV com pandas
data = pd.read_csv("tempo.csv")

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

# Definindo o callback para salvar o melhor modelo
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Treinando o modelo com o callback
modelo.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[checkpoint])

# Carregando o melhor modelo salvo
modelo = load_model('best_model.keras')

# Avaliando a acurácia do modelo nos dados de teste
loss, accuracy = modelo.evaluate(X_test, y_test)
st.write(f"Acurácia nos dados de teste: {accuracy * 100:.2f}%")

# Predição de um registro específico
st.sidebar.title('Parâmetros de Previsão')
param1 = st.sidebar.slider('Velocidade do Vento (Km)', min_value=float(X.min().min()), max_value=float(X.max().max()), value=5.0)
param2 = st.sidebar.slider('Umidade do Ar', min_value=float(X.min().min()), max_value=float(X.max().max()), value=70.0)
param3 = st.sidebar.slider('Temperatura', min_value=float(X.min().min()), max_value=float(X.max().max()), value=25.0)
param4 = st.sidebar.slider('Min Temperatura', min_value=float(X.min().min()), max_value=float(X.max().max()), value=20.0)
param5 = st.sidebar.slider('Max Temperatura', min_value=float(X.min().min()), max_value=float(X.max().max()), value=30.0)

novo_registro = np.array([[param1, param2, param3, param4, param5]])

# Fazendo a predição
predicao = modelo.predict(novo_registro)
classe_predita = np.round(predicao)

# Exibindo a classe predita
st.write(f"Predição para o novo registro: {'Vai Chover' if classe_predita[0][0] == 1 else 'Não Vai Chover'}")

# Matriz de Confusão
y_pred = (modelo.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Vai Chover', 'Vai Chover'])

# Plotando a matriz de confusão
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues')
st.pyplot(fig)

# Cálculo das métricas avaliativas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Exibindo as métricas avaliativas
st.subheader('Métricas Avaliativas')
st.write(f"**Acurácia:** {accuracy * 100:.2f}%")
st.write(f"**Precisão:** {precision * 100:.2f}%")
st.write(f"**Recall:** {recall * 100:.2f}%")
st.write(f"**F1-Score:** {f1:.2f}")

# Exibindo as fórmulas instanciadas de Acurácia, Precisão, Recall e F1-Score
st.subheader("Fórmulas Instanciadas")

# Fórmula da Acurácia
st.latex(r'\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}')
st.write(f"Com os valores calculados:")
st.latex(f"\\text{{Acurácia}} = \\frac{{{cm[1, 1]} + {cm[0, 0]}}}{{{cm.sum()}}} = {accuracy:.2f}")

# Fórmula da Precisão
st.latex(r'\text{Precisão} = \frac{TP}{TP + FP}')
st.write(f"Com os valores calculados:")
st.latex(f"\\text{{Precisão}} = \\frac{{{cm[1, 1]}}}{{{cm[1, 1]} + {cm[0, 1]}}} = {precision:.2f}")

# Fórmula do Recall
st.latex(r'\text{Recall} = \frac{TP}{TP + FN}')
st.write(f"Com os valores calculados:")
st.latex(f"\\text{{Recall}} = \\frac{{{cm[1, 1]}}}{{{cm[1, 1]} + {cm[1, 0]}}} = {recall:.2f}")

# Fórmula do F1-Score
st.latex(r'F1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}')
st.write(f"Com os valores calculados:")
st.latex(f"F1 = 2 \\cdot \\frac{{{precision:.2f} \\cdot {recall:.2f}}}{{{precision:.2f} + {recall:.2f}}} = {f1:.2f}")
