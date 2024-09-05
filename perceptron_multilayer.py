import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
# pip install tensorflow
# pip install scikit-learn

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

# Definindo o callback para salvar o melhor modelo
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Treinando o modelo com o callback
modelo.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[checkpoint])

# Carregando o melhor modelo salvo
modelo.load_weights('best_model.h5')

# Avaliando a acurácia do modelo nos dados de teste
loss, accuracy = modelo.evaluate(X_test, y_test)
print("Acurácia nos dados de teste: {:.2f}%".format(accuracy * 100))

# Predição de um registro específico
novo_registro = [[5, 70, 25, 20, 30]]  # Suponha que você tenha um registro novo representado como uma lista

# Convertendo para numpy array
novo_registro = np.array(novo_registro)

# Fazendo a predição
predicao = modelo.predict(novo_registro)

# Arredondando a predição para obter a classe (0 ou 1)
classe_predita = np.round(predicao)

# Exibindo a classe predita
print("Predição para o novo registro:", classe_predita[0][0])
