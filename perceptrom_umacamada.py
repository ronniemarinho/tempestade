import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#pip install tensorflow
#pip install scikit-learn

# Carregando os dados do arquivo CSV com pandas
data = pd.read_csv("tempo.csv")

# Pré-processamento dos dados
# Aqui você deve realizar o pré-processamento necessário, como codificação de variáveis categóricas, normalização, etc.

# Separando as features (X) e o target (y)
X = data.drop(columns=['Vai_Chover'])
y = data['Vai_Chover']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o modelo
#units: O número de neurônios na camada. Neste caso, é definido como 1, indicando que haverá apenas um neurônio na camada de saída. Como estamos lidando com um problema de classificação binária (prever se vai chover ou não), um neurônio de saída é suficiente.
#activation: A função de ativação aplicada à saída da camada. Neste caso, estamos usando a função sigmóide ('sigmoid'). A função sigmóide é comumente usada em problemas de classificação binária, pois mapeia a saída para o intervalo [0, 1], representando a probabilidade de pertencer à classe positiva.
#input_dim: O número de variáveis de entrada (ou características). Aqui, é definido como o número de colunas em X_train, que representa o número de características no conjunto de dados.
print(X_train.shape[1])
modelo = Sequential([
    Dense(units=1, activation='sigmoid', input_dim=X_train.shape[1])
])

# Definindo o otimizador com taxa de aprendizado configurável
learning_rate = 0.01  # Taxa de aprendizado
sgd = SGD(learning_rate=learning_rate)#"Stochastic Gradient Descent" (Descida de Gradiente Estocástica)

# Compilando o modelo com o otimizador personalizado
#loss='binary_crossentropy': Esta é a função de perda que será usada para avaliar a discrepância entre a saída prevista do modelo e a saída verdadeira durante o treinamento
modelo.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
#batch_size=1: Este é o tamanho do lote (batch) usado durante o treinamento. O treinamento em lotes significa que o modelo não vê todas as amostras de treinamento de uma vez, mas sim em lotes de tamanho especificado. Neste caso, o tamanho do lote é 1, o que significa que o modelo verá uma amostra por vez e fará uma atualização de peso a cada amostra (treinamento em modo online).
modelo.fit(X_train, y_train, epochs=100, batch_size=1)

# Avaliando a acurácia do modelo nos dados de teste
loss, accuracy = modelo.evaluate(X_test, y_test)
print("Acurácia nos dados de teste: {:.2f}%".format(accuracy * 100))

# Fazendo predições nos dados de teste
y_pred_prob = modelo.predict(X_test)
y_pred = np.round(y_pred_prob).astype(int).flatten()  # Convertendo probabilidades em classes (0 ou 1)

# Calculando as métricas de precisão, recall e F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precisão: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-score: {:.2f}%".format(f1 * 100))

# Gerando a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()

# Predição de um registro específico
# Suponha que você tenha um registro novo representado como uma lista chamada 'novo_registro'
novo_registro = [[5, 70, 25, 20, 30]]

# Convertendo para numpy array
novo_registro = np.array(novo_registro)

# Fazendo a predição
predicao = modelo.predict(novo_registro)

# Arredondando a predição para obter a classe (0 ou 1)
classe_predita = np.round(predicao)

# Exibindo a classe predita
print("Predição para o novo registro:", classe_predita[0][0])

