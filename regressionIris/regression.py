# Bibliotecas sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, balanced_accuracy_score

X_data, y_data = load_iris(return_X_y = True)

# Criar datasets de treinamento e testes
x_train, x_test, y_train, y_teste = train_test_split(X_data,y_data, train_size=0.8)

# Criar o modelo de logistics regression
model_log = LogisticRegression(penalty='l2', solver='lbfgs')

# Treinar o modelo
model_log.fit(x_train,y_train)

# Verificar a predição
num = 0
sample = x_test[num]
print(sample)

model_log.predict([sample])
print(model_log.predict_proba([sample]))

# Verificar a acurácia do modelo
y_pred = model_log.predict(x_test)

score1 = accuracy_score(y_pred,y_teste)
score2 = balanced_accuracy_score(y_pred,y_teste)

print(score1,score2)