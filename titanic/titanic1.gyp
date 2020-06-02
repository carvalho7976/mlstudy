import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

modelo = RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=0)


def tranformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0
train['Sex_binario'] = train['Sex'].map(tranformar_sexo)


variaveis = ['Sex_binario','Age']
x = train[variaveis]
y = train['Survived']

x = x.fillna(-1)
modelo.fit(x,y)

test['Sex_binario'] = test['Sex'].map(tranformar_sexo)
x_prev = test[variaveis]
x_prev = x_prev.fillna(-1)
p = modelo.predict(x_prev)
sub = pd.Series(p,index=test['PassengerId'], name="Survived")
sub.to_csv("primeiro_modelo.csv", header=True)

result = pd.read_csv("primeiro_modelo.csv")

print(result)