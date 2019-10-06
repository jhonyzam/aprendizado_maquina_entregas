# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import svm
from sklearn.utils import resample

data = pd.read_csv("base/Ortopedia_Coluna.csv", delimiter=';')

#Isolar a base de dados com a classe minoritaria
minoritaria = data[data['Fusao_de_Vertebras']==1]
majoritaria = data[data['Fusao_de_Vertebras']==0]
minoritaria_upsample = resample(minoritaria, replace=True, n_samples=7900, random_state=123)

data_balanceado = pd.concat([majoritaria, minoritaria_upsample])

attributes = data_balanceado.drop('Fusao_de_Vertebras', axis=1)
classes = data_balanceado['Fusao_de_Vertebras']

#Cria atributos "dummies" para as colunas que não são numericas no conjunto de dados
new_attributes = pd.get_dummies(attributes);

print(new_attributes)

#new_attributes.to_csv("base/Ortopedia_Coluna_New.csv", sep=";", index=None, header=True)

# Dividir os dados aleatóriamente em conjunto para aprendizado e conjunto para testes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_attributes, classes, test_size=0.20) #20% do tamanho do arquivo será usado para testes
# X_train: segmento dos atributos para treinamento do modelo
# X_test : segmento dos atributos para avaliação do modelo
# y_train: segmento das classes para treinamento do modelo
# y_testn: segmento das classes para avaliação do modelo

#Treinar o modelo
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#Aplicar o modelo gerado sobre os dados separados para testes
y_pred = classifier.predict(X_test)

#print(y_pred)

#Avaliar o modelo: Acurácia e matriz de contingência
from sklearn.metrics import classification_report, confusion_matrix
print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Instancia a Support o classificador (Vector Machine)
classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#
print("cros_val_score")
scores = cross_val_score(classifier, X_test, y_test, cv=10)
print(scores)
print("Precisao media:", scores.mean())

print("cros_validate")
scores = cross_validate(classifier, X_test, y_test, cv=10)
print(scores)
print("Precisao media:", scores['test_score'].mean())
#9864601126829976