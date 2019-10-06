# -*- coding: utf-8 -*-
import pandas as pd
import joblib
diabetes = pd.read_csv("base/diabetes.csv", delimiter=',')

attributes = diabetes.drop('class', axis=1)
classes = diabetes['class']

#Cria atributos "dummies" para as colunas que não são numericas no conjunto de dados
new_attributes = pd.get_dummies(attributes);

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
print("Resultado da Avaliação do Modelo Descision Tree")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Classificar uma nova instância
print("Classificar [6,25,46,59,0,46.2,1.1,21]")
#nova_instancia=[[0,137,40,35,168,43.1,2.288,33]]
nova_instancia=[[6,130,46,59,0,46.2,1.1,21]]
print(classifier.predict(nova_instancia))

#Salvar o modelo para uso posterior
joblib.dump(classifier, 'modelo_decision_tree.joblib')


'''
[[79 16]
 [21 38]]
                 precision    recall  f1-score   support

tested_negative       0.79      0.83      0.81        95
tested_positive       0.70      0.64      0.67        59

       accuracy                           0.76       154
      macro avg       0.75      0.74      0.74       154
   weighted avg       0.76      0.76      0.76       154

Classificar [6,25,46,59,0,46.2,1.1,21]
['tested_positive']'''