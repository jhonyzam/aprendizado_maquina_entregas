# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib

def get_wine_class(value):
    if value <= 3:
        return 'RUIM'

    if value <= 6:
        return 'MEDIO'

    return 'BOM'

data_red = pd.read_csv("base/winequality-red.csv", delimiter=';')
data_red['quality'] = data_red['quality'].map(get_wine_class)

#Isolar a base de dados com a classe minoritaria
ruim = data_red[data_red['quality']=='RUIM']
medio = data_red[data_red['quality']=='MEDIO']
bom = data_red[data_red['quality']=='BOM']

#print(data_red['quality'].value_counts())

ruim_upsample = resample(ruim, replace=True, n_samples=1372, random_state=0)
bom_upsample = resample(bom, replace=True, n_samples=1372, random_state=0)

data_balanceado = pd.concat([ruim_upsample, medio, bom_upsample])

#print(data_balanceado['quality'].value_counts())

#data_white = pd.read_csv("base/winequality-white.csv", delimiter=';')

#print("RED");
#print(data_red.count());

#print("WHITE");
#print(data_white.count());

attributes = data_balanceado.drop('quality', axis=1)
classes = data_balanceado['quality']

X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)
#rbf, poly, linear
classifier = svm.SVC(kernel = 'poly', C = 1, probability=True);
svm_model_linear = classifier.fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test)
svm_predictions_proba = svm_model_linear.predict_proba(X_test)

#print("Predict")
#print(svm_predictions)
#print("Predict proba")
#print(svm_predictions_proba)

#Avaliar o modelo: Acurácia e matriz de contingência
from sklearn.metrics import classification_report, confusion_matrix
print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))


#Salvar o modelo para uso posterior
joblib.dump(classifier, 'modelo/modelo_svm.joblib')
#carregar o modelo
#classifier = joblib.load('modelo_ortopedia.joblib')