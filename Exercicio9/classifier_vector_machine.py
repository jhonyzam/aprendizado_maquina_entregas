# -*- coding: utf-8 -*-
import joblib

classifier = joblib.load('modelo/modelo_svm.joblib')
nova_instancia=[[7.7,0.49,0.26,1.9,0.062,9,31,0.9966,3.39,0.64,9.6],[8,0.59,0.16,1.8,0.065,3,16,0.9962,3.42,0.92,10.5]]
print("Com o modelo salvo: ")
print("Classificar [7.7,0.49,0.26,1.9,0.062,9,31,0.9966,3.39,0.64,9.6]")
print("Predict") #retorna a classe
print(classifier.predict(nova_instancia))
print("Predict proba") # quantaticamente pode dar a classe
print(classifier.predict_proba(nova_instancia))
