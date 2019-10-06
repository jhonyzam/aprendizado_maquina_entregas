import joblib

classifier = joblib.load('modelo_diabetes.joblib')
nova_instancia=[[0,100,40,35,168,43.1,2.288,33]]
print("Com o modelo salvo: ")
print("Classificar [0,137,40,35,168,43.1,2.288,33]")
print("Predict") #retorna a classe
print(classifier.predict(nova_instancia))
print("Predict proba") # quantaticamente pode dar a classe
print(classifier.predict_proba(nova_instancia))