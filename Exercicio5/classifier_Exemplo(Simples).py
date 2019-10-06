import pandas as pd
import calendar
bank = pd.read_csv("files/bank.csv", delimiter=';')

# Exibe o número de linhas e o número de colunas
print(bank.shape)

# Divide os dados em dois conjuntos: Atributos e Classes
attributes = bank.drop('y', axis=1)
classes = bank['y']

#Cria dicionário e mapa para meses
d = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11,
     'dec': 12}
attributes['month'] = attributes['month'].map(d).astype(int)

#Cria atributos "dummies" para as colunas que não são numericas no conjunto de dados
new_attributes = pd.get_dummies(attributes, columns=['job', 'marital', 'education', 'default',	'housing',	'loan',
                                                  'contact', 'poutcome'],
               drop_first=True, prefix=['job_', 'marital_', 'education_', 'default_', 'housing_', 'loan_',
                                                  'contact_', 'poutcome_'])
print(new_attributes)

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

print(y_pred)

#Avaliar o modelo: Acurácia e matriz de contingência
from sklearn.metrics import classification_report, confusion_matrix
print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Classificar uma nova instância
print("Classificar [30, 1787, 19, 10, 79, 1, -1, 0, 0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1]")
nova_instancia=[[30, 1787, 19, 10, 79, 1, -1, 0, 0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1]]
print(classifier.predict(nova_instancia))

#Salvar o modelo para uso posterior
from sklearn.externals import joblib
joblib.dump(classifier, 'decisiontree1.joblib')

classifier = joblib.load('decisiontree1.joblib')
nova_instancia=[[30, 1787, 19, 10, 79, 1, -1, 0, 0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1]]
print("Com o modelo salvo: ")
print("Classificar [30, 1787, 19, 10, 79, 1, -1, 0, 0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1]")
print(classifier.predict(nova_instancia))