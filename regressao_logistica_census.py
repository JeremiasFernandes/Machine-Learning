


import pandas as pd

base = pd.read_csv('census.csv')

previsores =  base.iloc[:, 0:14].values # primeiro parametro siginifica que quero alocar todos os valores da coluna, e o segundo é que coluna a qual coluna
                                        # (com +1, pois é o limite superior, nesse caso quero as colunas de 0 a 13, entao coloco 0:14)
classe = base.iloc[:, 14]     # mesmo comando do de cima.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_previsores = LabelEncoder() # pegas as colunas da variavel previsores e transforma em um array de números correspondentes (somente as colunas categoricas)

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough') 
previsores = onehotencorder.fit_transform(previsores).toarray()   # substitui na variavél previsores o array com a transformação das informações categoricas em numeros (variaveis dummy), que fizemos na linha acima.

labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe) #tranforma os >=50k e < 50k em variaveis 0 e 1

# #escalona os previsores
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# previsores = scaler.fit_transform(previsores)


from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.15, random_state = 0)



from sklearn.linear_model import LogisticRegression   # biblioteca do algoritmo do naive bayes
classificador = LogisticRegression(random_state= 1)    # cria a regressao
classificador.fit(previsores_treinamento, classe_treinamento) # transforma a base em probabilidades, aqui eh onde ocorre o treinamento da maquina
previsoes = classificador.predict(previsores_teste) 

                 
                 
from sklearn.metrics import confusion_matrix, accuracy_score
precisao =  accuracy_score(classe_teste, previsoes)   # como parametro passo o gabarito (classe_teste) e o resultado da minha previsao.
matriz = confusion_matrix(classe_teste, previsoes)  
