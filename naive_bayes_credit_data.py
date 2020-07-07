

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:26:15 2020

@author: Jeremias
"""


import pandas as pd
import numpy as np
base = pd.read_csv('credit_data.csv')
print(base.describe())

#apagar coluna
#base.drop('age', 1, inplace=True) #o primeiro atributo é a coluna que quero apagar, o segundo é se quero apagar a coluna inteira, e o terceiro é se quero apagr sem salvar em nenhuma variável

#base.drop(base[base.age < 0].index, inplace=True)  # .index significa que irei apagar os indices

# para esse tipo de problema é interessante substituirmos os valores inconsistentes com a média do atributo

base.mean() # a média de todos os atributos da minha base

base['age'].mean() # a média do atributo 'age'

# mas como o atributo age é onde apresenta a inconsistencia da minha base, não é viável tirar a média logo de cara, pois os
# valores incosistentes irão alterar minha média, apresentando um valor pouco confiável.

   
base['age'][base.age > 0].mean() # agora, printa a média do atributo idade sem os valores iconsistentes, aumentando a confiabilidade do resultado.
base.loc[base['age'] < 0, 'age'] = 40.92 # o segundo atributo 'age' significa que o campo que quero atualizar é o campo 'age', ai passo o valor que irá receber.
base.loc[base['age'] < 0] 
base['age'].mean() # mostra a média da base atualizada.
pd.isnull(base['age']) # mostra um a um dos atributos 'age' do registro, sendo false se o argumento não é nulo (não posssui numero) e true caso for nulo.
base.loc[pd.isnull(base['age'])] # localiza os atributos nulos do campo 'age'.
#base.loc[pd.isnull(base['age'])] = 40.92

previsores = base.iloc[:, 1:4].values # cria a variável previsores, onde é armazenado toda informação previsora da minha base, que nesse caso seria "salário(anual), idade e o empréstimo que aquela pessoa fez.
                                      # o primeiro argumento ":" significa que quero armazenar todas as linhas, e os segundo é os campos que quero armazenar, no caso do campo 1 ao 3 (o 4 é o limite superior, ele não pega o 4 e sim até o 3).

classe =  base.iloc[:, 4].values     # neste caso, vou armazenar só o campo 4. ou seja, a classe meta da minha base.


# identifica as inconsistencias na base, e conserta os valores.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])

# padroniza os valores, para nao haver muita discrepancia na hora de aplicar os algoritmos euclidianos, ou afins.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsore_treinamentos, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0) # aqui dividimos a base de dados em 2: base de treinamento, e base de teste. os 2 primeiros parametros são minha base (previsores, e classe), e o 3 é a divisão para teste da minha base, neste caso, dividiu em 25% para teste.

from sklearn.naive_bayes import GaussianNB  # biblioteca do algoritmo do naive bayes
classificador = GaussianNB()    # criação do objeto
classificador.fit(previsore_treinamentos, classe_treinamento) # transforma a base em probabilidades, aqui eh onde ocorre o treinamento da maquina
previsoes = classificador.predict(previsores_teste) 


                 
from sklearn.metrics import confusion_matrix, accuracy_score
precisao =  accuracy_score(classe_teste, previsoes)   # como parametro passo o gabarito (classe_teste) e o resultado da minha previsao.
matriz = confusion_matrix(classe_teste, previsoes)  
