import pandas as pd
tabela = pd.read_csv("advertising.csv")
print(tabela)

print(tabela.info())

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(tabela.corr(),cmap = "Purples", annot = True)

plt.show()

x = tabela[["TV", "Radio", "Jornal"]]
y = tabela["Vendas"]

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

modelo_regressaolinear = LinearRegression()
modelo_arvorededecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvorededecisao.fit(x_treino, y_treino)

previsao_regresaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvorededecisao = modelo_arvorededecisao.predict(x_teste)

from sklearn.metrics import r2_score


print(r2_score(y_teste,previsao_regresaolinear))
print(r2_score(y_teste,previsao_arvorededecisao))

novos = pd.read_csv("Novos.csv")
display(novos)

print(modelo_arvorededecisao.predict(novos))