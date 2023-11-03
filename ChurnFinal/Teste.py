import Churn as ax
import pandas as pd
import numpy as np

e = np.e
periodos = 9
cdf = ax.lerArquivo("CDNOW_master.txt")
print(cdf)

dataVector = ax.controiVetorDatas(cdf, periodos)
print(dataVector)

matriz = ax.constroiMatrizClientePorPeriodo(cdf, periodos)
print(matriz[:5])
print("\n")

ax.preencheMatrizExponencial(matriz, dataVector, cdf, e)
print(matriz[:5])

dt = ax.matrizParaDataframe(matriz, dataVector)
print(dt)

ponderado = ax.calculaExponencial(periodos, e)
print(ponderado)
churn = ax.calculaChurnPonderado(dt, cdf, ponderado)
print(churn)

ax.salvaArquivo(churn, "churnTeste.csv")
teste = pd.read_csv( "churnTeste.csv", sep="\s+")
print(teste)