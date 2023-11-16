import Churn as ax
import pandas as pd
import numpy as np

e = np.e

matriz = [
            [0, 1, 0, 0 ,0],
            [0, 0, 1, 0 ,0],
            [0, 0, 0, 1 ,0],
            [0, 0, 1, 0 ,0],
            [0, 1, 0, 0 ,0]
        ]

ax.calculaChurnMatrizTeste(matriz)
teste = cdf = pd.read_csv( "churnComparacao.csv", sep="\s+")
print(teste)