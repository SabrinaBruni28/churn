import Comparacao as ax
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

vetor_real = [1.0, 1.0, 0.0, 1.0, 1.0 ]

df = ax.calculaChurnMatrizTeste(matriz, vetor_real)

ax.calculaChurnMatrizERRO(df)



teste1 = pd.read_csv( "churnComparacao.csv", sep="\s+")
teste2 = pd.read_csv( "churnERRO.csv", sep="\s+")

print(teste1)
print(teste2)