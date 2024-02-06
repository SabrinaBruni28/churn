import CasosTeste as ax

e = ax.np.e

matriz = ax.np.array( [
            [1 ,0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0 ,0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1 ,0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1 ,0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ] )

vetor_real = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

df = ax.calculaChurnMatrizTeste( matriz, vetor_real)

ax.calculaChurnMatrizERRO(df)



teste1 = ax.pd.read_csv( "churnComparacao.csv", sep="\s+")
teste2 = ax.pd.read_csv( "churnERRO.csv", sep="\s+")

print(teste1)
print()
print(teste2)