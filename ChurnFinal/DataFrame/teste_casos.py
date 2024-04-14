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

erro = ax.calculaChurnMatrizERRO(df)

print(df)
print()
print(erro)