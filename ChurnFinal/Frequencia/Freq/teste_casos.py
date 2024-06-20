import CasosTeste as c

matriz = c.np.array( [
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

df = c.calculaChurnMatrizTeste( matriz, vetor_real)

erro = c.calculaChurnMatrizERRO(df)

print(df)
print()
print(erro)