import Churn as ax
import pandas as pd
import numpy as np
import csv

def preencheMatriz():
    print()



def calculaChurnMatrizTeste( matriz: float ) -> pd.DataFrame:
    e = np.e
    df = pd.DataFrame( matriz )
    cdf = pd.DataFrame( {"id_cliente": range(len(df))})

    comparacaoChurn = pd.DataFrame()

    # Modelo Binário
    comparacaoChurn = ax.calculaChurnBinario( df, cdf )
    comparacaoChurn.rename(columns={'churn': "churnBinario"}, inplace=True)

    # Modelo Simples
    valorMedia = ax.calculaSimples( len(matriz[0])+1 )
    churn = ax.calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge(comparacaoChurn, churn, on = "id")
    comparacaoChurn.rename(columns={'churn': "churnSimples"}, inplace=True)

    # Modelo Linear
    valorMedia = ax.calculaLinear( len(matriz[0])+1 )
    churn = ax.calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge(comparacaoChurn, churn, on = "id")
    comparacaoChurn.rename(columns={'churn': "churnLinear"}, inplace=True)

    # Modelo exponencial de base 2
    valorMedia = ax.calculaExponencial( len(matriz[0])+1, 2 )
    churn = ax.calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge(comparacaoChurn, churn, on = "id")
    comparacaoChurn.rename(columns={'churn': "churnExponencial_2"}, inplace=True)

    # Modelo exponencial de base e
    valorMedia = ax.calculaExponencial( len(matriz[0])+1, e )
    churn = ax.calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge(comparacaoChurn, churn, on = "id")
    comparacaoChurn.rename(columns={'churn': "churnExponencial_e"}, inplace=True)

    # Modelo Rencente
    valorMedia = ax.calculaRecenciaCliente( matriz, df.columns )
    churn = ax.calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge(comparacaoChurn, churn, on = "id")
    comparacaoChurn.rename(columns={'churn': "churnRecente"}, inplace=True)

    ax.salvaArquivo(comparacaoChurn, "churnComparacao.csv")
    return comparacaoChurn