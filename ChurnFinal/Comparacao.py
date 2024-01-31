#####################################     CASOS DE TESTE      ########################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      ######################################

import Churn as ax
import pandas as pd
import numpy as np

#########################################       FUNÇÕES PARA OS CASOS DE TESTE    #####################################

# Função que calcula o churn de todos os modelos com base em uma matriz de teste com valores pré-definidos. #
def calculaChurnMatrizTeste( matriz: float, vetorReal: float ) -> pd.DataFrame:
    """Calcula o churn de todos os modelos com base em uma matriz de teste pré-definida;

    Args:
        matriz (float): matriz de teste com marcações de compras ou não em cada período
        vetorReal (float): vetor com os valores reais de churn

    Returns:
        pd.DataFrame: retorna o datafram resultante com os valores de churn de cada modelo;
    """
    
    e = np.e
    df = pd.DataFrame( matriz )
    cdf = pd.DataFrame( { "id_cliente": range( len( df ) ) } )

    comparacaoChurn = pd.DataFrame()

    # Modelo Binário
    comparacaoChurn = ax._calculaChurnBinario( df, cdf )
    comparacaoChurn.rename( columns = { 'churn': "churnBinario" }, inplace = True )

    # Modelo Simples 
    valorMedia = ax._calculaSimples( len(matriz[0])+1 )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename( columns = { 'churn': "churnSimples" }, inplace = True )

    # Modelo Linear
    valorMedia = ax._calculaLinear( len(matriz[0])+1 )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename(columns = { 'churn': "churnLinear" }, inplace = True )

    # Modelo exponencial de base 2
    valorMedia = ax._calculaExponencial( len(matriz[0])+1, 2 )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename(columns = { 'churn': "churnExponencial_2" }, inplace = True )

    # Modelo exponencial de base e
    valorMedia = ax._calculaExponencial( len(matriz[0])+1, e )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename( columns = { 'churn': "churnExponencial_e" }, inplace = True )

    # Modelo Rencente
    valorMedia = ax._calculaRecenciaCliente( matriz, df.columns )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename( columns = { 'churn': "churnRecente" }, inplace = True )

    # Valor Real
    comparacaoChurn["Real"] = vetorReal
    
    # Salva o dataframe em um arquivo CSV
    ax._salvaArquivo( comparacaoChurn, "churnComparacao.csv" )
    
    # Retorna o dataframe
    return comparacaoChurn


# Função que calcula o erro de todos os modelos com base no valor real pré-definido. #
def calculaChurnMatrizERRO(comparacaoChurn: pd.DataFrame) -> pd.DataFrame:
    """Calcula o erro de todos os modelos com base no resultado do teste e dos valores reais pré-definidos;

    Args:
        comparacaoChurn (pd.DataFrame): dataframe da comparação entre todos os modelos;

    Returns:
        pd.DataFrame: retorna o dataframe resultante com os erros;
    """

    comparacaoERRO = pd.DataFrame()

    comparacaoERRO["id"] = comparacaoChurn["id"]
    
    # Modelo Binário
    comparacaoERRO["ERROchurnBinario"] = comparacaoChurn["churnBinario"] - comparacaoChurn["Real"]
    
    # Modelo Simples
    comparacaoERRO["ERROchurnSimples"] = comparacaoChurn["churnSimples"] - comparacaoChurn["Real"]
    
    # Modelo Linear
    comparacaoERRO["ERROchurnLinear"] = comparacaoChurn["churnLinear"] - comparacaoChurn["Real"]

    # Modelo exponencial de base 2
    comparacaoERRO["ERROchurnExponencial_2"] = comparacaoChurn["churnExponencial_2"] - comparacaoChurn["Real"]

    # Modelo exponencial de base e
    comparacaoERRO["ERROchurnExponencial_e"] = comparacaoChurn["churnExponencial_e"] - comparacaoChurn["Real"]

    # Modelo Rencente
    comparacaoERRO["ERROchurnRecente"] = comparacaoChurn["churnRecente"] - comparacaoChurn["Real"]

    # Salva o dataframe em um arquivo CSV
    ax._salvaArquivo( comparacaoERRO, "churnERRO.csv" )
    
    # Retorna o dataframe
    return comparacaoERRO

################################################################################################################