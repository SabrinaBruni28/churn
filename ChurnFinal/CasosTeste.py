#####################################     CASOS DE TESTE      ########################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      ######################################

import Churn as ax
import pandas as pd
import numpy as np

#########################################       FUNÇÃO DE PREENCHER MATRIZ       ######################################

def _preencheNovaMatriz( matriz: np.ndarray, base: float ) -> np.ndarray: 
    """_summary_

    Args:
        matriz (np.ndarray): _description_
        base (float): _description_

    Returns:
        np.ndarray: _description_
    """
    
    num_linhas, num_colunas = matriz.shape
    
    nova_matriz = np.copy( matriz )
    
    for i in range( num_linhas ):
        for j in range( num_colunas):
            # f(a,b) => b, se a = 0 e a**b, se a != 0;
            # f(a,b) = a**b + b * !a;
            valor = ( base ** ( j + 1 ) ) + ( ( j + 1 ) * ( not base ) )
            nova_matriz[i][j] = nova_matriz[i][j] * valor
    
    return nova_matriz



################################################################################################################

#########################################       FUNÇÕES PARA OS CASOS DE TESTE    #####################################

# Função que calcula o churn de todos os modelos com base em uma matriz de teste com valores pré-definidos. #
def calculaChurnMatrizTeste( matriz: np.ndarray, vetorReal: float ) -> pd.DataFrame:
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
    
    # Converta a matriz em uma lista de strings #
    lista_strings = [''.join( map( str, linha ) ) for linha in matriz]

    # Crie um DataFrame com uma coluna dos casos de teste #
    comparacaoChurn = pd.DataFrame( {'casos_teste': lista_strings} )
    

    # Modelo Binário #
    churn = ax._calculaChurnBinario( df, cdf )
    comparacaoChurn = pd.concat( [comparacaoChurn, churn], axis=1 )
    comparacaoChurn.rename( columns = { 'churn': "churnBinario" }, inplace = True )

    # Modelo Simples #
    valorMedia = ax._calculaSimples( len(matriz[0])+1 )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename( columns = { 'churn': "churnSimples" }, inplace = True )

    # Modelo Linear #
    nova_matriz = _preencheNovaMatriz(matriz, 0)
    df = pd.DataFrame( nova_matriz )
    valorMedia = ax._calculaLinear( len(matriz[0])+1 )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename(columns = { 'churn': "churnLinear" }, inplace = True )

    # Modelo exponencial de base 2 #
    nova_matriz = _preencheNovaMatriz(matriz, 2)
    df = pd.DataFrame( nova_matriz )
    valorMedia = ax._calculaExponencial( len(matriz[0])+1, 2 )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename(columns = { 'churn': "churnExponencial_2" }, inplace = True )

    # Modelo exponencial de base e #
    nova_matriz = _preencheNovaMatriz(matriz, e)
    df = pd.DataFrame( nova_matriz )
    valorMedia = ax._calculaExponencial( len(matriz[0])+1, e )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename( columns = { 'churn': "churnExponencial_e" }, inplace = True )

    # Modelo Rencente #
    df = pd.DataFrame( matriz )
    valorMedia = ax._calculaRecenciaCliente( matriz, df.columns )
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    comparacaoChurn.rename( columns = { 'churn': "churnRecente" }, inplace = True )

    # Valor Real #
    comparacaoChurn["Real"] = vetorReal
    
    # Excluir a coluna de id #
    comparacaoChurn.drop( ['id'], axis = 1, inplace = True )
    
    # Salva o dataframe em um arquivo CSV #
    ax._salvaArquivo( comparacaoChurn, "churnComparacao.csv" )
    
    # Retorna o dataframe #
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
    
    comparacaoERRO["casos_teste"] = comparacaoChurn["casos_teste"]
    
    # Modelo Binário #
    comparacaoERRO["ERROchurnBinario"] = comparacaoChurn["churnBinario"] - comparacaoChurn["Real"]
    
    # Modelo Simples #
    comparacaoERRO["ERROchurnSimples"] = comparacaoChurn["churnSimples"] - comparacaoChurn["Real"]
    
    # Modelo Linear #
    comparacaoERRO["ERROchurnLinear"] = comparacaoChurn["churnLinear"] - comparacaoChurn["Real"]

    # Modelo exponencial de base 2 #
    comparacaoERRO["ERROchurnExponencial_2"] = comparacaoChurn["churnExponencial_2"] - comparacaoChurn["Real"]

    # Modelo exponencial de base e #
    comparacaoERRO["ERROchurnExponencial_e"] = comparacaoChurn["churnExponencial_e"] - comparacaoChurn["Real"]

    # Modelo Rencente #
    comparacaoERRO["ERROchurnRecente"] = comparacaoChurn["churnRecente"] - comparacaoChurn["Real"]

    # Salva o dataframe em um arquivo CSV #
    ax._salvaArquivo( comparacaoERRO, "churnERRO.csv" )
    
    # Retorna o dataframe #
    return comparacaoERRO

################################################################################################################