#####################################     CASOS DE TESTE      ########################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      ######################################

import Churn as ax
import pandas as pd
import numpy as np

#########################################       FUNÇÃO DE PREENCHER MATRIZ       ######################################

# Função que preenche a matriz de teste multiplicando pela base desejada. #
def _preencheNovaMatriz( matriz: np.ndarray, base: float ) -> np.ndarray: 
    """Preeche a matriz de teste com o valor da base do modelo desejado;

    Args:
        matriz (np.ndarray): matriz de teste;
        base (float): base do modelo desejado;

    Returns:
        np.ndarray: retorna a nova matriz preenchida com uma nova base;
    """
    
    num_linhas, num_colunas = matriz.shape
    
    nova_matriz = np.copy( matriz )
    
    for i in range( num_linhas ):
        for j in range( num_colunas ):
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
    
    # Salva a quantidade de linhas e colunas da matriz #
    num_linhas, num_colunas = matriz.shape
    
    # Cria dataframe com id's #
    cdf = pd.DataFrame( { "id_cliente": range( num_linhas ) } )
    
    # Converta a matriz em uma lista de strings #
    lista_strings = [''.join( map( str, linha ) ) for linha in matriz]

    # Crie um DataFrame com uma coluna dos casos de teste #
    comparacaoChurn = pd.DataFrame( {'casos_teste': lista_strings} )
    
    # Converter a matriz para float
    matriz = matriz.astype( float )
    
    ########################### Modelo Binário ##########################################
    
    # Transforma a matriz em dataframe #
    df = pd.DataFrame( matriz )
    
    # Calcula churn #
    churn = ax._calculaChurnBinario( df, cdf )
    
    # Concatena o dataframe com os casos de teste com o de churn calculado #
    comparacaoChurn = pd.concat( [comparacaoChurn, churn], axis=1 )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename( columns = { 'churn': "churnBinario" }, inplace = True )
    
    ####################################################################################

    ########################### Modelo Simples ##########################################
    
    # Transforma a matriz em dataframe #
    df = pd.DataFrame( matriz )
    
    # Calcula o valor do denominador #
    valorMedia = ax._calculaSimples( num_colunas + 1 )
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename( columns = { 'churn': "churnSimples" }, inplace = True )
    
    ####################################################################################

    ########################### Modelo Linear ###########################################
    
    # Preenche a matriz de teste com valor correspondente ao modelo #
    nova_matriz = _preencheNovaMatriz(matriz, 0)
    
    # Transforma a nova matriz em dataframe #
    df = pd.DataFrame( nova_matriz )
    
    # Calcula o valor do denominador #
    valorMedia = ax._calculaLinear( num_colunas + 1 )
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename(columns = { 'churn': "churnLinear" }, inplace = True )
    
    ####################################################################################

    ####################### Modelo exponencial de base 2 ################################
    
    # Preenche a matriz de teste com valor correspondente ao modelo #
    nova_matriz = _preencheNovaMatriz(matriz, 2)
    
    # Transforma a nova matriz em dataframe #
    df = pd.DataFrame( nova_matriz )
    
    # Calcula o valor do denominador #
    valorMedia = ax._calculaExponencial( num_colunas + 1, 2 )
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename(columns = { 'churn': "churnExponencial_2" }, inplace = True )
    
    ####################################################################################

    ####################### Modelo exponencial de base e ################################
    
    # Preenche a matriz de teste com valor correspondente ao modelo #
    nova_matriz = _preencheNovaMatriz(matriz, e)
    
    # Transforma a nova matriz em dataframe #
    df = pd.DataFrame( nova_matriz )
    
    # Calcula o valor do denominador #
    valorMedia = ax._calculaExponencial( num_colunas + 1, e )
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, cdf, valorMedia )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename( columns = { 'churn': "churnExponencial_e" }, inplace = True )
    
    ####################################################################################

    ########################## Modelo Rencente ##########################################
    
    # Transforma a matriz em dataframe #
    df = pd.DataFrame( matriz )
    
    # Calcula o valor do denominador #
    valorMedia = ax._calculaRecenciaCliente( matriz )
    
    # Calcula churn #
    churn = ax._calculaChurnInternoR( df, cdf, valorMedia )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename( columns = { 'churn': "churnRecente" }, inplace = True )
    
    ####################################################################################

    ############################# Valor Real ###########################################
    
    comparacaoChurn["Real"] = vetorReal
    
    ####################################################################################
    
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
    
    # Cria uma dataframe vazio #
    comparacaoERRO = pd.DataFrame()
    
    # O dataframe criado com a mesma coluna de casos de teste #
    comparacaoERRO["casos_teste"] = comparacaoChurn["casos_teste"]
    
    # Modelo Binário #
    comparacaoERRO["ERROchurnBinario"] = (comparacaoChurn["churnBinario"] - comparacaoChurn["Real"]).round(10)
    
    # Modelo Simples #
    comparacaoERRO["ERROchurnSimples"] = (comparacaoChurn["churnSimples"] - comparacaoChurn["Real"]).round(10)
    
    # Modelo Linear #
    comparacaoERRO["ERROchurnLinear"] = (comparacaoChurn["churnLinear"] - comparacaoChurn["Real"]).round(10)

    # Modelo exponencial de base 2 #
    comparacaoERRO["ERROchurnExponencial_2"] = (comparacaoChurn["churnExponencial_2"] - comparacaoChurn["Real"]).round(10)

    # Modelo exponencial de base e #
    comparacaoERRO["ERROchurnExponencial_e"] = (comparacaoChurn["churnExponencial_e"] - comparacaoChurn["Real"]).round(10)

    # Modelo Rencente #
    comparacaoERRO["ERROchurnRecente"] = (comparacaoChurn["churnRecente"] - comparacaoChurn["Real"]).round(10)

    # Salva o dataframe em um arquivo CSV #
    ax._salvaArquivo( comparacaoERRO, "churnERRO.csv" )
    
    # Retorna o dataframe #
    return comparacaoERRO

################################################################################################################