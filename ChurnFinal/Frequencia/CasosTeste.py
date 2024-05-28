#####################################     CASOS DE TESTE      #########################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      #######################################

import Churn as ax
import pandas as pd
import numpy as np

#########################################       FUNÇÃO DE PREENCHER MATRIZ       ######################################

# Função que preenche a matriz de teste multiplicando pela base desejada. #
def _preencheNovaMatriz( matriz: np.ndarray, base: float ) -> np.ndarray: 
    """
    Preeche a matriz de teste com o valor da base do modelo desejado;

    Args:
        matriz (np.ndarray): matriz de teste;
        base (float): base do modelo desejado;

    Returns:
        np.ndarray: retorna a nova matriz preenchida com uma nova base;
    """
    
    # Salva a quantida de linhas e colunas da matriz #
    num_linhas, num_colunas = matriz.shape
    
    # Faz uma cópia da matriz original #
    nova_matriz = np.copy( matriz )
    
    # Percorre a matriz #
    for i in range( num_linhas ):
        for j in range( num_colunas ):
            # f(a,b) => b, se a = 0 e a**b, se a != 0;
            # f(a,b) = a**b + b * !a;
            valor = ( base ** ( j + 1 ) ) + ( ( j + 1 ) * ( not base ) )

            # Multiplica a posição pela base desejada #
            # Se a posição for zero, continuará zero #
            # Se a posição for 1, o valor seja a nova base #
            nova_matriz[i][j] = nova_matriz[i][j] * valor
    
    # Retorna a nova matriz #
    return nova_matriz

#######################################################################################################################

#########################################       FUNÇÕES PARA OS CASOS DE TESTE    #####################################

# Função que calcula o churn de todos os modelos com base em uma matriz de teste com valores pré-definidos. #
def calculaChurnMatrizTeste( matriz: np.ndarray, vetorReal: float ) -> pd.DataFrame:
    """
    Calcula o churn de todos os modelos com base em uma matriz de teste pré-definida;

    Args:
        matriz (float): matriz de teste com marcações de compras ou não em cada período;
        (Deve-se possuir pelo menos 2 colunas e pelo menos 1 linha);
        vetorReal (float): vetor com os valores reais de churn;
        (Deve-se possuir a mesma quantidade de colunas que a matriz possui de linhas);

    Returns:
        pd.DataFrame: retorna o dataframe resultante com os valores de churn de cada modelo;
    """
    
    e = np.e
    
    # Salva a quantidade de linhas e colunas da matriz #
    num_linhas, num_colunas = matriz.shape
    
    # Converte a matriz em uma lista de strings #
    lista_strings = [''.join( map( str, linha ) ) for linha in matriz]

    # Cria um DataFrame com uma coluna dos casos de teste #
    comparacaoChurn = pd.DataFrame( {'casos_teste': lista_strings} )
    
    
    ########################### Modelo Binário ##########################################
    
    # Transforma a matriz em dataframe #
    df = pd.DataFrame( matriz )
    
    # Calcula churn #
    churn = ax._calculaChurnBinario( df )
    
    # Concatena o dataframe com os casos de teste com o de churn calculado #
    comparacaoChurn = pd.concat( [comparacaoChurn, churn], axis=1 )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename( columns = { 'churn': "churnBinario" }, inplace = True )
    
    ####################################################################################

    ########################### Modelo Simples ##########################################
    
    # Calcula o valor do denominador #
    valorMedia = num_colunas
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, valorMedia )
    
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
    valorMedia = ax._calculaLinear( num_colunas )
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, valorMedia )
    
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
    valorMedia = ax._calculaExponencial( num_colunas, 2 )
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, valorMedia )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename(columns = { 'churn': "churnExponencial_2" }, inplace = True )
    
    ####################################################################################

    ####################### Modelo exponencial de base e ################################
    
    # Converter a matriz para float
    matriz_f = matriz.astype( float )
    
    # Preenche a matriz de teste com valor correspondente ao modelo #
    nova_matriz = _preencheNovaMatriz(matriz_f, e)
    
    # Transforma a nova matriz em dataframe #
    df = pd.DataFrame( nova_matriz )
    
    # Calcula o valor do denominador #
    valorMedia = ax._calculaExponencial( num_colunas, e )
    
    # Calcula churn #
    churn = ax._calculaChurnInterno( df, valorMedia )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename( columns = { 'churn': "churnExponencial_e" }, inplace = True )
    
    ####################################################################################

    ########################## Modelo Rencente ##########################################
    
    # Transforma a matriz em dataframe #
    df = pd.DataFrame( matriz )
    
    # Cria uma coluna de valor do denominador da média para cada cliente preenchida com zero #
    df["Dmedia"] = 0
    
    # Calcula o valor do denominador #
    df.apply(ax._calculaRecenciaCliente, axis=1)
    
    # Calcula churn #
    churn = ax._calculaChurnInternoR( df )
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    comparacaoChurn = pd.merge( comparacaoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    comparacaoChurn.rename( columns = { 'churn': "churnRecente" }, inplace = True )
    
    ####################################################################################

    ############################# Valor Real ###########################################
    
    # Adiciona uma coluna com os valores reais de churn #
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
    """
    Calcula o erro de todos os modelos com base no resultado do teste e dos valores reais pré-definidos;

    Args:
        comparacaoChurn (pd.DataFrame): dataframe da comparação entre todos os modelos;

    Returns:
        pd.DataFrame: retorna o dataframe resultante com os erros;
    """
    
    # Cria uma dataframe vazio #
    comparacaoERRO = pd.DataFrame()
    
    # O dataframe criado recebe a mesma coluna de casos de teste #
    comparacaoERRO["casos_teste"] = comparacaoChurn["casos_teste"]
    
    ########## Cada coluna de um modelo é subtraída pelo valor real e arredondada para 10 casas decimais ####################
    
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

    #########################################################################################################################

    # Salva o dataframe em um arquivo CSV #
    ax._salvaArquivo( comparacaoERRO, "churnERRO.csv" )
    
    # Retorna o dataframe #
    return comparacaoERRO

#############################################################################################################################
