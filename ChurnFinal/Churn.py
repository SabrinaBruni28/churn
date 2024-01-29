
######################################       CÁLCULO DE CHURN        #################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      ######################################

import pandas as pd
import numpy as np
import csv

###########################################         AUXILIARES        ################################################

# Função que define os tipos de cada coluna, elimina as colunas desnecessárias e acrescenta colunas. #
def defineDataframe( cdf: pd.DataFrame ) -> None:
    """_summary_

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
    """
    
    cdf.drop( ['categoria','valor'], axis = 1, inplace = True )
    cdf['char_date'] = cdf["char_date"].astype( str )
    cdf["date"] = pd.to_datetime( cdf["char_date"], format = "%Y%m%d" )
    cdf["char_date"] = cdf["date"].dt.strftime( "%Y-%m-%d" )

# Função que calcula a quantidade de clientes no Dataframe. #
def totalClientes( cdf: pd.DataFrame ) -> int:
    """_summary_

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;

    Returns:
        int: retorna o total de cliente únicos com base no id;
    """
    
    return len( cdf["id_cliente"].unique() )

# Função que transforma a matriz para um DataFrame #
def matrizParaDataframe( matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex ) -> pd.DataFrame:
    """_summary_

    Args:
        matIdChurn (np.ndarray): matriz de clientes por períodos preenchida;
        datesVector (pd.DatetimeIndex): vetor das datas dos períodos;

    Returns:
        pd.DataFrame: retorna uma dataframe da matriz com o cabeçalho de datas;
    """
    
    df_id_churn = pd.DataFrame( matIdChurn )
    df_id_churn.columns = datesVector[:-1].astype( str )
    return df_id_churn

######################################################################################################################

#####################################         MANIPULADORES DE ARQUIVO        ########################################

# Função que lê um arquivo csv e salva como um Dataframe. #
def lerArquivo( arquivo: str ) -> pd.DataFrame:
    """_summary_

    Args:
        arquivo (str): nome do arquivo de transações;

    Returns:
        pd.DataFrame: retorna o arquivo como um dataframe;
    """
    
    nomes_colunas = ["id_cliente", "char_date", "categoria", "valor"]
    cdf = pd.read_csv( arquivo, sep="\s+", names=nomes_colunas )
    defineDataframe( cdf )
    return cdf

# Função que salva o Dataframe em um arquivo. #
def salvaArquivo( churn: pd.DataFrame, nomeArquivo: str ) -> None:
    """_summary_

    Args:
        churn (pd.DataFrame): o dataframe de churn resultante;
        nomeArquivo (str): nome do arquivo para salvar o dataframe;
    """
    
    churn.to_csv( nomeArquivo, index = False, header = True, quoting = csv.QUOTE_NONNUMERIC )

######################################################################################################################

#########################################         CONSTRUTORES        ################################################

# Função que controi o vetor de datas de inicio de cada período. #
def controiVetorDatas( cdf: pd.DataFrame, totalPeriodos: int ) -> pd.DatetimeIndex:
    """_summary_

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        pd.DatetimeIndex: retorna um vetor de datas de períodos no intervalo das transações;
    """
    
    dates_vector = pd.date_range( start = min( cdf["date"] ), end = max( cdf["date"] ), periods = totalPeriodos )
    last_date = dates_vector[-1]
    last_date = last_date + pd.DateOffset( days = 1 )
    dates_vector = dates_vector[:-1].append( pd.DatetimeIndex( [last_date] ) )
    return dates_vector

# Função que constroi a matriz de clientes por períodos preenchida com zeros. #
def constroiMatrizClientePorPeriodo( cdf: pd.DataFrame, totalPeriodos: int ) -> np.ndarray:
    """_summary_

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        np.ndarray: retorna uma matriz de clientes por períodos preenchida com zeros;
    """
    
    mat_id_churn = np.zeros( ( totalClientes( cdf ), totalPeriodos-1 ) )
    return mat_id_churn

# Função que constroi o Dataframe do churn. #
def constroiChurn( media: pd.Series, cdf: pd.DataFrame ) -> pd.DataFrame:
    """_summary_

    Args:
        media (pd.Series): serie de valores de churn já calculados;
        cdf (pd.DataFrame): dataframe do arquivo de transações;

    Returns:
        pd.DataFrame: retorna um dataframe de churn com id do cliente e valores de churn;
    """
    
    churn = pd.DataFrame( media )
    churn['id'] = cdf['id_cliente'].unique().astype( str )
    churn = churn.rename( columns = {churn.columns[0]: 'churn'} )
    return churn[['id', 'churn']]

######################################################################################################################

###############################       CALCULADORES DO DENOMINADOR DA MÉDIA        ######################################

# Função que retorna um vetor com o valor total que irá dividir no calculo da média baseado a recência do cliente. #
def calculaRecenciaCliente( matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex ) -> list:
    """_summary_

    Args:
        matIdChurn (np.ndarray): matriz de clientes por períodos preenchida;
        datesVector (pd.DatetimeIndex):  vetor das datas dos períodos;

    Returns:
        list: retorna uma lista dos valores do denominador para o cálculo da média de cada cliente individualmente;
    """
    
    media_por_cliente = []
    for j in range( len( matIdChurn ) ):
        comeca_compra = 0
        media_por_cliente.append(0)
        for i in range( len( datesVector ) - 1 ):
            if comeca_compra:
                media_por_cliente[j] += 1
            elif matIdChurn[j][i] == 1:
                comeca_compra = 1
                media_por_cliente[j] += 1
    return media_por_cliente

# Função que calcula o valor que divide na média ponderada linear. #
def calculaLinear( totalPeriodos: int ) -> float:
    """_summary_

    Args:
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        float: retorna o valor do denominador para o cálculo da média linear para todos os clientes;
    """
    
    # PA
    # Sn = n( a1 + an ) / 2
    return ( totalPeriodos - 1 ) * ( totalPeriodos ) / 2.0

# Função que calcula o valor que divide na média ponderada exponencial de qualquer base. #
def calculaExponencial( totalPeriodos: int, base: float ) -> float:
    """_summary_

    Args:
        totalPeriodos (int): total de períodos de separação de dados desejada;
        base (float): base desejada para o cálculo exponencial;

    Returns:
        float: retorna o valor do denominador para o cálculo da média exponencial para todos os clientes;
    """
    
    # PG
    # Sn = a1( q**n - 1 ) / (q-1)
    return ( base ) * ( base ** ( totalPeriodos-1 ) - 1 ) / ( base - 1 )

# Função que calcula o valor que divide na média simples. #
def calculaSimples( totalPeriodos: int ) -> int:
    """_summary_

    Args:
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        int: retorna o valor do denominador para o cálculo da média simples para todos os clientes;
    """
    
    return totalPeriodos - 1

######################################################################################################################

##################################         PREENCHEDORES DE MATRIZ        ############################################

# Função que preenche a matriz com uma base escolhida nas datas de compras que correspondem a determinado período. #
def preencheMatriz( matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex, cdf: pd.DataFrame, base: float = 1 ) -> None:
    """Preeche a matriz de clientes por períodos, marcando a coluna que corresponde ao período de uma transação realizada;

    Args:
        matIdChurn (np.ndarray):  matriz de clientes por períodos;
        datesVector (pd.DatetimeIndex): vetor das datas dos períodos;
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        base (float, optional): base usada para preencher a matriz.
                0 -> para preencher linear;
                1 -> para preencher com 1 independente da coluna (Defaults);
                n -> [!= 1 e != 0] para preencher exponencialmente com a base n e com o expoente linear;
    """
   
    for i in range( len( datesVector ) - 1 ):
        for j in range( len( cdf['id_cliente'] ) ):
            if ( ( datesVector[i] <= cdf.loc[j, 'date'] ) and ( cdf.loc[j, 'date'] < datesVector[i + 1] ) ):
                # f(a,b) => b, se a = 0 e a**b, se a != 0;
                # f(a,b) = a**b + b * !a;
                matIdChurn[cdf.loc[j, 'id_cliente'] - 1, i] = ( base ** ( i + 1 ) + ( i + 1 ) * ( not base ) )

######################################################################################################################

################################         CALCULADORES DE CHURN        ################################################

# Função eu calcula o valor do Churn Binário de cada cliente e salva em um novo Dataframe. #
def calculaChurnBinario( dfIdChurn: pd.DataFrame, cdf: pd.DataFrame ) -> pd.DataFrame:
    """Calcula o churn binário;

    Args:
        dfIdChurn (pd.DataFrame): dataFrame com os valores da matriz preenchidos;
        cdf (pd.DataFrame): dataframe do arquivo de transações;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn Binário;
    """
    
    media = (1 - dfIdChurn.mean( axis = 1 ) ).round().astype( int )
    churn = constroiChurn(media, cdf)
    return churn

# Função eu calcula o valor de qualquer Churn Ponderado de cada cliente e salva em um novo Dataframe. #
def calculaChurnInterno( dfIdChurn: pd.DataFrame, cdf: pd.DataFrame, valorMedia: float ) -> pd.DataFrame:
    """ Calcula o churn de qualquer modelo, exceto o binário;

    Args:
        dfIdChurn (pd.DataFrame): dataFrame com os valores da matriz preenchidos;
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        valorMedia (float): valor do denominador calculado por outras funções;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """
    
    media = 1 - ( dfIdChurn.sum( axis = 1 ) / valorMedia )
    churn = constroiChurn(media, cdf)
    return churn

def calculaChurn( arquivo: str, modelo: str = "simples", periodos: int = 10, base: float = 1 ) -> pd.DataFrame:
    """Calcula a probabilidade de churn em qualquer modelo com base em um arquivo de transações;

    Args:
        arquivo (str): arquivo de transações;
        modelo (str, optional): nome do modelo desejado;
            modelo= "binario";
            modelo= "simples" (Defaults);
            modelo= "linear";
            modelo= "exponencial";
            modelo= "recente";
        periodos (int, optional): total de períodos de separação de dados desejada (Defaults to 10);
        base (float, optional): base desejada para o cálculo exponencial (Defaults to 1);

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """

    # Leitura do DataFrame de transação
    cdf = lerArquivo( arquivo )

    # Construção do vetor de data inicial de cada período
    dataVector = controiVetorDatas( cdf, periodos )

    # Constroi a matriz de clientes por período
    matriz = constroiMatrizClientePorPeriodo( cdf, periodos )

    # Preenche a matriz
    preencheMatriz( matriz, dataVector, cdf, base )

    # Transforma a matriz para um DataFrame
    dt = matrizParaDataframe( matriz, dataVector )

    # Caso o modelo seja o Binário
    if ( modelo == "binario" ):
        churn = calculaChurnBinario( dt, cdf )
        salvaArquivo(churn, "churnBinario.csv")

    # Caso o modelo seja o Simples
    elif ( modelo == "simples" ):
        valorMedia = calculaSimples( periodos )
        churn = calculaChurnInterno( dt, cdf, valorMedia )
        salvaArquivo(churn, "churnSimples.csv")

    # Caso o modelo seja o Linear
    elif( modelo == "linear" ):
        valorMedia = calculaLinear( periodos )
        churn = calculaChurnInterno( dt, cdf, valorMedia )
        salvaArquivo(churn, "churnLinear.csv")

    # Caso o modelo seja o exponencial
    elif ( modelo == "exponencial" ):
        valorMedia = calculaExponencial( periodos, base )
        churn = calculaChurnInterno( dt, cdf, valorMedia )
        salvaArquivo(churn, "churnExponencial.csv")

    # Caso o modelo seja o Rencente
    elif ( modelo == "recente" ):
        valorMedia = calculaRecenciaCliente( matriz, dataVector )
        churn = calculaChurnInterno( dt, cdf, valorMedia )
        salvaArquivo(churn, "churnRecente.csv")

    return churn


######################################################################################################################