######################################       CÁLCULO DE CHURN        #################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      ######################################

import pandas as pd
import numpy as np
import csv

###########################################         AUXILIARES        ################################################

# Função que define os tipos de cada coluna, elimina as colunas desnecessárias e acrescenta colunas. #
def _defineDataframe( cdf: pd.DataFrame ) -> pd.DataFrame:
    """
    Manipula o dataframe de transações escolhendo as colunas e definindo seus tipos;

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
    
    Returns:
        pd.DataFrame: retorna o novo dataframe;
    """
    novocdf = pd.DataFrame()
    
    # Seleciona a coluna id dos clientes"
    novocdf["id_cliente"] = cdf["id_cliente"]
    
    # Cria uma nova coluna de data como string #
    novocdf["date"] = cdf["char_date"].astype( str )
    
    # Define a coluna de data como tipo data #
    novocdf["date"] = pd.to_datetime( novocdf["date"], format = "%Y%m%d" )
    
    
    return novocdf

# Função que calcula a quantidade de clientes no Dataframe. #
def _totalClientes( cdf: pd.DataFrame ) -> int:
    """
    Calcula a quantidade de clientes diferentes em um dataframe de transações;

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;

    Returns:
        int: retorna o total de cliente únicos com base no id;
    """
    # Retorna o total de clientes únicos no arquivo de transação #
    return len( cdf["id_cliente"].unique() )

# Função que transforma a matriz para um DataFrame #
def _matrizParaDataframe( matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex ) -> pd.DataFrame:
    """
    Transforma uma matriz para um dataframe;

    Args:
        matIdChurn (np.ndarray): matriz de clientes por períodos preenchida;
        datesVector (pd.DatetimeIndex): vetor das datas dos períodos;

    Returns:
        pd.DataFrame: retorna uma dataframe da matriz com o cabeçalho de datas;
    """
    # Transforma a matriz em dataframe #
    df_id_churn = pd.DataFrame( matIdChurn )

    # Coloca as colunas como string do vetor de datas #
    df_id_churn.columns = datesVector[:-1].astype( str )

    # Retorna o dataframe da matriz #
    return df_id_churn

######################################################################################################################

#####################################         MANIPULADORES DE ARQUIVO        ########################################

# Função que lê um arquivo csv e salva como um Dataframe. #
def _lerArquivo( arquivo: str ) -> pd.DataFrame:
    """
    Lê um arquivo de CSV e salva em uma variável de dataframe;

    Args:
        arquivo (str): nome do arquivo de transações;

    Returns:
        pd.DataFrame: retorna o arquivo como um dataframe;
    """
    # Define nome das colunas para o arquivo #
    nomes_colunas = ["id_cliente", "char_date", "categoria", "valor"]
    
    # Cria um dataframe do arquivo com as colunas renomeadas #
    cdf = pd.read_csv( arquivo, sep="\s+", names=nomes_colunas )
    
    # Define o formado do dataframe #
    cdf = _defineDataframe( cdf )

    # Retorna o dataframe do arquivo #
    return cdf

# Função que salva o Dataframe em um arquivo. #
def _salvaArquivo( churn: pd.DataFrame, nomeArquivo: str ) -> None:
    """
    Salva um dataframe em um arquivo CSV;

    Args:
        churn (pd.DataFrame): o dataframe de churn resultante;
        nomeArquivo (str): nome do arquivo para salvar o dataframe;
    """
    
    churn.to_csv( nomeArquivo, index = False, header = True, quoting = csv.QUOTE_NONNUMERIC )

######################################################################################################################

#########################################         CONSTRUTORES        ################################################

# Função que controi o vetor de datas de inicio de cada período. #
def _controiVetorDatas( cdf: pd.DataFrame, totalPeriodos: int ) -> pd.DatetimeIndex:
    """
    Constroi um vetor de datas de períodos com base em um intervalo de transações e quantidade de períodos desejados;

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        pd.DatetimeIndex: retorna um vetor de datas;
    """
    # Cria uma vetor de datas com base no intervalo da maior e da menor data do arquivo de transações e na quantidade de períodos escolhida #
    dates_vector = pd.date_range( start = min( cdf["date"] ), end = max( cdf["date"] ), periods = totalPeriodos )
    
    # Salva a última data # 
    last_date = dates_vector[-1]
    
    # Soma um dia na última data #
    last_date = last_date + pd.DateOffset( days = 1 )
    
    # Substitui a última data do vetor pela data somada com um dia, para que a última data esteja incluída no intervalo #
    dates_vector = dates_vector[:-1].append( pd.DatetimeIndex( [last_date] ) )

    # Retorna oo vetor de datas #
    return dates_vector

# Função que constroi a matriz de clientes por períodos preenchida com zeros. #
def _constroiMatrizClientePorPeriodo( cdf: pd.DataFrame, totalPeriodos: int ) -> np.ndarray:
    """
    Constroi uma matriz de clientes por períodos;

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        np.ndarray: retorna a matriz de clientes por períodos preenchida com zeros;
    """
    
    # Cria uma matriz preenchida com zeros #
    mat_id_churn = np.zeros( ( _totalClientes( cdf ), totalPeriodos-1 ) )

    # Retorna a matriz de clientes por período #
    return mat_id_churn

# Função que constroi o Dataframe do churn. #
def _constroiChurn( media: pd.Series, cdf: pd.DataFrame ) -> pd.DataFrame:
    """
    Constroi o dataframe do churn;

    Args:
        media (pd.Series): serie de valores de churn já calculados;
        cdf (pd.DataFrame): dataframe do arquivo de transações;

    Returns:
        pd.DataFrame: retorna um dataframe de churn com id do cliente e valores de churn;
    """
    # Cria uma dataframe com os valores de churn #
    churn = pd.DataFrame( media )
    
    # Cria uma coluna de id dos diferentes clientes do arquivo de transações # 
    churn['id'] = cdf['id_cliente'].unique().astype( str )
    
    # Renomeia a coluna de valores de churn #
    churn = churn.rename( columns = {churn.columns[0]: 'churn'} )
    
    # Retorna o dataframe de id de cliente e valor de churn # 
    return churn[['id', 'churn']]

######################################################################################################################

###############################       CALCULADORES DO DENOMINADOR DA MÉDIA        ######################################

# Função que retorna um vetor com o valor total que irá do denominador no calculo da média baseado na recência do cliente. #
def _calculaRecenciaCliente( matIdChurn: np.ndarray ) -> list:
    """
    Calcula o valor do denominador do modelo de média por recência de cada cliente;

    Args:
        matIdChurn (np.ndarray): matriz de clientes por períodos preenchida;
        datesVector (pd.DatetimeIndex):  vetor das datas dos períodos;

    Returns:
        list: retorna uma lista dos valores do denominador para o cálculo da média de cada cliente individualmente;
    """
    
    # Salva o número de linhas e número de colunas da matriz #
    num_linhas, num_colunas = matIdChurn.shape

    # Cria uma lista vazia #
    media_por_cliente = []
    
    # Percorre a matriz #
    for j in range( num_linhas ):
        # Começa o valor de compra com zero #
        comeca_compra = 0
        # Adiciona um espaço na lista com valor zero #
        media_por_cliente.append(0)
        
        for i in range( num_colunas ):
            # Se o valor de compra estiver como 1 quer dizer que já fez alguma compra #
            if comeca_compra:
                # A média desse cliente é somada mais 1 #
                media_por_cliente[j] += 1
            # Se o valor de compra ainda estiver zero quer dizer que ainda não foi feita uma compra #
            
            # Se o valor da posição estiver com valor 1 então é a sua primeira compra #
            elif matIdChurn[j][i] == 1:
                # O valor de compra se torna 1, pois foi feita uma compra #
                comeca_compra = 1
                # A média desse cliente é somada mais 1 #
                media_por_cliente[j] += 1
    
    # Retorna a lista de média de cada cliente #
    return media_por_cliente

# Função que calcula o valor do denominador na média ponderada linear. #
def _calculaLinear( totalPeriodos: int ) -> float:
    """
    Calcula o valor do denominador do modelo de média linear;

    Args:
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        float: retorna o valor do denominador para o cálculo da média linear para todos os clientes;
    """
    
    # PA
    # Sn = n( a1 + an ) / 2
    return ( totalPeriodos - 1 ) * ( totalPeriodos ) / 2.0

# Função que calcula o valor do denominador na média ponderada exponencial de qualquer base. #
def _calculaExponencial( totalPeriodos: int, base: float ) -> float:
    """
    Calcula o valor do denominador do modelo de média exponencial;

    Args:
        totalPeriodos (int): total de períodos de separação de dados desejada;
        base (float): base desejada para o cálculo exponencial;

    Returns:
        float: retorna o valor do denominador para o cálculo da média exponencial para todos os clientes;
    """
    
    # PG
    # Sn = a1( q**n - 1 ) / (q-1)
    return ( base ) * ( base ** ( totalPeriodos-1 ) - 1 ) / ( base - 1 )

# Função que calcula o valor do denominador na média simples. #
def _calculaSimples( totalPeriodos: int ) -> int:
    """
    Calcula o valor do denominador do modelo de média simples;

    Args:
        totalPeriodos (int): total de períodos de separação de dados desejada;

    Returns:
        int: retorna o valor do denominador para o cálculo da média simples para todos os clientes;
    """
    
    return totalPeriodos - 1

######################################################################################################################

##################################         PREENCHEDORES DE MATRIZ        ############################################

# Função que preenche a matriz com uma base escolhida nas datas de compras que correspondem a determinado período. #
def _preencheMatriz( matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex, cdf: pd.DataFrame, base: float = 1 ) -> None:
    """
    Preeche a matriz de clientes por períodos, marcando a coluna que corresponde ao período de uma transação realizada;

    Args:
        matIdChurn (np.ndarray):  matriz de clientes por períodos;
        datesVector (pd.DatetimeIndex): vetor das datas dos períodos;
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        base (float, optional): base usada para preencher a matriz.
                0 -> para preencher linear;
                1 -> para preencher com 1 independente da coluna (Defaults);
                n -> [!= 1 e != 0] para preencher exponencialmente com a base n e com o expoente linear;
    """
    # Percorre o vetor de datas #
    for i in range( len( datesVector ) - 1 ):
        # Percorre o arquivo de transações #
        for j in range( len( cdf['id_cliente'] ) ):
    
            # Se a transação estiver entre a data atual incluída e aproxima não incluída, então o cliente fez compra nesse período #
            if ( ( datesVector[i] <= cdf.loc[j, 'date'] ) and ( cdf.loc[j, 'date'] < datesVector[i + 1] ) ):
                # f(a,b) => b, se a = 0 e a**b, se a != 0;
                # f(a,b) = a**b + b * !a;

                # Na posição do cliente na matriz é colocado o valor de preenchimento do modelo desejado #
                matIdChurn[cdf.loc[j, 'id_cliente'] - 1, i] = ( base ** ( i + 1 ) + ( i + 1 ) * ( not base ) )

######################################################################################################################

################################         CALCULADORES DE CHURN        ################################################

# Função eu calcula o valor do Churn Binário de cada cliente e salva em um novo Dataframe. #
def _calculaChurnBinario( dfIdChurn: pd.DataFrame, cdf: pd.DataFrame ) -> pd.DataFrame:
    """
    Calcula o churn binário;

    Args:
        dfIdChurn (pd.DataFrame): dataFrame com os valores da matriz preenchidos;
        cdf (pd.DataFrame): dataframe do arquivo de transações;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn Binário;
    """
    # Calcula o valor de churn pela média arredondada #
    media = (1 - dfIdChurn.mean( axis = 1 ) ).round().astype( int )
    
    # Constroi o datafram de churn #
    churn = _constroiChurn( media, cdf )

    # Retorna o dataframe de churn #
    return churn

# Função eu calcula o valor de qualquer Churn Ponderado de cada cliente e salva em um novo Dataframe.#
def _calculaChurnInterno( dfIdChurn: pd.DataFrame, cdf: pd.DataFrame, valorMedia: float ) -> pd.DataFrame:
    """ 
    Calcula o churn de qualquer modelo, exceto o binário e o recente;

    Args:
        dfIdChurn (pd.DataFrame): dataFrame com os valores da matriz preenchidos;
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        valorMedia (float): valor do denominador calculado por outras funções;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """
    # Calcula o valor de churn pela média #
    media = 1 - ( dfIdChurn.sum( axis = 1 ) / valorMedia )
    
    # Ajusta a resposta para 10 casa decimais para evitar erros de float 64 #
    media = media.round(10).abs()
    
    # Constroi o datafram de churn #
    churn = _constroiChurn( media, cdf )

    # Retorna o dataframe de churn #
    return churn

# Função eu calcula o valor do Churn Rencente de cada cliente e salva em um novo Dataframe.#
def _calculaChurnInternoR( dfIdChurn: pd.DataFrame, cdf: pd.DataFrame, valorMedia: pd.Series ) -> pd.DataFrame:
    """ 
    Calcula o churn pelo modelo recente;

    Args:
        dfIdChurn (pd.DataFrame): dataFrame com os valores da matriz preenchidos;
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        valorMedia (float): valor do denominador calculado por outras funções;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """
    # Calcula o valor de churn pela média de cada cliente de acordo com o seu valor na lista #
    # Caso o valor desse clinte seja zero, o resultado final dele será zero #
    media = 1 - ( dfIdChurn.sum(axis=1) / [v if v != 0 else 1 for v in valorMedia] )
    
    # Constroi o datafram de churn #
    churn = _constroiChurn( media, cdf )

    # Retorna o dataframe de churn #
    return churn


# Função que calcula o churn com base em um arquivo de transação com qualquer um dos modelos disponíveis. #
def calculaChurn( arquivo: str, modelo: str = "simples", periodos: int = 10, base: float = 2 ) -> pd.DataFrame:
    """
    Calcula a probabilidade de churn em qualquer modelo com base em um arquivo de transações;

    Args:
        arquivo (str): arquivo de transações;
        modelo (str, optional): nome do modelo desejado;
            modelo= "binario";
            modelo= "simples" (Defaults);
            modelo= "linear";
            modelo= "exponencial";
            modelo= "recente";
        periodos (int, optional): total de períodos de separação de dados desejada 
        (Defaults to 10);
        (Preencher com valores maiores que 1);
        base (float, optional): base desejada para o cálculo exponencial;
        (Preencher com valores maiores que 1);
        (Defaults to 2);

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """

    # Leitura do DataFrame de transação #
    cdf = _lerArquivo( arquivo )

    # Construção do vetor de data inicial de cada período #
    dataVector = _controiVetorDatas( cdf, periodos )

    # Constroi a matriz de clientes por período #
    matriz = _constroiMatrizClientePorPeriodo( cdf, periodos )

    ###################### Caso o modelo seja o Binário #############################
    if ( modelo == "binario" ):
        # Preenche a matriz
        _preencheMatriz( matriz, dataVector, cdf, 1 )

        # Transforma a matriz para um DataFrame #
        dt = _matrizParaDataframe( matriz, dataVector )
        
        # Calcula o churn #
        churn = _calculaChurnBinario( dt, cdf )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnBinario.csv" )

    #################### Caso o modelo seja o Simples ##############################
    elif ( modelo == "simples" ):
        # Preenche a matriz
        _preencheMatriz( matriz, dataVector, cdf, 1 )

        # Transforma a matriz para um DataFrame #
        dt = _matrizParaDataframe( matriz, dataVector )
        
        # Calcula o valor do denominador #
        valorMedia = _calculaSimples( periodos )
        
        # Calcula o churn #
        churn = _calculaChurnInterno( dt, cdf, valorMedia )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnSimples.csv" )

    ##################### Caso o modelo seja o Linear #############################
    elif( modelo == "linear" ):
        # Preenche a matriz
        _preencheMatriz( matriz, dataVector, cdf, 0 )

        # Transforma a matriz para um DataFrame #
        dt = _matrizParaDataframe( matriz, dataVector )
        
        # Calcula o valor do denominador #
        valorMedia = _calculaLinear( periodos )
        
        # Calcula o churn #
        churn = _calculaChurnInterno( dt, cdf, valorMedia )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnLinear.csv" )

    ################### Caso o modelo seja o exponencial #########################
    elif ( modelo == "exponencial" ):
        # Preenche a matriz
        _preencheMatriz( matriz, dataVector, cdf, base )

        # Transforma a matriz para um DataFrame #
        dt = _matrizParaDataframe( matriz, dataVector )
        
        # Calcula o valor do denominador #
        valorMedia = _calculaExponencial( periodos, base )
        
        # Calcula o churn #
        churn = _calculaChurnInterno( dt, cdf, valorMedia )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnExponencial.csv" )

    ################## Caso o modelo seja o Rencente ############################
    elif ( modelo == "recente" ):
        # Preenche a matriz
        _preencheMatriz( matriz, dataVector, cdf, 1 )

        # Transforma a matriz para um DataFrame #
        dt = _matrizParaDataframe( matriz, dataVector )
        
        # Calcula o valor do denominador #
        valorMedia = _calculaRecenciaCliente( matriz )
        
        # Calcula o churn #
        churn = _calculaChurnInternoR( dt, cdf, valorMedia )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnRecente.csv" )

    ################# Caso o modelo escolhido não exista #########################
    else:
        churn = None
    
    # Retorna o dataframe resultante #
    return churn

######################################################################################################################