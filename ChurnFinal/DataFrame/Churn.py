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

# Função que constroi a tabela de clientes por períodos preenchida com zeros. #
def _constroiTabelaClientePorPeriodo( cdf: pd.DataFrame, dates_vector: pd.DatetimeIndex ) -> pd.DataFrame:
    """
    Constroi uma tabela de clientes por períodos;

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        dates_vector (pd.DatetimeIndex): vetor de datas dos períodos;

    Returns:
        pd.DataFrame: retorna a tabela de clientes por períodos preenchida com zeros;
    """
    # Cria um dataframe com o índice de id de clientes e colunas com a data dos períodos #
    tabela_idCliente = pd.DataFrame(index=cdf["id_cliente"].unique(), columns= dates_vector[:-1])
    
    # Preenche a tabela com zeros #
    tabela_idCliente = tabela_idCliente.fillna(0)
    
    return tabela_idCliente

# Função que constroi o Dataframe do churn. #
def _constroiChurn( media: pd.Series, tabela: pd.DataFrame ) -> pd.DataFrame:
    """
    Constroi o dataframe do churn;

    Args:
        media (pd.Series): serie de valores de churn já calculados;
        tabela (pd.DataFrame): dataframe de clientes por períodos;

    Returns:
        pd.DataFrame: retorna um dataframe de churn com id do cliente e valores de churn;
    """
    # Cria uma dataframe com os valores de churn #
    churn = pd.DataFrame( media )
    
    # Cria uma coluna de id dos diferentes clientes do arquivo de transações # 
    churn['id'] = tabela.index
    
    # Renomeia a coluna de valores de churn #
    churn = churn.rename( columns = {churn.columns[0]: 'churn'} )
    
    # Retorna o dataframe de id de cliente e valor de churn # 
    return churn[['id', 'churn']]

######################################################################################################################

###############################       CALCULADORES DO DENOMINADOR DA MÉDIA        ######################################

# Função que retorna um vetor com o valor total que irá do denominador no calculo da média baseado na recência do cliente. #
def _calculaRecenciaCliente( row: pd.DataFrame ) -> None:
    """
    Calcula o valor do denominador do modelo de média por recência de cada cliente;

    Args:
        row (pd.DataFrame): linha da tabela de clientes por períodos com a coluna 'media' adicionada;

    """
    
    # Começa o valor de compra com zero #
    comeca_compra = 0
    row["Dmedia"] = 0
    row_copy = row.copy()
    
    for col, value in row_copy.iloc[:-1].items():
        # Se o valor de compra estiver como 1 quer dizer que já fez alguma compra #
        if comeca_compra:
            # A média desse cliente é somada mais 1 #
            row["Dmedia"] += 1
        # Se o valor de compra ainda estiver zero quer dizer que ainda não foi feita uma compra #
        
        # Se o valor da posição estiver com valor 1 então é a sua primeira compra #
        elif value == 1:
            # O valor de compra se torna 1, pois foi feita uma compra #
            comeca_compra = 1
            # A média desse cliente é somada mais 1 #
            row["Dmedia"] += 1

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
    # Sn = a1( q**n - 1 ) / (q - 1)
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

######################################         PREENCHEDOR       #####################################################

# Função que preenche a tabela com uma base escolhida nas datas de compras que correspondem a determinado período. #
def _preencheTabela( row: pd.Series, tabela: pd.DataFrame, datesVector: pd.DatetimeIndex, base: float = 1 ) -> None:
    """
    Preeche a tabela de clientes por períodos, marcando a coluna que corresponde ao período de uma transação realizada;

    Args:
        row (pd.Series): linha do data frame de transações;
        tabela (pd.DataFrame): tabela de clientes por período a ser preenchida;
        datesVector (pd.DatetimeIndex): vetor das datas dos períodos;
        base (float, optional): base usada para preencher a matriz.
            0 -> para preencher linear;
            1 -> para preencher com 1 independente da coluna (Defaults);
            n -> [!= 1 e != 0] para preencher exponencialmente com a base n e com o expoente linear;
    
    """
    
    # Percorre o vetor de datas #
    for i in range( len( datesVector ) - 1 ):

        # Se a transação estiver entre a data atual incluída e aproxima não incluída, então o cliente fez compra nesse período #
        if ( ( datesVector[i] <= row["date"] ) and ( row["date"] < datesVector[i + 1] ) ):
            # f(a,b) => b, se a = 0 e a**b, se a != 0;
            # f(a,b) = a**b + b * !a;

            # Na linha do cliente na coluna do período que a data corresponde na tabela é colocado o valor de preenchimento do modelo desejado #
            tabela.loc[row.name, datesVector[i]] = ( base ** ( i + 1 ) + ( i + 1 ) * ( not base ) )
            break

######################################################################################################################

################################         CALCULADORES DE CHURN        ################################################

# Função eu calcula o valor do Churn Binário de cada cliente e salva em um novo Dataframe. #
def _calculaChurnBinario( tabela: pd.DataFrame ) -> pd.DataFrame:
    """
    Calcula o churn binário;

    Args:
        tabela (pd.DataFrame): dataFrame de clientes por períodos;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn Binário;
    """
    # Calcula o valor de churn pela média arredondada #
    media = (1 - tabela.mean( axis = 1 ) ).round().astype( int )
    
    # Constroi o datafram de churn #
    churn = _constroiChurn( media, tabela )

    # Retorna o dataframe de churn #
    return churn

# Função eu calcula o valor de qualquer Churn Ponderado de cada cliente e salva em um novo Dataframe.#
def _calculaChurnInterno( tabela: pd.DataFrame, valorMedia: float ) -> pd.DataFrame:
    """
    Calcula o churn de qualquer modelo, exceto o binárioe o recente;

    Args:
        tabela (pd.DataFrame): dataFrame de clientes por períodos;
        valorMedia (float): valor do denominador calculado por outras funções;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """
    # Calcula o valor de churn pela média #
    media = 1 - ( tabela.sum( axis = 1 ) / valorMedia )
    
    # Ajusta a resposta para 10 casa decimais para evitar erros de float 64 #
    media = media.round(10).abs()
    
    # Constroi o datafram de churn #
    churn = _constroiChurn( media, tabela )

    # Retorna o dataframe de churn #
    return churn

# Função eu calcula o valor do Churn Rencente de cada cliente e salva em um novo Dataframe.#
def _calculaChurnInternoR( tabela: pd.DataFrame ) -> pd.DataFrame:
    """
    Calcula o churn pelo modelo recente;

    Args:
        tabela (pd.DataFrame): dataFrame de clientes por períodos com a coluna 'media' adicionada;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """
    # Calcula o valor de churn pela média de cada cliente de acordo com o seu valor na lista #
    # Caso o valor desse clinte seja zero, o resultado final dele será zero #
    media = 1 - ( tabela.iloc[:, :-1].sum(axis=1) / [v if v != 0 else 1 for v in tabela["Dmedia"]] )
    
    # Constroi o datafram de churn #
    churn = _constroiChurn( media, tabela )

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
        (Preencher com valores maiores que 1);
        (Defaults to 10);
        
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

    # Constroi a tabela de clientes por período #
    tabela = _constroiTabelaClientePorPeriodo(cdf, dataVector)
    
    cdf.set_index("id_cliente", inplace=True)

    ###################### Caso o modelo seja o Binário #############################
    if ( modelo == "binario" ):
        # Preenche a tabela #
        cdf.apply( _preencheTabela, args=( tabela, dataVector, 1 ), axis=1 )
        
        # Calcula o churn #
        churn = _calculaChurnBinario( tabela ) 
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnBinario.csv" )

    #################### Caso o modelo seja o Simples ##############################
    elif ( modelo == "simples" ):
        # Preenche a tabela #
        cdf.apply( _preencheTabela, args=( tabela, dataVector, 1 ), axis=1 )
        
        # Calcula o valor do denominador #
        valorMedia = _calculaSimples( periodos )
        
        # Calcula o churn #
        churn = _calculaChurnInterno( tabela, valorMedia )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnSimples.csv" )

    ##################### Caso o modelo seja o Linear #############################
    elif( modelo == "linear" ):
        # Preenche a tabela #
        cdf.apply( _preencheTabela, args=( tabela, dataVector, 0 ), axis=1 )
        
        # Calcula o valor do denominador #
        valorMedia = _calculaLinear( periodos )
        
        # Calcula o churn #
        churn = _calculaChurnInterno( tabela, valorMedia )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnLinear.csv" )

    ################### Caso o modelo seja o exponencial #########################
    elif ( modelo == "exponencial" ):
        # Preenche a tabela #
        cdf.apply( _preencheTabela, args=( tabela, dataVector, base ), axis=1 )
        
        # Calcula o valor do denominador #
        valorMedia = _calculaExponencial( periodos, base )
        
        # Calcula o churn #
        churn = _calculaChurnInterno( tabela, valorMedia )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnExponencial.csv" )

    ################## Caso o modelo seja o Rencente ############################
    elif ( modelo == "recente" ):
        # Preenche a tabela #
        cdf.apply( _preencheTabela, args=( tabela, dataVector, 1 ), axis=1 )
        
        # Cria uma coluna de valor do denominador da média para cada cliente preenchida com zero #
        tabela["Dmedia"] = 0
        
        # Calcula o valor do denominador #
        tabela.apply(_calculaRecenciaCliente, axis=1)
        
        # Calcula o churn #
        churn = _calculaChurnInternoR( tabela )
        
        # Salva em um arquivo CSV #
        _salvaArquivo( churn, "churnRecente.csv" )

    ################# Caso o modelo escolhido não exista #########################
    else:
        churn = None

    # Retorna o dataframe resultante #
    return churn

######################################################################################################################
