######################################       CÁLCULO DE CHURN        #################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      ######################################

import pandas as pd
from numpy import e
import numpy as np
from dateutil.parser import parse
import csv
import os
import time
import sys
import bisect

###########################################         AUXILIARES        ################################################

def _verificarTipoArquivo(caminho_arquivo) -> int:
    # Obter a extensão do arquivo
    _, extensao = os.path.splitext(caminho_arquivo)
    
    # Verificar se a extensão é .txt ou .csv
    if extensao == '.txt':
        return 0
    elif extensao == '.csv':
        return 1
    else:
        return 2


# Função que define as colunas importantes e seu tipo para o cálculo. #
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
    novocdf["date"] = pd.to_datetime(novocdf["date"], format="%Y-%m-%d")

    print(novocdf)
    
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
    nomes_colunas = ["id_cliente", "char_date", "valor"]
    
    # Verifica o tipo do arquivo #
    b = _verificarTipoArquivo(arquivo)

    # Cria um dataframe do arquivo com as colunas renomeadas #
    if b == 1:
        cdf = pd.read_csv( arquivo, header=0, index_col=0 )
    else:
        cdf = pd.read_csv( arquivo, sep="\s+", header=0 )
    
    print(cdf)
    
    cdf.columns = nomes_colunas
    
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
    
    churn.to_csv( nomeArquivo, header = True, quoting = csv.QUOTE_NONNUMERIC )

######################################################################################################################

#########################################         CONSTRUTORES        ################################################

# Função que controi o vetor de datas de inicio de cada período. #
def _controiVetorDatas( cdf: pd.DataFrame, frequencia: str, dataInicial: str, dataFinal: str ) -> pd.DatetimeIndex:
    """
    Constroi um vetor de datas de períodos com base em um intervalo de de datas e uma frequência;

    Args:
        cdf (pd.DataFrame): dataframe do arquivo de transações;
        frequencia (str): tamanho entre o períodos (H, D, W, M, ou Y);
        dataInicial (str): data de inicio da avaliação;
        dataFinal (str): data final da avaliação;

    Returns:
        pd.DatetimeIndex: retorna um vetor de datas;
    """
    # Caso não seja passada uma data inicial #
    if dataInicial == None:
        dataInicial = min( cdf["date"] )
        
    # Convertendo a data inicial passada para o formato padrão reconhecido #
    else:  
        dataInicial = parse(dataInicial)
        
    # Caso não seja passada uma data inicial #
    if dataFinal == None:
        dataFinal = max( cdf["date"] ) + pd.DateOffset( days = 1 ) # Somado um dia na data final #
        
    # Convertendo a data final passada para o formato padrão reconhecido #
    else:
        dataFinal = parse(dataFinal) + pd.DateOffset( days = 1 ) # Somado um dia na data final #
    
    # Cria uma vetor de datas com base me uma data inicial, uma data final e na frequência escolhida #
    dates_vector = pd.date_range( start = dataInicial, end = dataFinal, freq = frequencia )
    
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
    total = 0
    row_copy = row.copy()
    
    for col, value in row_copy.iloc[:-1].items():
        # Se o valor de compra estiver como 1 quer dizer que já fez alguma compra #
        if comeca_compra:
            # A média desse cliente é somada mais 1 #
            total += 1
        # Se o valor de compra ainda estiver zero quer dizer que ainda não foi feita uma compra #
        
        # Se o valor da posição estiver com valor 1 então é a sua primeira compra #
        elif value != 0:
            # O valor de compra se torna 1, pois foi feita uma compra #
            comeca_compra = 1
            # A média desse cliente é somada mais 1 #
            total += 1
    return total

# Função que calcula o valor do denominador na média ponderada linear. #
def _calculaLinear( tamanho: int, inicial: int = 1 ) -> float:
    """
    Calcula o valor do denominador do modelo de média linear;

    Args:
        tamanho (int): total de períodos de separação das datas;
        inicial (int): valor do primeiro ponderamento do intervalo atual;

    Returns:
        float: retorna o valor do denominador para o cálculo da média linear para cada intervalo;
    """
    
    # PA
    # Sn = n( a1 + an ) / 2
    return ( tamanho ) * ( inicial + ( inicial + ( tamanho - 1 ) ) ) / 2.0

# Função que calcula o valor do denominador na média ponderada exponencial de qualquer base. #
def _calculaExponencial( tamanho: int, base: float, inicial: int = 1 ) -> float:
    """
    Calcula o valor do denominador do modelo de média exponencial;

    Args:
        tamanho (int): total de períodos de separação de dados desejada;
        base (float): base desejada para o cálculo exponencial;
        inicial (int): valor do expoente do primeito ponderamento do intervalo atual;

    Returns:
        float: retorna o valor do denominador para o cálculo da média exponencial para cada intervalo;
    """
    
    # PG
    # Sn = a1( q**n - 1 ) / (q - 1)
    return ( base**inicial ) * ( base ** ( inicial + (tamanho - 1) ) - 1 ) / ( base - 1 )


######################################################################################################################

######################################         PREENCHEDOR       #####################################################

def _transformaTabela( row: pd.Series, base: float = 0 ) -> pd.Series:
    tamanho = int( row["Dmedia"] )
    inicial = int( len(row) - tamanho - 2 )

    # Cria um vetor de ponderações #
    multiplicadores = [( base ** ( i + 1 ) + ( i + 1 ) * ( not base ) ) for i in range(tamanho)] if tamanho != 0 else [1]
    
    # Faz uma cópia dos dados antes de modificar
    row_copy = row.copy()
    row_copy[inicial:-2] *= multiplicadores
    
    return row_copy


def _preencheTabela(row: pd.Series, tabela: pd.DataFrame, datesVector: pd.DatetimeIndex) -> None:
    """
    Preenche a tabela de clientes por períodos, marcando a coluna que corresponde ao período de uma transação realizada.

    Args:
        row (pd.Series): linha do DataFrame de transações.
        tabela (pd.DataFrame): tabela de clientes por período a ser preenchida.
        datesVector (pd.DatetimeIndex): vetor das datas dos períodos.
    """
    
    # Encontre o índice da data que corresponde à transação usando busca binária
    idx = np.searchsorted(datesVector, row["date"], side='right') - 1

    # Certifique-se de que o índice está dentro dos limites válidos
    if (0 <= idx) and (idx < (len(datesVector) - 1)):
        period_date = datesVector[idx]

        # Preencha a tabela
        tabela.at[row.name, period_date] = 1

######################################################################################################################

################################         CALCULADORES DE CHURN        ################################################

# Função para calcular a média personalizada
def calculaMedia( block: list, block_size: float ):
    return 1 - ( block.sum() / ( block_size if block_size != 0 else 1 ) )

# Função auxiliar para aplicar a lógica em cada cliente
def process_customer(row, modelo: int ):
    block_size = row['Media_Intervalos']*2  # Obter o tamanho do bloco
    purchases = row.iloc[:-2]  # Remover as últimas duas colunas

    # Dividir os períodos de compra em blocos com base no block_size
    blocks = [purchases[i:i + int(block_size)] for i in range(0, len(purchases), int(block_size))]

    if modelo == 1: 
        block_means = [calculaMedia(block, _calculaLinear(block_size, i*block_size + 1 )) for i, block in enumerate(blocks)]

    elif modelo == 2: 
        block_means = [calculaMedia(block, _calculaExponencial(block_size, 2, i*block_size + 1 )) for i, block in enumerate(blocks)]

    elif modelo == 3: 
        block_means = [calculaMedia(block, _calculaExponencial(block_size, np.e, i*block_size + 1 )) for i, block in enumerate(blocks)]

    else: 
        block_means = [calculaMedia(block, block_size) for block in blocks]
    
    # Retornar a média geral das médias dos blocos
    return np.mean( block_means )

# Função para calcular médias com blocos variáveis
def calculate_variable_block_averages( df: pd.DataFrame, modelo: int ):

    # Aplicar a função a cada linha do DataFrame
    result = df.apply( lambda row: process_customer( row, modelo ), axis=1 )
    
    return result

# Função eu calcula o valor do Churn Rencente de cada cliente e salva em um novo Dataframe.#
def _calculaChurnInternoR( tabela: pd.DataFrame, modelo: int ) -> pd.DataFrame:
    """
    Calcula o churn pelo modelo recente;

    Args:
        tabela (pd.DataFrame): dataFrame de clientes por períodos com a coluna 'media' adicionada;

    Returns:
        pd.DataFrame: retorna o dataFrame resultante do churn calculado;
    """
    tabela['Media_Intervalos'] = np.ceil(tabela['Media_Intervalos'])
    print(tabela)
    # Calcula o valor de churn pela média de cada cliente de acordo com o seu valor na lista #
    media = calculate_variable_block_averages( tabela, modelo )

    # Constroi o datafram de churn #
    churn = _constroiChurn( media, tabela )

    # Retorna o dataframe de churn #
    return churn

def report_time(start_time):
        elapsed_time = time.time() - start_time
        sys.stdout.write(f"\rTempo decorrido: {elapsed_time:.2f} segundos")
        print()
        sys.stdout.flush()

def calcular_intervalos(row):
    indices = np.where(row[1:-1] == 1)[0]  # Ignorar a coluna 'id_cliente' e "Dmedia"
    if len(indices) > 1:
        intervalos = np.diff(indices)
        return np.mean(intervalos)
    else:
        return 0

# Função que calcula o churn com base em um arquivo de transação com qualquer um dos modelos disponíveis. #
def calculaAllChurn( arquivo: str, dataInicial: str = None, dataFinal: str = None, freq: str = "M" ) -> pd.DataFrame:
    """
    Calcula a probabilidade de churn de clientes de TODOS os modelos com base em um arquivo de transações e períodos;

    Args:
        arquivo (str): arquivo de transações;
        
        dataInicial (str): data inicial para os períodos;
        dataFinal (str): data final para os períodos;
        
            --> (As datas podem ser passadas com qualquer separador conhecido entre datas e em qualquer ordem);
            
            --> (É mais aconselhável o modelo MM/DD/YYYY para que não ocorra confusão entre o dia e o mês, pois por padrão é reconhecido o mês primeiro);
            
            --> (Defaults to None) - É utilizado a data inicial ou a final do próprio dataset de transações;
        
        freq (str): tamanho de cada período de intervalo;
        
            --> (A frequência é passada como uma letra);
            
            --> (H - hora, D - day, W - week, M - month, Y - year);
            
            --> (É possível passar uma quantidade em cada tipo adicionando um número antes da letra);
            
            --> (Por padrão a data inicial é ajustada para o final do mês, final do ano ou da semana);
            
            --> (Para mudar para ser ajustado para o começo do mês ou do ano, adicione um "S" após a letra da frequência);
            
            --> (Defaults to M);

    Returns:
        pd.DataFrame: retorna a probabilidade de churn de todos os modelos;
    """
    start_time = time.time()

    # Leitura do DataFrame de transação #
    cdf = _lerArquivo( arquivo )

    # Construção do vetor de data inicial de cada período #
    dataVector = _controiVetorDatas( cdf, freq, dataInicial, dataFinal )

    # Constroi a tabela de clientes por período #
    tabela = _constroiTabelaClientePorPeriodo(cdf, dataVector)

    print(tabela)
    
    cdf.set_index("id_cliente", inplace=True)

    report_time(start_time)

    # Preenche a tabela #
    cdf.apply( lambda row: _preencheTabela(row, tabela, dataVector), axis=1 )

    print(tabela)

    report_time(start_time)

    print(tabela)

    # Salva a tabela em um arquivo #
    #_salvaArquivo( tabela, "../Analises/Arquivos/tabelaTotalBANK.csv" )
    
    # Cria uma coluna de valor do denominador da média para cada cliente preenchida com zero #
    tabela["Dmedia"] = 0
    print(tabela)
    
    # Calcula a quantidade de períodos #
    tabela["Dmedia"] = tabela.apply( lambda row: _calculaRecenciaCliente(row), axis=1 )

    report_time(start_time)

    tabela['Media_Intervalos'] = tabela.apply(calcular_intervalos, axis=1)

    report_time(start_time)

    print("Linear")
    print(tabela)
    ########################### Modelo Linear ###########################################
    
    # Multiplica a tabela pela sua ponderação #
    tabelaNova = tabela.apply( lambda row: _transformaTabela( row ), axis=1 )
    print(tabelaNova)

    report_time(start_time)

    # Calcula o churn #
    churn = _calculaChurnInternoR( tabelaNova, 1 )

    report_time(start_time)
    
    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    resultadoChurn = churn
    print(churn)
    # Renomeia a coluna de churn para o nome do modelo #
    resultadoChurn.rename(columns = { 'churn': "churnLinear" }, inplace = True )
    
    ####################################################################################
    print("Exponencial base 2")
    print(tabela)
    ####################### Modelo exponencial de base 2 ################################

    report_time(start_time)

    # Multiplica a tabela pela sua ponderação #
    tabelaNova = tabela.apply(lambda row: _transformaTabela( row, 2 ), axis=1)
    print(tabelaNova)

    report_time(start_time)

    # Calcula o valor do denominador #
    tabelaNova["Dmedia"] = tabelaNova["Dmedia"].apply( lambda row: _calculaExponencial(row, 2))
    print(tabelaNova)

    report_time(start_time)

    # Calcula o churn #
    churn = _calculaChurnInternoR( tabelaNova, 2 )
    print(churn)

    report_time(start_time)

    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    resultadoChurn = pd.merge( resultadoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    resultadoChurn.rename(columns = { 'churn': "churnExponencial_2" }, inplace = True )
    
    ####################################################################################
    print("Exponencial base e")
    print(tabela)
    ####################### Modelo exponencial de base e ################################
    
    report_time(start_time)

    # Multiplica a tabela pela sua ponderação #
    tabelaNova = tabela.apply(lambda row: _transformaTabela( row, e ), axis=1)
    print(tabelaNova)

    report_time(start_time)

    # Calcula o valor do denominador #
    tabelaNova["Dmedia"] = tabelaNova["Dmedia"].apply( lambda row: _calculaExponencial(row, e))
    print(tabelaNova)

    report_time(start_time)

    # Calcula o churn #
    churn = _calculaChurnInternoR( tabelaNova, 3 )
    print(churn)

    report_time(start_time)

    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    resultadoChurn = pd.merge( resultadoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    resultadoChurn.rename( columns = { 'churn': "churnExponencial_e" }, inplace = True )
    
    ####################################################################################
    print("Recente")
    print(tabela)
    ########################## Modelo Recente ##########################################
    
    report_time(start_time)

    # Calcula churn #
    churn = _calculaChurnInternoR( tabela, 4 )
    print(churn)

    report_time(start_time)

    # Faz um merge do dataframe até então com o dataframe de churn calculado com base no id #
    resultadoChurn = pd.merge( resultadoChurn, churn, on = "id" )
    
    # Renomeia a coluna de churn para o nome do modelo #
    resultadoChurn.rename( columns = { 'churn': "churnRecente" }, inplace = True )
    
    ####################################################################################

    resultadoChurn['Media_Intervalos'] = resultadoChurn['id'].map(tabela['Media_Intervalos'])
    print(resultadoChurn)

    report_time(start_time)
    print()

    # Salva o dataframe em um arquivo CSV #
    _salvaArquivo( resultadoChurn, "../../Analises/Arquivos/churnResultadoTransNovo.csv" )
    
    report_time(start_time)
    print()

    # Retorna o dataframe #
    return resultadoChurn

######################################################################################################################