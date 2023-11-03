
######################################       CÁLCULO DE CHURN        #################################################

#############################       Autor(a): Sabrina Bruni de Souza Faria      ######################################

import pandas as pd
import numpy as np
import csv

###########################################         AUXILIARES        ################################################

# Função que define os tipos de cada coluna, elimina as colunas desnecessárias e acrescenta colunas. #
def defineDataframe(cdf: pd.DataFrame):
    cdf.drop( ['categoria','valor'], axis=1, inplace=True )
    cdf['char_date'] = cdf["char_date"].astype( str )
    cdf["date"] = pd.to_datetime( cdf["char_date"], format="%Y%m%d" )
    cdf["char_date"] = cdf["date"].dt.strftime( "%Y-%m-%d" )

# Função que calcula a quantidade de clientes no Dataframe. #
def totalClientes(cdf: pd.DataFrame) -> int:
    return len( cdf["id_cliente"].unique() )

# Função que transforma a matriz para um DataFrame #
def matrizParaDataframe(matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex) -> pd.DataFrame:
    df_id_churn = pd.DataFrame(matIdChurn)
    df_id_churn.columns = datesVector[:-1].astype(str)
    return df_id_churn

######################################################################################################################

#####################################         MANIPULADORES DE ARQUIVO        ########################################

# Função que lê um arquivo csv e salva como um Dataframe. #
def lerArquivo(arquivo: str) -> pd.DataFrame:
    nomes_colunas = ["id_cliente", "char_date", "categoria", "valor"]
    cdf = pd.read_csv( arquivo, sep="\s+", names=nomes_colunas )
    defineDataframe( cdf )
    return cdf

# Função que salva o Dataframe em um arquivo. #
def salvaArquivo(churn: pd.DataFrame, nomeArquivo: str):
    churn.to_csv(nomeArquivo, index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
    #churn.to_csv(nomeArquivo, index=False, header=True, quoting=csv.QUOTE_NONE)

######################################################################################################################

#########################################         CONSTRUTORES        ################################################

# Função que controi o vetor de datas de inicio de cada período. #
def controiVetorDatas(cdf: pd.DataFrame, totalPeriodos: int) -> pd.DatetimeIndex:
    dates_vector = pd.date_range(start=min(cdf["date"]), end=max(cdf["date"]), periods=totalPeriodos)
    last_date = dates_vector[-1]
    last_date = last_date + pd.DateOffset(days=1)
    dates_vector = dates_vector[:-1].append(pd.DatetimeIndex([last_date]))
    return dates_vector

# Função que constroi a matriz de clientes por períodos preenchida com zeros. #
def constroiMatrizClientePorPeriodo(cdf: pd.DataFrame, totalPeriodos) -> np.ndarray:
    mat_id_churn = np.zeros( ( totalClientes(cdf), totalPeriodos-1 ) )
    return mat_id_churn

# Função que constroi o Dataframe do churn. #
def constroiChurn(media: pd.Series, cdf: pd.DataFrame) -> pd.DataFrame:
    churn = pd.DataFrame(media)
    churn['id'] = cdf['id_cliente'].unique().astype(str)
    churn = churn.rename(columns={churn.columns[0]: 'churn'})
    return churn[['id', 'churn']]

######################################################################################################################

###############################       CALCULADORES DO DIVIDENDO DA MÉDIA        ######################################

# Função que retorna um vetor com o valor total que irá dividir no calculo da média baseado a recência do cliente. #
def calculaRecenciaCliente(matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex) -> list:
    media_por_cliente = []
    for j in range(len(matIdChurn)):
        comeca_compra = 0
        media_por_cliente.append(0)
        for i in range(len(datesVector)-1):
            if comeca_compra:
                media_por_cliente[j] += 1
            elif matIdChurn[j][i] == 1:
                comeca_compra = 1
                media_por_cliente[j] += 1
    return media_por_cliente

# Função que calcula o valor que divide na média ponderada linear. #
def calculaLinear(totalPeriodos: int) -> float:
    # PA
    # Sn = n( a1 + an ) / 2
    return (totalPeriodos - 1) * (totalPeriodos) / 2.0

# Função que calcula o valor que divide na média ponderada exponencial de qualquer base. #
def calculaExponencial(totalPeriodos: int, base: float) -> float:
    # PG
    # Sn = a1( q**n - 1 ) / (q-1)
    return (base)*(base**(totalPeriodos-1) - 1) / (base - 1)

######################################################################################################################

##################################         PREENCHEDORES DE MATRIZ        ############################################

# Função que preenche a matriz com 1 nas datas de compras que correspondem a determinado período. #
def preencheMatrizUm(matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex, cdf: pd.DataFrame):
    for i in range(len(datesVector)-1):
        for j in range(len(cdf['id_cliente'])):
            if ((datesVector[i] <= cdf.loc[j,'date']) and (cdf.loc[j,'date'] < datesVector[i+1])):
                matIdChurn[cdf.loc[j, 'id_cliente']-1,i] = 1

# Função que preenche a matriz com o valor da coluna nas datas de compras que correspondem a determinado período. #
def preencheMatrizLinear(matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex, cdf: pd.DataFrame):
    for i in range(len(datesVector)-1):
        for j in range(len(cdf['id_cliente'])):
            if ((datesVector[i] <= cdf.loc[j,'date']) and (cdf.loc[j,'date'] < datesVector[i+1])):
                matIdChurn[cdf.loc[j, 'id_cliente']-1,i] = i+1

# Função que preenche a matriz com uma base nas datas de compras que correspondem a determinado período. #
def preencheMatrizExponencial(matIdChurn: np.ndarray, datesVector: pd.DatetimeIndex, cdf: pd.DataFrame, base: float):
    for i in range(len(datesVector)-1):
        for j in range(len(cdf['id_cliente'])):
            if ((datesVector[i] <= cdf.loc[j,'date']) and (cdf.loc[j,'date'] < datesVector[i+1])):
                matIdChurn[cdf.loc[j, 'id_cliente']-1,i] = base**(i+1)

######################################################################################################################

################################         CALCULADORES DE CHURN        ################################################

# Função eu calcula o valor do Churn Binário de cada cliente e salva em um novo Dataframe. #
def calculaChurnBinario(dfIdChurn: pd.DataFrame, cdf: pd.DataFrame) -> pd.DataFrame:
    media = (1- dfIdChurn.mean(axis=1)).round().astype(int)
    churn = constroiChurn(media, cdf)
    return churn

# Função eu calcula o valor do Churn de Média Simples de cada cliente e salva em um novo Dataframe. #
def calculaChurnSimples(dfIdChurn: pd.DataFrame, cdf: pd.DataFrame) -> pd.DataFrame:
    media = 1 - dfIdChurn.mean(axis=1)
    churn = constroiChurn(media, cdf)
    return churn

# Função eu calcula o valor de qualquer Churn Ponderado de cada cliente e salva em um novo Dataframe. #
def calculaChurnPonderado(dfIdChurn: pd.DataFrame, cdf: pd.DataFrame, valorMedia: float) -> pd.DataFrame:
    media = 1 - (dfIdChurn.sum(axis=1) / valorMedia)
    churn = constroiChurn(media, cdf)
    return churn

######################################################################################################################