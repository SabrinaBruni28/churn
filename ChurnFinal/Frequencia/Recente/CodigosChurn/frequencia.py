import Churn as c
import pandas as pd

def calculaFrequencia( df: pd.DataFrame ) -> pd.Timedelta:
    # Fixar o índice #
    cdf.set_index("id_cliente", inplace=True)

    # Calcular os intervalos entre as datas de transações de cada cliente #
    cdf["intervalo"] = df.groupby(df.index)['date'].diff()

    # Remover os valores nulos que surgem após a diferença #
    cdf.dropna(subset=['intervalo'], inplace=True)

    # Calcular a frequência de cada cliente a partir da média dos intervalos para cada cliente #
    media_intervalos = cdf.groupby(df.index)['intervalo'].mean()

    # Calcula a média das frequências de todos os clientes #
    media_frequencias = media_intervalos.mean()

    return media_frequencias



arquivo = "../../ArquivosTransacoes/bank.csv"

# Leitura do DataFrame de transação #
cdf = c._lerArquivo( arquivo )

# Cálculo da Frequência #
f = calculaFrequencia(cdf)
print("Frequência:", f)
