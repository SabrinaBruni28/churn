# Cáculo de Churn Recente

import pandas as pd
import numpy as np
import csv
from datetime import datetime

arquivo = "CDNOW_master.txt"
nomes_colunas = ["id_cliente", "char_date", "categoria", "valor"]
cdf = pd.read_csv(arquivo, sep="\s+", names=nomes_colunas)

cdf.drop(['categoria','valor'], axis=1, inplace=True)
cdf['char_date'] = cdf["char_date"].astype(str)
cdf["date"] = pd.to_datetime(cdf["char_date"], format="%Y%m%d")
cdf["char_date"] = cdf["date"].dt.strftime("%Y-%m-%d")

total_periodos = 9
total_customer = len(cdf["id_cliente"].unique())

dates_vector = pd.date_range(start=min(cdf["date"]), end=max(cdf["date"]), periods=total_periodos)
last_date = dates_vector[-1]
last_date = last_date + pd.DateOffset(days=1)
dates_vector = dates_vector[:-1].append(pd.DatetimeIndex([last_date]))

mat_id_churn = np.zeros((total_customer, len(dates_vector) - 1))

for i in range(len(dates_vector)-1):
    for j in range(len(cdf['id_cliente'])):
        if ((dates_vector[i] <= cdf.loc[j,'date']) and (cdf.loc[j,'date'] < dates_vector[i+1])):
            mat_id_churn[cdf.loc[j, 'id_cliente']-1,i] = 1

media_por_cliente = []
for j in range(total_customer):
    comeca_compra = 0
    media_por_cliente.append(0)
    for i in range(len(dates_vector)-1):
        if comeca_compra:
            media_por_cliente[j] += 1
        elif mat_id_churn[j][i] == 1:
            comeca_compra = 1
            media_por_cliente[j] += 1

df_id_churn = pd.DataFrame(mat_id_churn)
df_id_churn.columns = dates_vector[:-1].astype(str)

churn = 1 - (df_id_churn.sum(axis=1) / media_por_cliente)
churn = pd.DataFrame(churn)
churn.insert(0,'id',cdf['id_cliente'].unique().astype(str))
churn.rename(columns={0: 'churn'}, inplace=True)

churn.to_csv('calculoChurnRecente.txt', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)