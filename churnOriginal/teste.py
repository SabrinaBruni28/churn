import pandas as pd
import numpy as np
from datetime import datetime

cdf = pd.read_table("CDNOW_master.txt", header=None)
cdf = pd.DataFrame(np.column_stack([cdf.iloc[:, 0:2], np.zeros(len(cdf))]), columns=["id", "char_date", "date"])
cdf["char_date"] = cdf["char_date"].astype(str)
cdf["date"] = pd.to_datetime(cdf["char_date"], format="%Y%m%d")
cdf["char_date"] = cdf["date"].dt.strftime("%Y%d%m")
totalChurnZones = 9
dates_vector = pd.date_range(start=min(cdf["date"]), end=max(cdf["date"]), periods=totalChurnZones)
dates_vector = np.append(dates_vector[:-1], dates_vector[-1] + pd.DateOffset(days=1))
totalCustomers = len(cdf["id"].unique())
mat_id_churn = np.zeros((totalCustomers, len(dates_vector) - 1))
df_id_churn = pd.DataFrame(mat_id_churn)
df_id_churn.columns = dates_vector[:-1].strftime("%Y%d%m")
i = 0
while dates_vector[i] < dates_vector[-1]:
    for j in range(len(cdf)):
        if (dates_vector[i] <= cdf.loc[j, "date"]) and (cdf.loc[j, "date"] < dates_vector[i + 1]):
            mat_id_churn[cdf.loc[j, "id"], i] = 1
    i += 1
df_id_churn = pd.DataFrame(mat_id_churn)
v = df_id_churn.apply(lambda x: round(x.sum() / (totalChurnZones - 1)), axis=1)
v.to_csv("CDNOW_master_churn.txt", index=True)