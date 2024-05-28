import Churn as c 

e = c.np.e

teste = c.calculaAllChurn( arquivo="CDNOW_master.txt", freq="2M", dataInicial="31/12/1996", dataFinal="30/06/1998" )
print(teste)