import Churn as c 

e = c.e

teste = c.calculaAllChurn( arquivo="../../ArquivosTransacoes/trans_clean.csv", dataFinal= '1998-08-31', freq="2W" )
print(teste)