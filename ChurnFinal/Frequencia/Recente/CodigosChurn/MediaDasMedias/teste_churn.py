import Churn as c 

e = c.e

teste = c.calculaAllChurn( arquivo="../../../ArquivosTransacoes/trans_clean.csv", dataFinal= '1998-10-22', freq="7D" )
print(teste)