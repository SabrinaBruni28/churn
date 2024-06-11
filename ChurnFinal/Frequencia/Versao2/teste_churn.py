import Churn as c 

e = c.e

teste = c.calculaChurn( arquivo="../CDNOW_master.txt", freq="M", modelo="recente" )
print(teste)