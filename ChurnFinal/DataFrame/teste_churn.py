import Churn as c 
e = c.np.e

teste = c.calculaChurn("CDNOW_master.txt", periodos=10, modelo="recente")
print(teste)
