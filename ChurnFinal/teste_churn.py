import Churn as c 
e = c.np.e

teste = c.calculaChurn("CDNOW_master.txt", periodos=9, modelo="exponencial", base=e)
print(teste)