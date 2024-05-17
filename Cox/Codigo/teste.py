import cox as cox
import pandas as pd

# Exemplo de novos dados de clientes para previsão
novos_clientes = pd.DataFrame({
    'idade': [28, 34, 52, 18, 85, 71, 10, 22, 64],
    'gasto_mensal': [120, 200, 150, 100, 580, 1000, 15, 800, 230]
})

# Calculando a função de sobrevivência para novos clientes
sobrevivencia = cox.cox_model.predict_survival_function(novos_clientes)

# Mostrando a probabilidade de sobrevivência (não churn) em diferentes períodos de tempo
print(sobrevivencia)
