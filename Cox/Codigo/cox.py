import pandas as pd
from lifelines import CoxPHFitter

# Exemplo de DataFrame com dados fictícios
data = pd.DataFrame({
    'tempo_ate_churn': [10, 20, 30, 40, 50],
    'status_churn': [1, 0, 0, 1, 1],
    'idade': [25, 35, 45, 30, 40],
    'gasto_mensal': [100, 150, 200, 250, 300]
})

# Ajustando o modelo de Cox
cox_model = CoxPHFitter()
cox_model.fit(data, duration_col='tempo_ate_churn', event_col='status_churn')

# Imprimindo o resumo do modelo
cox_model.print_summary()
