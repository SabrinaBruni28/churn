import pandas as pd
from lifelines import CoxPHFitter

# Exemplo de DataFrame com dados fictícios de várias empresas
data = pd.DataFrame({
    'tempo_ate_churn': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
    'status_churn': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    'idade': [25, 35, 45, 30, 40, 28, 34, 52, 22, 38],
    'gasto_mensal': [100, 150, 200, 250, 300, 120, 180, 240, 220, 160],
    'frequencia_uso': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
    'tipo_empresa': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']  # Variável categórica adicional
})

# Pré-processamento dos dados
# Convertendo a variável categórica 'tipo_empresa' para variáveis dummy
data = pd.get_dummies(data, columns=['tipo_empresa'], drop_first=True)

# Ajustando o modelo de Cox
cox_model = CoxPHFitter()
cox_model.fit(data, duration_col='tempo_ate_churn', event_col='status_churn')

# Imprimindo o resumo do modelo
cox_model.print_summary()