from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.metrics import confusion_matrix,f1_score

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV

from lifetimes.utils import calibration_and_holdout_data,summary_data_from_transaction_data
from lifetimes import GammaGammaFitter
from lifetimes import ParetoNBDFitter
from lifetimes import BetaGeoFitter

from datetime import timedelta
from datetime import datetime

import Churn.Churn as Churn

import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


arquivoTransactions = "Arquivos/transactions.csv"
arquivoOlist = "Arquivos/dataset_Olist.csv"
arquivoShopping = "Arquivos/datasetShopping.csv"
churnTeste = "churnRecente"


def resultadoML(dadosML, churn):
    coluna = 'ExpectedML' + churn
    print(coluna, mean_squared_error(dadosML['Real Expected'], dadosML[coluna]))

def erroML(dadosML, churn):
    coluna = 'ExpectedML' + churn
    dadosML['ErroML' + churn] = abs(dadosML['Real Expected'] - dadosML[coluna])

def graficoErro(dadosML, texto, x=[], labels=[]):
    # Criando a figura com tamanho ajustável
    fig, ax = plt.subplots(figsize=(6, 6))  # Largura: 12, Altura: 8

    # Gerando o boxplot
    sns.boxplot(data=dadosML[x], orient="v", showfliers=False, ax=ax)

    # Configurando o título e os rótulos dos eixos
    ax.set_title(texto, fontsize=14)
    ax.set_ylabel("Erro", fontsize=12)
    ax.set_xlabel("Abordagem", fontsize=12)

    # Configurando os rótulos do eixo X
    ax.set_xticklabels(labels, rotation=45, fontsize=10)

    # Exibindo o gráfico
    plt.tight_layout()
    plt.show()

def graficoComparacao(dadosML, texto, x=[], labels=[]):
    x_labels = {
        "frequency_cal": "Compras no período de calibração",
        "recency_cal": "Age of customer at last purchase",
        "T_cal": "Age of customer at the end of calibration period",
        "time_since_last_purchase": "Time since user made last purchase",
    }

    # Criando a figura e os eixos com tamanho personalizado
    fig, ax = plt.subplots(figsize=(12, 6))  # Aumenta a largura e a altura do gráfico

    # Agrupando e plotando diretamente no eixo
    dadosML.groupby('frequency_cal')[["Real Expected"] + x].mean().plot(ax=ax)

    # Configurando o título e os labels
    ax.set_title("Compras no período de Observação vs \n" + texto, fontsize=14)
    ax.set_xlabel(x_labels['frequency_cal'], fontsize=12)
    ax.set_ylabel("Média de compras\n no período de Observação", fontsize=12)

    # Ajustando o espaçamento dos ticks
    ax.tick_params(axis='x', rotation=45)  # Rotaciona os valores do eixo X, se necessário
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Reduz a quantidade de ticks no eixo X
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))  # Reduz a quantidade de ticks no eixo Y

    # Ajustando a legenda para ficar fora do gráfico
    ax.legend(['Real'] + labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    # Ajustando o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()

def graficoComparacaoPorTopico(dadosML, texto, x=[], labels=[]):
    x_labels = {
        "frequency_cal": "Compras no período de calibração",
        "recency_cal": "Age of customer at last purchase",
        "T_cal": "Age of customer at the end of calibration period",
        "time_since_last_purchase": "Time since user made last purchase",
    }
    
    for i, coluna in enumerate(x):  # Itera sobre cada métrica em x
        # Criando a figura e os eixos com tamanho personalizado
        fig, ax = plt.subplots(figsize=(12, 6))  # Aumenta a largura e a altura do gráfico

        # Agrupando e plotando diretamente no eixo
        dadosML.groupby('frequency_cal')[["Real Expected", coluna]].mean().plot(ax=ax)

        # Configurando o título e os labels
        ax.set_title(f"Compras no período de Observação vs \n{texto} ({labels[i]})", fontsize=14)
        ax.set_xlabel(x_labels['frequency_cal'], fontsize=12)
        ax.set_ylabel("Média de compras\n no período de Observação", fontsize=12)

        # Ajustando o espaçamento dos ticks
        ax.tick_params(axis='x', rotation=45)  # Rotaciona os valores do eixo X, se necessário
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Reduz a quantidade de ticks no eixo X
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))  # Reduz a quantidade de ticks no eixo Y

        # Ajustando a legenda para ficar fora do gráfico
        ax.legend(['Real', labels[i]], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

        # Ajustando o layout para evitar sobreposição
        plt.tight_layout()
        plt.show()


def graficoComparacaoSeparada(dadosML, texto, x=[], labels=[]):
    x_labels = {
        "frequency_cal": "Compras no período de calibração",
        "recency_cal": "Age of customer at last purchase",
        "T_cal": "Age of customer at the end of calibration period",
        "time_since_last_purchase": "Time since user made last purchase",
    }

    # Criando subgráficos com base no número de variáveis em x
    num_graficos = len(x)
    fig, axes = plt.subplots(1, num_graficos, figsize=(12 * num_graficos, 6), sharey=True)

    # Garantir que axes seja sempre uma lista (mesmo com um único gráfico)
    if num_graficos == 1:
        axes = [axes]

    for i, coluna in enumerate(x):
        ax = axes[i]
        
        # Plotando os dados no subgráfico correspondente
        dadosML.groupby('frequency_cal')[["Real Expected", coluna]].mean().plot(ax=ax)

        # Configuração do título e labels de cada subgráfico
        ax.set_title(f"{texto} ({labels[i]})", fontsize=14)
        ax.set_xlabel(x_labels.get('frequency_cal', 'Frequency Cal'), fontsize=12)
        if i == 0:  # Apenas no primeiro gráfico
            ax.set_ylabel("Média de compras\n no período de Observação", fontsize=12)

        # Ajustando o espaçamento dos ticks
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))

        # Ajustando a legenda
        ax.legend(['Real', labels[i]], loc='upper left', fontsize=10)

    # Ajustando o layout
    plt.tight_layout()
    plt.show()


#Baseado no dataframe passado, e na frequência, divide os dados de acordo com a porcentagem passada.
#Retorna qual é a data correspondente à aquela divisão passada no split, e a ultima data da separação
def getPeriodos(df, #Dataframe do Pandas 
                colunaData, #Nome da coluna onde estão as datas
                frequencia, #Frequência em que será observado, Ex: "W" - Weeks
                split = 0.8 #Porcentagem da divisão dos dados para separar em treino e calibração
               ):
    primeiraData = df[colunaData].sort_values().values[0]
    ultimaData = df[colunaData].sort_values().values[-1]
    rangeDatas = pd.date_range(start=primeiraData,end=ultimaData,freq=frequencia)
    indiceCorte = round(len(rangeDatas) * split)
    return rangeDatas[indiceCorte],ultimaData

#Processa um dataset de acordo com o padrão RFMT
# O retorno desta função consiste em:
#Se teste for true : Retorna o dataset considerando a data de divisão para dividir o período de 'calibração' e 'holdout'.
#Se for falso: Retorna todo o dataset processado pelo padrão RFMT.
def processarRFM(arquivo #Nome do arquivo
                 ,colunaID #Nome da coluna onde encontra-se os identificadores
                 ,colunaData  #Nome da coluna onde encontra-se as datas
                 ,colunaValor  #Nome da coluna onde encontra-se os valores monetários
                 ,frequencia = 'W' #Frequência em que será observado, Ex: "W" - Weeks
                 ,calibrationEnd = None #Caso queira passar a data do fim do período de calibração
                 ,ObservationEnd = None #Caso queira passar a data do fim do período de Obsersvação
                 ,split = 0.8 # Porcentagem da divisão dos dados para separar em Obsersvação e calibração
                 , teste = True #Verdadeiro caso queira separar os dados em Obsersvação e calibração
                ):
    df = pd.read_csv(arquivo)
    df = df.dropna()
    
    if calibrationEnd == None:
        calibrationEnd,ObservationEnd = getPeriodos(df,colunaData,frequencia,split)
        
    if teste == False:
        rfm_cal_holdout = summary_data_from_transaction_data(transactions=df,
                                                  customer_id_col=colunaID, 
                                                   datetime_col=colunaData,
                                                   monetary_value_col = colunaValor,
                                                   freq=frequencia)
        dataFinal = ObservationEnd
    else:
        rfm_cal_holdout = calibration_and_holdout_data(transactions=df,
                                                  customer_id_col=colunaID, 
                                                   datetime_col=colunaData,
                                                   monetary_value_col = colunaValor,
                                                   freq=frequencia,
                                                   calibration_period_end=calibrationEnd,
                                                   observation_period_end=ObservationEnd)
        dataFinal = calibrationEnd.strftime('%Y-%m-%d')

    dataInicial = pd.to_datetime(df[colunaData]).dt.date.min().strftime('%Y-%m-%d')
    churn = Churn.calculaAllChurn( arquivo, colunaID, colunaData, dataInicial, dataFinal, frequencia )
    rfm_cal_holdout = pd.concat([rfm_cal_holdout, churn], axis=1)

    return rfm_cal_holdout.fillna(0.0)

#Função utilizada para criar os modelos de regressão
#Retorna o MSE e o modelo
def createModelRegressor(RegressorModel #Modelo que será treinado (Tem de ter a função fit e predict implementadas)
                         , X_train #Dados que serão usados para o treino
                         , X_test #Dados que serão usados para o teste
                         , Y_train #Targets dos dados de treino
                         , Y_test #Targets dos dados de testes
                        ): 
    regressor = RegressorModel
    regressor.fit(X_train, Y_train)
    pred = regressor.predict(X_test)
    mse = mean_squared_error(Y_test, pred) #Utilizando o MSE, caso queira outra métrica, trocar nesta parte!
    return mse, regressor #Retorna o MSE e o Regressor

#Função feita para escolher o melhor modelo de acordo com a situação
#Retorna o melho modelo para a situação
def escolheModelo(dfRFM,  # Dataframe já processado pelo RFM
                  target = 'monetary_value_holdout',  # Nome da coluna de target, sendo a coluna de valor monetário prevista ou frequência
                  tunning = False,  # Caso queira fazer o Tunning de hyperparâmetros, deixar como true
                  churn = None):  # Se churn for None, ele não será utilizado nas colunas de treino
    # Colunas utilizadas para treino
    Xcol = ['frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout']
    if churn is not None:
        Xcol.append(churn)  # Adiciona a coluna churn se não for None
        
    #Colunas utilizadas para target
    Ycol = [target]

    X = dfRFM[Xcol]
    Y = dfRFM[Ycol]

    X_train, X_test, Y_train, Y_test = train_test_split(X.values, np.ravel(Y.values), random_state=42)
    
    if tunning == False:
        lasso = LassoCV()
        Enet = ElasticNet()
        rf = RandomForestRegressor()
        krr = KernelRidge()
        GBoost = GradientBoostingRegressor()
        HGBoost = HistGradientBoostingRegressor()
        model_xgb = xgb.XGBRegressor()
        model_lgb = lgb.LGBMRegressor(objective='regression')
    
    else:
        #lasso = LassoCV()
        grid = {'n_alphas' : [100,200,500,100],'max_iter' : [1000,1500,2000], 'random_state' : [42]}
        lasso = GridSearchCV(estimator=LassoCV(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")

        grid = {"max_iter": [1000,1500,2000],"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],"l1_ratio": np.arange(0.0, 1.0, 0.1), 'random_state' : [42]}
        Enet = GridSearchCV(estimator=ElasticNet(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")
        #Enet = ElasticNet()

        grid = {'bootstrap': [True, False],'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10], 'n_estimators': [200, 800, 1000],'random_state' : [42]}    
        rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")
        #rf = RandomForestRegressor()

        grid = {"alpha": [0.001, 0.01, 0.1, 1,], "coef0" : [0.01,0.1,1,10,100] ,'degree_': [1,3,5,10],'random_state':[42]}
        krr = GridSearchCV(estimator=KernelRidge(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")
        #krr = KernelRidge()

        grid = {'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1],'max_depth':[1,2,4],'subsample':[.5,.75,1],'random_state':[42]}
        GBoost = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")
        #GBoost = GradientBoostingRegressor()
        grid = {'learning_rate':[.001,0.01,.1],'max_depth':[1,2,4,None],'max_leaf_nodes' : [31,None],'random_state':[42]}

        HGBoost = GridSearchCV(estimator=HistGradientBoostingRegressor(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")
        #HGBoost = HistGradientBoostingRegressor()

        grid = { 'max_depth': [3,6,10],'learning_rate': [0.01, 0.05, 0.1],'n_estimators': [100, 500, 1000],'colsample_bytree': [0.3, 0.7],'random_state':[42]}
        #model_xgb = xgb.XGBRegressor()
        model_xgb = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")
        #model_lgb = lgb.LGBMRegressor(objective='regression')
        model_lgb =GridSearchCV(estimator=lgb.LGBMRegressor(), param_grid=grid, n_jobs=-1, scoring="neg_mean_squared_error")
    
    
    models = [lasso, Enet, rf, GBoost, HGBoost, model_xgb, model_lgb]#,krr]

    bestModel, bestScore = None, None
    for i in models:
        score = createModelRegressor(i, X_train, X_test, Y_train, Y_test)
        if bestScore == None or bestScore > score[0]:
            bestScore, bestModel = score
        if tunning:
            print(type(i.best_estimator_).__name__, " mse: {:.4f} \n".format(score[0]))
        else:
            print(type(i).__name__, " mse: {:.4f} \n".format(score[0]))

    if tunning :
        return bestModel.best_estimator_
    else:
        return bestModel
    #prediction = bestModel.predict(X_test[Xcol])
    #X_test['ExpectedML'] = prediction
    #X_test['Real Expected'] = Y_test
    #return X_test


# Função para criar o modelo BG/NBD
def criarModeloBGF(dadosRFM #Dataset já processado pelo RFM
                   ,teste = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
                   ,penalizer = 0.1# Coeficiente de penalização usado pelo modelo
                  ):
    # instantiation of BG-NBD model
    bgf = BetaGeoFitter(penalizer_coef=penalizer)

    # fitting of BG-NBD model
    if teste:
        bgf.fit(frequency=dadosRFM['frequency_cal'],
                recency=dadosRFM['recency_cal'],
                T=dadosRFM['T_cal'])
    else:
        bgf.fit(frequency=dadosRFM['frequency'],
                recency=dadosRFM['recency'],
                T=dadosRFM['T'])

    return bgf

#Dado um período, retorna o número de transações esperadas até lá
def comprasEsperadas(model #Modelo BG/NBD ou de Pareto esperado para realizar a predição
                     ,rfm #Dataset já processado pelo RFM
                     ,numPeriodos = 180 #Numero de períodos em dia para que deseja efetuar a predição
                     , teste = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
                    ):
    if teste:
        return model.conditional_expected_number_of_purchases_up_to_time(numPeriodos, rfm['frequency_cal'].values, rfm['recency_cal'].values, rfm['T_cal'].values)
    return model.conditional_expected_number_of_purchases_up_to_time(numPeriodos, rfm['frequency'].values, rfm['recency'].values, rfm['T'].values)


# Função para criar o modelo de Pareto/NBD
def criarModeloPareto(dadosRFM #Dataset já processado pelo RFM
                      ,teste = True  #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
                      ,penalizer = 0.1# Coeficiente de penalização usado pelo modelo
):
    # instantiation of Pareto model
    pareto = ParetoNBDFitter(penalizer_coef=penalizer)

    # fitting of the model
    if teste:
        pareto.fit(frequency=dadosRFM['frequency_cal'],
                recency=dadosRFM['recency_cal'],
                T=dadosRFM['T_cal'])
    else:
        pareto.fit(frequency=dadosRFM['frequency'],
                recency=dadosRFM['recency'],
                T=dadosRFM['T'])

    return pareto


#Função feita para executar os testes dos 3 modelos em predizer a frequência de compras. 
#Retorna um dataframe contendo a predição de cada modelo
def testarModelosPredicaoFrequencia(splits = [i/20 for i in range(11,17)], # Quais divisões serão testadas, um vetor contendo os decimais
                                    file = arquivoTransactions, #Nome do arquivo
                                    cID = "customer_id", #Nome da coluna contendo o ID dos clientes
                                    cDate = "date",#Nome da coluna onde contém a data das compras
                                    cMonetary = 'amount', #Nome da coluna onde contém a média dos valores monetários
                                    churn = [] #Qual cálculo de churn será usado
                                   ):
    df2 = processarRFM(file, cID, cDate, cMonetary,split = splits[0])
    for i in splits[1:-1]:
        df1 = processarRFM(file, cID, cDate, cMonetary,split = i)
        df2 = pd.concat([df2, df1], ignore_index=True)

    dfValidacao = processarRFM(file, cID, cDate, cMonetary,split = splits[-1])
    start = time.time()

    print("Criando modelo ML")
    start = time.time()

    modelML = escolheModelo(df2, target = 'frequency_holdout')
    end = time.time()
    print("Tempo ML: ",timedelta(seconds = end - start))
    
    dfValidacao['Real Expected'] = dfValidacao['frequency_holdout']
    dfValidacao['ExpectedML'] = modelML.predict(dfValidacao[['frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal','duration_holdout']])
    dfValidacao[dfValidacao['ExpectedML'] < 0] = 0
    
    for c in churn:
        coluna = 'ExpectedML' + c
        modelML = escolheModelo(df2, target = 'frequency_holdout', churn = c)
        dfValidacao[coluna] = modelML.predict(dfValidacao[['frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal','duration_holdout', c]])
        dfValidacao[dfValidacao[coluna] < 0] = 0

    return dfValidacao

#Função para criar o modelo que prevê o valor monetário médio de cada cliente, usando o modelo Gamma Gamma.
#Retorna o dataset com a coluna 'ExpectedGammaGamma'
def preverValorGGF(rfm, #Dataset já processado pelo RFM
                   coefPenalizacao = 0.01, #Coeficiente de penalização utilizada pelo Gamma Gamma
                   teste = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
                  ):
    monetary = "monetary_value"
    frequency = "frequency"
    if teste:
        monetary = "monetary_value_cal"
        frequency = "frequency_cal"
    ggf = GammaGammaFitter(coefPenalizacao)
    rfm = rfm[rfm[monetary] > 0]
    ggf.fit(rfm[frequency],rfm[monetary])
    print(ggf)
    rfm['ExpectedGammaGamma'] = ggf.conditional_expected_average_profit(rfm[frequency], rfm[monetary])
    return rfm

#Função feita para aplicar todos os passos para prever o valor monetário esperado de um cliente, utilizando tanto o Gamma-Gamma quanto modelos de aprendizado de máquina
#Retorna um dataset contendo as predições esperadas tanto pro Gamma-Gamma quanto para os modelos de aprendizado de máquina
def testarModelosPredicaoMonetario(splits = [i/20 for i in range(11,17)],# Quais divisões serão testadas, um vetor contendo os decimais
                                   file = arquivoTransactions, #Nome do arquivo onde os dados se encontram
                                   cID = "customer_id", #Nome da coluna de ID do cliente
                                   cDate = "date", #Nome da coluna de data
                                   cMonetary = 'amount', #Nome da coluna onde estão os valores monetários
                                   churn = [] #Qual cálculo de churn será usado
                                  ):
    df2 = processarRFM(file, cID, cDate, cMonetary,split = splits[0])
    for i in splits[1:-1]:
        f = open(file,'r')
        df1 = processarRFM(file, cID, cDate, cMonetary,split = i)
        df2 = pd.concat([df2, df1], ignore_index=True)

    df2['target'] = ((df2['monetary_value_cal'] * df2['frequency_cal']) + (df2['monetary_value_holdout'] * df2['frequency_holdout'])) / (df2['frequency_cal'] + df2['frequency_holdout'])
    df2['target'] = df2['target'].fillna(0)
    
    dfValidacao = processarRFM(file, cID, cDate, cMonetary,split = splits[-1])
    dfValidacao['target'] = ((dfValidacao['monetary_value_cal'] * dfValidacao['frequency_cal']) + (dfValidacao['monetary_value_holdout'] * dfValidacao['frequency_holdout'])) / (dfValidacao['frequency_cal'] + dfValidacao['frequency_holdout'])
    dfValidacao['target'] = dfValidacao['target'].fillna(0)

    modelML = escolheModelo(df2,target = 'target')
    dfValidacao['Real Expected'] = dfValidacao['target']
    dfValidacao['ExpectedML'] = modelML.predict(dfValidacao[['frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal','duration_holdout']])
    dfValidacao[dfValidacao['ExpectedML'] < 0]= 0
    
    for c in churn:
        coluna = 'ExpectedML' + c
        modelML = escolheModelo(df2,target = 'target', churn = c)
        dfValidacao[coluna] = modelML.predict(dfValidacao[['frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal','duration_holdout', c]])
        dfValidacao[dfValidacao[coluna] < 0]= 0
    
    dfValidacao = preverValorGGF(dfValidacao)
    return dfValidacao

def evaluate_clv(actual, predicted, bins, title = ""):
    print(f"Average absolute error: {mean_absolute_error(actual, predicted)}")
    #Evaluate numeric
    ypbot = np.percentile(actual, 1)
    yptop = np.percentile(actual, 99)
    ypad = 0.2*(yptop - ypbot)
    ymin = ypbot - ypad
    ymax = yptop + ypad
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(x=predicted, y=actual)
    plt.xlabel('Previsto')
    plt.ylabel('Atual')
    plt.title('Previsto vs Atual '+title)
    ax.set_xlim([ymin, ymax])
    ax.set_ylim([ymin, ymax])
    plt.show()
    
    #Evaluate Bins
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    est.fit(np.array(actual).reshape(-1, 1))
    actual_bin = est.transform(np.array(actual).reshape(-1, 1)).ravel()
    predicted_bin = est.transform(np.array(predicted).reshape(-1, 1)).ravel()
    
    cm = confusion_matrix(actual_bin, predicted_bin, normalize='true')
    df_cm = pd.DataFrame(cm, index = range(1, bins+1),
                      columns = range(1, bins+1))
    plt.figure(figsize = (20,10))
    sns.heatmap(df_cm, annot=True)

    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()
    print(f'F1 score: {f1_score(actual_bin, predicted_bin, average="macro")}')
    print('Samples in each bin: \n')
    print(pd.Series(actual_bin).value_counts())

def expectedLTV(dadosML, Modelo, x=[]):
    # Calcula LTV ML
    dadosML['LTV ML'] = calculaLTV(dadosML, Modelo, 'ExpectedML', teste=True, ML=True, LIFE=1)['LTV']
    
    # Calcula LTV Real
    dadosML['LTV Real'] = (dadosML['frequency_holdout'] * dadosML['target']) / 1.06
    
    # Calcula Erro para LTV ML
    dadosML['ErroLTV ML'] = abs(dadosML['LTV Real'] - dadosML['LTV ML'])
    
    # Define valores negativos como 0 para as colunas 'LTV ML' e 'ErroLTV ML'
    dadosML.loc[dadosML['LTV ML'] < 0, 'LTV ML'] = 0
    
    # Loop para calcular LTV para os elementos de 'x'
    for i in x:
        coluna = 'LTV' + i
        dadosML[coluna] = calculaLTV(dadosML, Modelo, i, teste=True, ML=True, LIFE=1)['LTV']
        
        # Define valores negativos como 0 para a nova coluna
        dadosML.loc[dadosML[coluna] < 0, coluna] = 0
        
        # Calcula o erro para a nova coluna
        dadosML['ErroLTV' + i] = abs(dadosML['LTV Real'] - dadosML[coluna])

    return dadosML


#Função feita para adaptar o cálculo do LTV tanto para o contexto de aprendizado de máquina quanto para modelos probabilísticos
#Retorna um dataframe contendo as colunas do LTV esperado
def calculaLTV(df # Dataframe que já tenha sido processado pelo RFM
               ,modelo # Modelo utilizado para prever as compras esperadas
               ,monetaryCol #Nome da coluna onde possui o valor médio esperado por transação
               , DISCOUNT_a = 0.06 #Taxa de desconto anual
               ,LIFE = 12 #Meses que deseja calcular o lifetime
               ,freq = "D" #Frequência que os dados estão
               ,teste = True #Caso seja para prever de acordo com o período de observação ou não
               , ML = False #Caso o modelo passado seja de aprendizado de máquina, necessista ser true
              ):
    #DISCOUNT_a annual discount rate
    #LIFE lifetime expected for the customers in months
    #Freq: Date unit of the frequency
    coluna = 'LTV'
    dfRetorno = df.copy()
    if teste == True:
        dfRetorno = dfRetorno.rename(columns = {'frequency_cal':'frequency', 'recency_cal':'recency', 'T_cal':'T', "monetary_value_cal" : 'monetary_value'})
    dfRetorno[coluna] = 0 
    factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[freq] 
    for i in np.arange(1, LIFE + 1) * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        if ML:
            dfRetorno['X1'] = i
            dfRetorno['X2'] = i - factor
            expectedPurchase = modelo.predict(dfRetorno[['frequency', 'recency', 'T', 'monetary_value','X1']].values
                                             ) - modelo.predict(dfRetorno[['frequency', 'recency', 'T', 'monetary_value','X2']].values)
            dfRetorno['Expected'+str(i)] = expectedPurchase

        else:
            expectedPurchase = ltv.comprasEsperadas(modelo, df,i,teste = teste) - ltv.comprasEsperadas(modelo, df,i - factor,teste = teste)
            dfRetorno['Expected'+str(i)] = expectedPurchase
        # sum up the CLV estimates of all of the periods and apply discounted cash flow
        dfRetorno[coluna] = dfRetorno[coluna] + ((df[monetaryCol] * expectedPurchase) / (1 + DISCOUNT_a) ** (i / factor))
    return dfRetorno

#Função para criar os modelos de aprendizado de máquina separados dos testes                                

def prepararML(splits = [i/20 for i in range(11,17)], # Quais divisões serão testadas, um vetor contendo os decimais
               file = arquivoTransactions, #Nome do arquivo onde os dados se encontram
               cID = "customer_id", #Nome da coluna de ID do cliente
               cDate = "date", #Nome da coluna de data
               cMonetary = 'amount',  #Nome da coluna onde estão os valores monetários
               churn = []
              ):

    f = pd.read_csv(file)
    
    df2 = processarRFM(file, cID, cDate, cMonetary,split = splits[0])
    for i in splits[1:-1]:
        df1 = processarRFM(file, cID, cDate, cMonetary,split = i)
        df2 = pd.concat([df2, df1])
        df2 = df2.fillna(0.0)

    dfValidacao = processarRFM(file, cID, cDate, cMonetary,split = splits[-1])
    
    modelML = escolheModelo(df2,target = 'frequency_holdout')
    return modelML