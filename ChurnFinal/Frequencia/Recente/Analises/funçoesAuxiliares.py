import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import precision_recall_fscore_support
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# Função para contar zeros e uns em uma coluna
def contar_zeros_uns(coluna):
    contagem = coluna.value_counts()
    zeros = contagem.get(0, 0)
    uns = contagem.get(1, 0)
    return pd.Series({'zeros': zeros, 'uns': uns})

def quebra_linha(texto, n=20):
    resultado = ''
    while len(texto) > n:
        # Encontra a posição do primeiro espaço após n caracteres
        espaço = texto[:n].rfind(' ')
        if espaço == -1:  # Se não encontrar espaço, quebra em n
            espaço = n
        # Adiciona a parte do texto até o espaço encontrado e adiciona uma quebra de linha
        resultado += texto[:espaço] + '\n'
        # Remove a parte processada do texto original
        texto = texto[espaço:].strip()
    resultado += texto  # Adiciona o restante do texto
    return resultado

def totalPorCategoria(df, coluna1, coluna2, tamanhos: tuple = (12, 10), limiteinfy: float = None, limitesupy: float = None, bins: float = 10, largura_barras: float = 0.8, espaçamento: float = 0.2, divisaoGrafico: int = 1, title: str = ""):
    # Verificar se divisaoGrafico é válido
    if divisaoGrafico <= 0:
        raise ValueError("O número de divisões do gráfico deve ser maior que zero.")
    
    # Calcular a quantidade de estudantes por curso e sexo
    total = df.groupby([coluna1, coluna2]).size().reset_index(name='Quantidade')
    
    # Dividir os dados em várias partes
    n_categorias = len(total[coluna1].unique())
    parte_tamanho = n_categorias // divisaoGrafico
    totais = []
    
    for i in range(divisaoGrafico):
        inicio = parte_tamanho * i
        fim = parte_tamanho * (i + 1) if i < divisaoGrafico - 1 else n_categorias
        totais.append(total[total[coluna1].isin(total[coluna1].unique()[inicio:fim])])
    
    # Criar subplots
    fig, axes = plt.subplots(divisaoGrafico, 1, figsize=tamanhos, sharey=True)
    
    if divisaoGrafico == 1:
        axes = [axes]  # Convertendo para lista para facilitar o loop

    for i in range(divisaoGrafico):
        sns.barplot(x=coluna1, y='Quantidade', hue=coluna2, data=totais[i], 
                    ax=axes[i], width=largura_barras, dodge=espaçamento)
        axes[i].set_title('{}'.format(title))
        axes[i].tick_params(axis='x', rotation=45)
        # Ajusta os limites do eixo Y
        if limiteinfy is not None and limitesupy is not None:
            axes[i].set_ylim(limiteinfy, limitesupy)
        
        # Ajusta os ticks do eixo Y
        if limiteinfy is not None and limitesupy is not None:
            axes[i].set_yticks(np.arange(limiteinfy, limitesupy + bins, bins))
        # Adicionar as quantidades em cima de cada barra
        for p in axes[i].patches:
            height = int(p.get_height())
            if height != 0:
                axes[i].annotate(f'{height}', 
                                 xy=(p.get_x() + p.get_width() / 2, height),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom')
        axes[i].legend(title=coluna2, bbox_to_anchor=(1.05, 1), loc='upper left')
    # Ajustar a visualização dos gráficos
    plt.tight_layout()
    plt.show()

def graficoEvasaoCursoSexo(df: pd.DataFrame, tamanhos: tuple = (15, 10), limiteinfy: float = None, limitesupy: float = None, bins: float = 10, divisaoGrafico=1):
    # Verificar se divisaoGrafico é válido
    if divisaoGrafico <= 0:
        raise ValueError("O número de divisões do gráfico deve ser maior que zero.")
    
    # Calcular o total de estudantes por curso e sexo
    total_por_grupo = df.groupby(['Curso', 'Sexo']).size().reset_index(name='total')
    
    # Calcular o número de evasões por curso e sexo
    evasoes_por_grupo = df.groupby(['Curso', 'Sexo'])['Evasao_Categoria'].sum().reset_index()
    
    # Juntar os dois DataFrames
    df_porcentagem = pd.merge(evasoes_por_grupo, total_por_grupo, on=['Curso', 'Sexo'])
    
    # Calcular a porcentagem de evasão e de não evasão
    df_porcentagem['porcentagemEvasao'] = (df_porcentagem['Evasao_Categoria'] / df_porcentagem['total']) * 100
    df_porcentagem['porcentagemNaoEvasao'] = 100 - df_porcentagem['porcentagemEvasao']
    
    # Criar DataFrames separados para evasão e não evasão
    evasao_df = df_porcentagem[['Curso', 'Sexo', 'porcentagemEvasao']].copy()
    evasao_df['Categoria'] = 'Evasão'
    evasao_df.rename(columns={'porcentagemEvasao': 'Porcentagem'}, inplace=True)
    
    nao_evasao_df = df_porcentagem[['Curso', 'Sexo', 'porcentagemNaoEvasao']].copy()
    nao_evasao_df['Categoria'] = 'Não Evasão'
    nao_evasao_df.rename(columns={'porcentagemNaoEvasao': 'Porcentagem'}, inplace=True)
    
    # Combinar os dois DataFrames
    df_final = pd.concat([evasao_df, nao_evasao_df])
    
    # Criar uma nova coluna que combina Sexo e Categoria
    df_final['Sexo_Categoria'] = df_final['Sexo'] + ' - ' + df_final['Categoria']
    
    # Definir a ordem desejada para as barras
    ordem = {
        'F - Evasão': 1,
        'F - Não Evasão': 2,
        'M - Evasão': 3,
        'M - Não Evasão': 4
    }
    
    # Adicionar a coluna de ordenação
    df_final['Ordenacao'] = df_final['Sexo_Categoria'].map(ordem)
    
    # Ordenar o DataFrame com base na coluna de ordenação e no curso
    df_final = df_final.sort_values(by=['Ordenacao', 'Curso'])
    
    # Dividir o DataFrame em partes
    n_cursos = len(df_final['Curso'].unique())
    parte_tamanho = n_cursos // divisaoGrafico
    partes = []
    
    for i in range(divisaoGrafico):
        inicio = parte_tamanho * i
        fim = parte_tamanho * (i + 1) if i < divisaoGrafico - 1 else n_cursos
        partes.append(df_final[df_final['Curso'].isin(df_final['Curso'].unique()[inicio:fim])])
    
    # Criar subplots
    fig, axes = plt.subplots(divisaoGrafico, 1, figsize=tamanhos, sharey=True, constrained_layout=True)
    
    if divisaoGrafico == 1:
        axes = [axes]  # Converter para lista para facilitar o loop
    
    # Definir cores específicas para cada categoria
    cores = {
        'F - Evasão': 'lightcoral',
        'F - Não Evasão': 'lightblue',
        'M - Evasão': 'red',
        'M - Não Evasão': 'blue'
    }
    
    for i in range(divisaoGrafico):
        sns.barplot(x='Curso', y='Porcentagem', hue='Sexo_Categoria', data=partes[i], dodge=0.5, palette=cores, ax=axes[i])
        axes[i].set_title('Parte {}: Porcentagem de Evasão e Não Evasão por Curso e Sexo'.format(i + 1))
        axes[i].tick_params(axis='x', rotation=45)
        
        # Ajusta os limites do eixo Y
        if limiteinfy is not None and limitesupy is not None:
            axes[i].set_ylim(limiteinfy, limitesupy)
        
        # Ajusta os ticks do eixo Y
        if limiteinfy is not None and limitesupy is not None:
            axes[i].set_yticks(np.arange(limiteinfy, limitesupy + bins, bins))
        
        # Adicionar as porcentagens em cima de cada barra
        for p in axes[i].patches:
            height = p.get_height()
            if height > 0:
                offset = 0
                if height > 50:  # Ajustar o valor conforme necessário
                    offset = 5
                
                axes[i].annotate(f'{height:.1f}%',
                                 xy=(p.get_x() + p.get_width() / 2, height),
                                 xytext=(0, offset),  # Deslocamento para cima
                                 textcoords='offset points',
                                 ha='center', va='bottom',
                                 fontsize=10)  # Aumentar o tamanho da fonte dos valores
        
        axes[i].legend(title='Sexo_Categoria', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.show()

def agrupar(df: pd.DataFrame, categoriaAgrupar: str, coluna:str) -> list:
    # Identificar categorias únicas na coluna 
    categorias = df[categoriaAgrupar].unique()
    
    # Lista para armazenar os grupos
    grupos = []
    
    # Iterar sobre cada categoria e adicionar o DataFrame correspondente à lista
    for categoria in categorias:
        grupo = df[df[categoriaAgrupar] == categoria][coluna]
        grupos.append(grupo)
    return grupos

def validacaoR2MatrizConfusao(real, predict: list, titulos: list, tamanhos: tuple = (10,5), cmap='viridis'):
    n = len(predict)
    if n != len(titulos):
        raise ValueError("As listas devem ter o mesmo tamanho.")
    r2 = []
    CM = []
    for i in range(n):
        r2.append(r2_score(real, predict[i]))
        CM.append(confusion_matrix(real, predict[i]))
    print("R² =", r2)
    
    matrizConfusao(matrizes = CM, titulos = titulos, tamanhos = tamanhos, cmap = cmap)
    return r2, CM


def matrizConfusao(matrizes: list, titulos: list, tamanhos: tuple = (10, 5), cmap='viridis'):
    n = len(matrizes)
    
    # Verificar se o número de matrizes e títulos são iguais
    if n != len(titulos):
        raise ValueError("O número de matrizes deve ser igual ao número de títulos.")
    
    # Criar uma figura com subplots
    fig, ax = plt.subplots(1, n, figsize=tamanhos)
    
    # Se houver apenas uma matriz, ax não será uma lista, então tratamos esse caso separadamente
    if n == 1:
        ax = [ax]  # Transforme ax em uma lista para tratar de forma consistente

    for i in range(n):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrizes[i])
        disp.plot(ax=ax[i], colorbar=False, cmap=cmap)  # Aplicar o cmap aqui
        ax[i].set_title(titulos[i])

    plt.tight_layout()
    plt.show()

def regressaoLinear(X, Y):
    lm = LinearRegression()
    lm.fit(X.values.reshape(-1,1), Y)
    
    print('Coeficiente estimado: ', lm.coef_)
    print('R² (score): ', lm.score(X.values.reshape(-1,1), Y))
    
    # Mostrar os coeficientes da regressão.
    print('Intercept %.3f ' % lm.intercept_)

def testeANOVA(df, grupos: list, variavelAnalise: str, variavelGrupos: str):
    # Realizar o teste ANOVA
    f_statistic, p_value = stats.f_oneway(*grupos)
    
    print(f"F-statistic: {f_statistic}")
    print(f"P-value: {p_value}")

    # Realizar ANOVA usando statsmodels
    modelo = ols('{} ~ C({})'.format(variavelAnalise, variavelGrupos), data=df).fit()
    anova_table = sm.stats.anova_lm(modelo, typ=2)
    print(anova_table)
    print("\n\n")
    
    # Teste de Tukey
    tukey = pairwise_tukeyhsd(endog=df[variavelAnalise], groups=df[variavelGrupos], alpha=0.05)
    print(tukey)

def matrizCorrelacao(correlation, tamanhos=(5,8)):
    # Configurar o tamanho da figura
    plt.figure(figsize=tamanhos)
    
    # Criar o heatmap
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    
    # Adicionar título
    plt.title('Matriz de Correlação')
    
    # Exibir o gráfico
    plt.show()
    
def treinarModelo(algoritmo, X_train, Y_train, X_test, Y_test):
    """
    Função que treina um modelo e calcula a acurácia com base nos dados de treino e teste.
    Retorna o Y previsto e a acurácia.
    O algoritmo é a instância do modelo desejado.
    """
    # Treina o modelo usando os dados de treino
    algoritmo.fit(X_train, Y_train)
    
    # Faz a previsão usando os dados de teste
    Y_predicted = algoritmo.predict(X_test)
    
    # Calcula a acurácia
    accuracy = np.sum(Y_predicted == Y_test) / len(Y_test)
    
    # Exibe a acurácia
    print('Acurácia: {:.2f}%'.format(100 * accuracy))
    return Y_predicted, accuracy


def calculaValoresAvaliacao(predicted, real):
    # Calcula a acurácia
    accuracy = np.sum(predicted == real) / len(real)
    
    # Exibe a acurácia
    print('Acurácia: {:.2f}%'.format(100 * accuracy))
    
    # Exiba o cabeçalho para as métricas
    print("\n{:<10} {:<12} {:<12} {:<12}".format('Classe', 'Precisão', 'Revocação', 'F1-Score'))
    print('-' * 45)
    
    # Exiba as métricas para cada classe
    for i, label in enumerate(np.unique(predicted)):

        # Calcula a precisão (evita divisão por zero)
        if np.sum(predicted == label) != 0:
            precisao = np.sum((predicted == label) & (real == label)) / np.sum(predicted == label)
        else:
            precisao = 0.0

        # Calcula a revocação (evita divisão por zero)
        if np.sum(real == label) != 0:
            revocacao = np.sum((predicted == label) & (real == label)) / np.sum(real == label)
        else:
            revocacao = 0.0

        # Calcula o F1-Score (evita divisão por zero)
        if (precisao + revocacao) != 0:
            F1 = 2 * (precisao * revocacao) / (precisao + revocacao)
        else:
            F1 = 0.0

        # Exibe as métricas lado a lado
        print("{:<10} {:<12.2f} {:<12.2f} {:<12.2f}".format(label, 100 * precisao, 100 * revocacao, 100 * F1))


def calculaPRFsS(Y_test, Y_predicted, algoritmo):
    """
    Função que calcula a precisão, a revogação, o fscore e o suporte para cada classe da classificação.
    * algoritmo é a instância do algoritmo que foi feito a aprendizado.
    """
    # Calcula as métricas
    precisões, revogações, fscores, suporte = precision_recall_fscore_support(Y_test, Y_predicted, average=None)
    
    # Obtenha as classes na ordem correta
    labels = algoritmo.classes_
    
    # Exiba as métricas para cada classe
    for i, label in enumerate(labels):
        print(f"Classe: {label}")
        print(f"  Precisão: {precisões[i]:.2f}")
        print(f"  Revogação: {revogações[i]:.2f}")
        print(f"  F-score: {fscores[i]:.2f}")
        print(f"  Suporte: {suporte[i]}\n")

# Função que mapeia os semestres em anos "e meio" #
def semester_to_year(semestre):
    
    # Todos os semestres deverão ser convertidos no ano a que se referem... #
    convertido_ano = float(semestre[:-2])

    # ... mas os que forem "/2" deverão contar 0.5 a mais que os /0 e /1. #
    if semestre[-1] == '2':
        convertido_ano = convertido_ano + 0.5

    return convertido_ano

# Função que calcula porcentagem dos tipos (coluna2) para cada grupo (coluna1) #
def totais(dataframe: pd.DataFrame, coluna1: str, coluna2: str):

    # Agrupa e conta as combinações
    contagem = dataframe.groupby([coluna1, coluna2]).size().reset_index(name='Contagem')
    # Calcula a porcentagem
    total_por_tipo = contagem.groupby(coluna1)['Contagem'].transform('sum')
    contagem['Porcentagem'] = (contagem['Contagem'] / total_por_tipo) * 100

    # Retorna um vetor de totais e totais de evasão na ordem alfabética do tipo #
    return contagem

# Função que faz um gráfico de barras de porcentagem dos tipos (coluna2) para cada grupo (coluna1)
def barrasPorcentagemT(dfPorcentagem: pd.DataFrame, coluna1: str, coluna2: str, paleta: str = 'viridis', tamanhos: tuple = (15, 10), limiteinfy: float = None, limitesupy: float = None, bins: int = 10, divisaoGrafico=1, title: str = ""):
    # Verificar número de categorias únicas
    n_categorias = len(dfPorcentagem[coluna1].unique())
    parte_tamanho = n_categorias // divisaoGrafico
    partes = []

    # Dividir o DataFrame em partes
    for i in range(divisaoGrafico):
        inicio = parte_tamanho * i
        fim = parte_tamanho * (i + 1) if i < divisaoGrafico - 1 else n_categorias
        partes.append(dfPorcentagem[dfPorcentagem[coluna1].isin(dfPorcentagem[coluna1].unique()[inicio:fim])])
    
    # Criar subplots
    fig, axes = plt.subplots(divisaoGrafico, 1, figsize=tamanhos, sharey=True, constrained_layout=True)
    
    if divisaoGrafico == 1:
        axes = [axes]  # Converter para lista para facilitar o loop
    
    for i in range(divisaoGrafico):
        ax = sns.barplot(x=coluna1, y='Porcentagem', hue=coluna2, data=partes[i], palette=paleta, ax=axes[i])

        # Ajusta os limites do eixo Y
        if limiteinfy is not None and limitesupy is not None:
            ax.set_ylim(limiteinfy, limitesupy)
        
        # Ajusta os ticks do eixo Y
        if limiteinfy is not None and limitesupy is not None:
            ax.set_yticks(np.arange(limiteinfy, limitesupy + bins, bins))
        
        # Adicionar rótulos de porcentagem nas barras
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            xytext=(0, 5), 
                            textcoords='offset points')
        
        # Ajustar o gráfico
        axes[i].set_xticks(axes[i].get_xticks())  # Garantir o uso de FixedLocator
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].set_xlabel(coluna1)
        axes[i].set_ylabel('Porcentagem')
        if title:
            axes[i].set_title(f'{title} - Parte {i + 1}')
        else:
            axes[i].set_title(f'Porcentagem de {coluna2} por {coluna1} - Parte {i + 1}')
    
    plt.legend(title=coluna2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Função que faz dois gráficos de pizza, um de total e um de evasão, com porcentagem de cada grupo (coluna) #
def entradaEvasao(dataframe: pd.DataFrame, coluna: str, colors: list = ['pink', 'blue'], tamanhos: tuple = (9, 6), title: str = ""):

    # Extrai os valores únicos da coluna especificada e os ordena, criando uma série de rótulos para os gráficos de pizza.
    labels = pd.Series(dataframe[coluna].unique()).sort_values()

    # Agrupa o DataFrame pela coluna especificada e conta o número de ocorrências em cada grupo. Isso cria uma série com os tamanhos para o gráfico de pizza do total de entradas.
    tamanhosEnt = dataframe.groupby(coluna)[coluna].count()

    # Ordena o DataFrame pela coluna especificada e reseta os índices (removendo o índice antigo e criando um novo).
    dataframe = dataframe.sort_values(by=coluna).reset_index(drop=True)

    # Filtra o DataFrame para manter apenas as linhas onde a situação do aluno é "Evasão".
    dataframe = dataframe[dataframe["Evasao_Categoria"] == 1]

    # Agrupa o DataFrame filtrado pela coluna especificada e conta o número de ocorrências em cada grupo. Isso cria uma série com os tamanhos para o gráfico de pizza de evasão.
    tamanhosEva = dataframe.groupby(coluna)[coluna].count()

    # Cria uma figura e dois eixos (ax1 e ax2) dispostos em uma linha com duas colunas, com o tamanho especificado.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tamanhos)

    # Desenha o primeiro gráfico de pizza (total de entradas) usando os tamanhos calculados anteriormente, rótulos, cores especificadas e exibe as porcentagens.
    ax1.pie(tamanhosEnt, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

    # Define o título do primeiro gráfico de pizza.
    ax1.set_title('Total em relação a {}'.format(coluna))

    # Desenha o segundo gráfico de pizza (evasão) usando os tamanhos calculados para evasão, rótulos, cores especificadas e exibe as porcentagens.
    ax2.pie(tamanhosEva, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

    # Define o título do segundo gráfico de pizza.
    ax2.set_title('Evasão em relação a {}'.format(coluna))

    # Verifica se um título foi fornecido como argumento. Se não, define um título padrão.
    if title == "":
        # Define o título principal da figura (engloba ambos os gráficos de pizza) usando a coluna analisada.
        fig.suptitle('Distribuição por {}'.format(coluna), fontsize=16)
    else:
        # Se um título foi fornecido, usa o título especificado.
        fig.suptitle(title, fontsize=16)

    # Ajusta o layout da figura para evitar sobreposição dos gráficos e do título, ajustando os limites do layout.
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Exibe a figura com os gráficos de pizza.
    plt.show()

# Função que faz um gráfico de hexbin entre coluna1 e coluna2 #
def hexBin(dataframe: pd.DataFrame, coluna1: str, coluna2: str, tamanhos: tuple = (10, 6),limiteinfy: float = None, limitesupy: float = None, bins: float = 10, color: str = "Blues"):

    # Cria uma nova figura com o tamanho especificado.
    plt.figure(figsize=tamanhos)

    # Gera um gráfico hexbin, que é um gráfico de dispersão com hexágonos para contagem de pontos em uma grade.
    # A cor dos hexágonos é determinada pelo mapa de cores (cmap), e 'mincnt=1' garante que apenas hexágonos com pelo menos um ponto sejam mostrados.
    plt.hexbin(dataframe[coluna1], dataframe[coluna2], gridsize=30, cmap=color, mincnt=1)

    # Adiciona uma barra de cores à direita do gráfico, que indica a contagem de pontos em cada hexágono.
    plt.colorbar(label="Contagem")

    # Define o título do gráfico, mostrando a relação entre as duas colunas especificadas.
    plt.title("Relação entre {} e {}".format(coluna1, coluna2))

    # Define o rótulo do eixo Y com o nome da segunda coluna.
    plt.ylabel(coluna2)

    # Define o rótulo do eixo X com o nome da primeira coluna.
    plt.xlabel(coluna1)

    # Define os limites do eixo Y, se fornecidos.
    if limiteinfy is not None or limitesupy is not None:
        plt.ylim(limiteinfy, limitesupy)

    # Exibe o gráfico gerado.
    plt.show()

# Função que faz um gráfico de histograma 2D entre a coluna1 e coluna2 #
def histo2D(dataframe: pd.DataFrame, coluna1: str, coluna2: str, binsy: int = 7, binsx: int = 10, color: str = "Blues", tamanhos: tuple = (10, 6), limiteinfy: float = None, limitesupy: float = None):

    # Verifica se a coluna1 é categórica (tipo object ou category) e converte para numérica se necessário.
    if dataframe[coluna1].dtype == 'object' or dataframe[coluna1].dtype.name == 'category':
        # Converte a coluna categórica para um tipo category e atribui códigos numéricos a cada categoria.
        categorias = dataframe[coluna1].astype('category')
        dataframe['Categoria_Num'] = categorias.cat.codes
        # Define os dados para o eixo X como os códigos numéricos das categorias.
        x_data = dataframe['Categoria_Num']
        # Armazena os rótulos das categorias originais para usá-los no eixo X.
        x_labels = categorias.cat.categories
        # Define o número de bins para o eixo X como o número de categorias únicas.
        x_bins = len(x_labels)
    else:
        # Se a coluna1 não for categórica, utiliza os valores da coluna original como dados para o eixo X.
        x_data = dataframe[coluna1]
        # Armazena os valores únicos da coluna1 como rótulos.
        x_labels = dataframe[coluna1].unique()
        # Define o número de bins para o eixo X como o valor passado pelo usuário (binsx).
        x_bins = binsx

    # Cria uma nova figura com o tamanho especificado e plota um histograma 2D com escala logarítmica.
    plt.figure(figsize=tamanhos)
    plt.hist2d(x=x_data, y=dataframe[coluna2], bins=[x_bins, binsy], cmap=color, norm=LogNorm())
    # Adiciona uma barra de cores à direita do gráfico para indicar a contagem de pontos em cada bin.
    plt.colorbar(label="Contagem")

    # Se a coluna1 era categórica, configura os rótulos do eixo X para mostrar os nomes das categorias originais.
    if dataframe[coluna1].dtype == 'object' or dataframe[coluna1].dtype.name == 'category':
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45)

    # Adiciona título e rótulos aos eixos X e Y.
    plt.title("Relação {} e {}".format(coluna1, coluna2))
    plt.xlabel(coluna1)
    plt.ylabel(coluna2)
    
    # Define os limites do eixo Y, se fornecidos.
    if limiteinfy is not None or limitesupy is not None:
        plt.ylim(limiteinfy, limitesupy)

    # Exibe o gráfico gerado.
    plt.show()

# Função que faz um gráfico de pizza das porcentagens de cada tipo (coluna) #
def pizza(dataframe: pd.DataFrame, coluna: str, colors: list = ['pink', 'blue'], tamanhos: tuple = (10, 6), title: str = ''):

    # Obtém os rótulos únicos da coluna, ordenados em ordem crescente.
    labels = sorted(dataframe[coluna].unique())

    # Conta o número de ocorrências de cada rótulo na coluna.
    qnt = dataframe.groupby(coluna)[coluna].count()

    # Define o valor de "explode" para destacar cada fatia da pizza, afastando-as ligeiramente do centro.
    explode = [0.1] * len(labels)
    
    # Cria uma figura e um eixo com o tamanho especificado.
    fig, ax = plt.subplots(figsize=tamanhos)

    # Gera o gráfico de pizza, onde:
    # - qnt são as quantidades para cada fatia,
    # - explode define o afastamento das fatias,
    # - colors especifica as cores,
    # - autopct mostra as porcentagens em cada fatia,
    # - shadow adiciona uma sombra ao gráfico,
    # - startangle define o ângulo inicial para a primeira fatia,
    # - textprops define as propriedades do texto.
    wedges, texts, autotexts = ax.pie(qnt, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops=dict(color="black"))
    
    # Adiciona uma legenda ao gráfico, posicionando-a à esquerda do eixo e ajustando o tamanho das fontes.
    ax.legend(wedges, labels[:len(qnt)], title=title, loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small', title_fontsize='small')
    
    # Assegura que o gráfico de pizza seja desenhado como um círculo, não um oval.
    plt.axis('equal')

    # Ajusta o layout da figura para evitar sobreposição e dar espaço para a legenda.
    plt.tight_layout(rect=[0, 0, 0.75, 0.5])

    # Define o título do gráfico.
    plt.title(title)

    # Exibe o gráfico.
    plt.show()

# Função que cria um gráfico de dispersão (scatter plot) para mostrar a relação entre duas colunas #
def scatterPlot(dataframe: pd.DataFrame, coluna1: str, coluna2: str, label: str, limiteinfy: float = None, limitesupy: float = None, color: str = 'orange', tamanhos: tuple = (10, 6)):
    
    # Cria uma figura e define o tamanho do gráfico.
    plt.figure(figsize=tamanhos)
    
    # Cria um gráfico de dispersão (scatter plot) usando as colunas especificadas.
    dataframe.plot(kind='scatter', x=coluna1, y=coluna2, color=color, label=label)
    
    # Adiciona um rótulo no eixo Y com o nome da `coluna2`.
    plt.ylabel(coluna2)
    
    # Adiciona um rótulo no eixo X com o nome da `coluna1`.
    plt.xlabel(coluna1)
    
    # Define o título do gráfico, descrevendo a relação entre as colunas `coluna1` e `coluna2`.
    plt.title("Relação entre {} e {}".format(coluna1, coluna2))
    
    # Define os limites do eixo Y, se fornecidos.
    if limiteinfy is not None or limitesupy is not None:
        plt.ylim(limiteinfy, limitesupy)
    
    # Exibe o gráfico.
    plt.show()

# Função que cria um gráfico de barras para mostrar a relação entre duas colunas
def barras(group: pd.DataFrame, colors: list = ["red"], limiteinfy: float = None, limitesupy: float = None, bins: int = 10, tamanhos: tuple = (12, 6), labelx: str = "", labely: str = "", title: str = "", legend: str = "", largura_barras: float = 0.8, espacamento: float = 0.2):
    # Verifica se 'group' é uma Series ou DataFrame e ajusta conforme necessário
    if isinstance(group, pd.Series):
        group = group.to_frame()
    
    n_barras = len(group.columns)  # Número de grupos de barras por categoria
    n_categorias = len(group)  # Número de categorias
    
    # Define a posição das barras com base no espaçamento e largura
    indices = np.arange(n_categorias) * (largura_barras + espacamento)
    
    fig, ax = plt.subplots(figsize=tamanhos)  # Cria o gráfico
    
    # Plotando as barras com largura e espaçamento ajustados
    for i, col in enumerate(group.columns):
        barra_positions = indices + i * (largura_barras + espacamento) / n_barras
        ax.bar(barra_positions, group[col], width=largura_barras / n_barras, color=colors[i % len(colors)], label=col)
    
    # Ajusta os limites do eixo Y
    if limiteinfy is not None and limitesupy is not None:
        ax.set_ylim(limiteinfy, limitesupy)
    
    # Ajusta os ticks do eixo Y
    if limiteinfy is not None and limitesupy is not None:
        ax.set_yticks(np.arange(limiteinfy, limitesupy + bins, bins))
    
    # Adiciona os valores das barras no gráfico
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    
    # Adiciona rótulos e título
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.set_title(title)
    
    # Ajusta os rótulos do eixo X
    plt.xticks(indices + (largura_barras / 2), group.index, rotation=45, ha='right')
    
    # Exibe a legenda
    ax.legend(title=legend)
    
    # Exibe o gráfico
    plt.tight_layout()  # Ajusta o layout para evitar sobreposições
    plt.show()

# Função que cria um boxplot para mostrar a distribuição de uma variável y em relação a outra variável x #
def boxPlot(dataframe: pd.DataFrame, x: str, y: str, title: str = "", tamanhos: tuple = (8,5)):
    
    # Configura o tamanho da figura com as dimensões especificadas.
    plt.figure(figsize=tamanhos)
    
    # Cria o boxplot usando o Seaborn (sns). Os dados vêm do `dataframe`.
    # A variável `x` é colocada no eixo X, e a variável `y` no eixo Y.
    g = sns.boxplot(data=dataframe, x='EstadiaEmAnos', y='Situacao_Aluno_Agrupada')
    
    # Adiciona um rótulo no eixo X, utilizando o nome da variável `x`.
    plt.xlabel(x)
    
    # Adiciona um rótulo no eixo Y, utilizando o nome da variável `y`.
    plt.ylabel(y)
    
    # Define o título do gráfico. Se nenhum título for passado, o padrão é uma string vazia.
    plt.title(title)
    
    # Ajusta o layout da figura para evitar sobreposição de elementos.
    plt.tight_layout()
    
    # Exibe o gráfico.
    plt.show()
