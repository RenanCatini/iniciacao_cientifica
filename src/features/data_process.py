# Aqui estão os dados das previsões
path_dados = "C:/Users/renan/Arquivos Unifal/iniciacao_cientifica/data/processed/Dados_Arritimia_final.csv"

import pandas as pd

# Importar os dados dos exames
dados = pd.read_csv(path_dados)
dados.dropna(inplace=True)

# Separarar os dados de treino e retirar colunas "não úteis"
x = dados.drop(columns=["Idade", "Canal (Derivação)", "Sexo", "Medicamentos", "Registro", "Rotulos", "Rotulos_Nome", "Indicador_Zero", "Padrao_AAMI", "Tipo"])
y = dados["Padrao_AAMI"]
y_binary = [0 if label == "N" else 1 for label in y]

# Separar treino e teste para o binário
from sklearn.model_selection import train_test_split

# Dados separados em 70% teste e 30% treino, e divsão proporcional dos dados
x_train, x_test, y_binary_train, y_binary_test = train_test_split(
    x, y_binary, test_size=0.3, random_state=42, stratify=y_binary)