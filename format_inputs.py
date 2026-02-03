import pandas as pd
import numpy as np

import os
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# CODIGO PARA JUPYTER NOTEBBOK
# CODED FOR JUPYTER NOTEBOOK

base_path = "/content/drive/MyDrive/DATASET"
CELULAS = ["c3023", "c3024",
           "c3025", "c3026", "c3027"]
# dicionario com: - NOME DA CELULA/DATASET DELA
dataframes = {}
for celula in CELULAS:
  try:
    dataframes[celula] = pd.read_parquet(f'{base_path}/df_unificado_{celula}.parquet')
  except FileNotFoundError:
    print(f"Arquivo para célula {celula} não encontrado")


# DIVIDE O DATASET DE DETERMINADA CELULA DO DICIONARIO EM DADOS DE TREINO E TESTE
#######
# RETORNA DICIONARIO COM OS DADOS ORGANIZADOS E O SCALER UTILIZADO
def split_scale(dic_dataframes, celula_):

  try:

    X = dic_dataframes[celula_].drop(columns=['soc_corrigido', 'timestamp',
                                             'capacidade']).values\
                                             .astype('float32')

    y = dic_dataframes[celula_]['soc_corrigido'].values.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }, scaler_x

  except KeyError:
    print("erro: celula fornecida nao esta no dicionario.")
    return None


# APLICA FUNÇÃO splitScale() NO DATASET DE CADA CELULA DO PROJETO E
# GERANDO E RETORNANDO DOIS DICIONARIOS:
# nome da celula/dicionario com inputs ||| nome da celula/scaler
def criar_inputs_celulas(dataframes_dict):
  try:

    celulas = ["c3023", "c3024",
           "c3025", "c3026", "c3027"]
    inputs_dict = {}
    scalers_dict = {}

    for celula in celulas:

      inputs, scalers = split_scale(dataframes_dict, celula)
      inputs_dict[celula] = inputs
      scalers_dict[celula] = scalers

    return inputs_dict, scalers_dict

  except KeyError:
    print('erro: celula nao esta no dicionario')
    return None


# USE EXAMPLE
if __name__ == "__main__":

    inputs, scalers = criar_inputs_celulas(dataframes)

    pasta_scalers = "/content/drive/MyDrive/some_path"
    pasta_inputs = "/content/drive/MyDrive/some_path"
    celulas = ['c3023', 'c3024', 'c3025', 'c3026',
               'c3027']

    celula_exemplo = celulas[0]

    exemplo_scaler = str(scalers[celula_exemplo])
    exemplo_input = str(inputs[celula_exemplo])

    info_scalers = {
        "Ex dado salvo em um dos arquivos 'scalers_celula.pkl'":
            exemplo_scaler,
        "Scaler se refere a":
            "X_train e X_test"
    }
    info_inputs = {
        "nomes": [
            "'X_train_scaled'",
            "'X_test_scaled'",
            "'y_train'",
            "'y_test'"
        ],
        "exemplo arrays": exemplo_input
    }

    with open(os.path.join(pasta_scalers, "exemplo_scalers.json"), "w",
              encoding="utf-8") as f:
        json.dump(info_scalers, f, indent=2, ensure_ascii=False)

    with open(os.path.join(pasta_inputs, "exemplo_inputs.json"), "w",
              encoding="utf-8") as f:
        json.dump(info_inputs, f, indent=2, ensure_ascii=False)

    for celula in celulas:
        with open(f'{pasta_scalers}scalers_{celula}.pkl', 'wb') as f:
            pickle.dump(scalers[celula], f)

        with open(f'{pasta_inputs}inputs_{celula}.pkl', 'wb') as f:
            pickle.dump(inputs[celula], f)
            