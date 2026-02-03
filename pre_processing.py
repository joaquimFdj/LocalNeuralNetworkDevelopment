import os
import pandas as pd
import numpy as np

# CODIGO PARA USO EM JUPYTER NOTEBOOKS
# CODED FOR BEING USED IN JUPYTER NOTEBOOKS


# "celula" é a celula que se deseja ler (ler CSVs dessa celula)
def ler_csvs_de_uma_celula(celula, base_path):

  todos_arquivos = os.listdir(base_path)
  arquivos_celula = [f for f in todos_arquivos if f.startswith(celula + "-")
   and f.endswith(".csv")]

  arquivos_celula.sort(key=lambda f: int(f.rsplit('-', 1)[1].split('.')[0]))

  dataframes = {}
  for nome_arquivo in arquivos_celula:
    caminho_completo = os.path.join(base_path, nome_arquivo)
    df = pd.read_csv(caminho_completo)
    dataframes[nome_arquivo] = df
  return dataframes


def limpar_dataframes(dic_dataframes):
  clean_dfs = {}
  for nome_df, df in dic_dataframes.items():
    df_clean = df.dropna()
    df_clean.columns = df_clean.columns.str.lower().str.strip()
    clean_dfs[nome_df] = df_clean
  return clean_dfs


def print_info_df(nome_arquivo, df):
  print("#####################################")
  print("ARQUIVO -", nome_arquivo, ":")
  print("shape dos dados no arquivo -", df.shape)
  print("\nInfo dos dados:")
  df.info()
  print("\nPrimeiras 5 linhas:")
  print(df.head())
  print("\nEstatisticas descritivas:")
  print(df.describe())
  print("\nValores unicos por coluna:")
  print(df.nunique())
  print("\npercentual de missing por coluna:")
  print(((df.isnull().sum() / len(df)) * 100).round(2))
  print("\nLinhas duplicadas:", df.duplicated().sum())
  print("#####################################")


def info_basica_celula(dic_dataframes, quant_a_ser_lida='all'):
  dic_len = len(dic_dataframes)

  if quant_a_ser_lida == 'all':
    print(f'LENDO TODOS OS ARQUIVOS DA CELULA ({dic_len}):\n')
    for nome_arquivo, df in dic_dataframes.items():
      print_info_df(nome_arquivo, df)
    return

  if isinstance(quant_a_ser_lida, int) and quant_a_ser_lida > 0:
    if quant_a_ser_lida > dic_len:
      print(f'>>>A QUANT INSERIDA ({quant_a_ser_lida}) E MAIOR QUE O NUMERO DE ARQUIVOS ({dic_len}) NA CELULA!!!\n')
      print(f'LENDO TODOS OS ARQUIVOS DA CELULA ({dic_len}):\n')
      for nome_arquivo, df in dic_dataframes.items():
        print_info_df(nome_arquivo, df)

    elif quant_a_ser_lida == dic_len:
      print(f'QUANT INSERIDA = TOTAL DE ARQUIVOS NA CELULA ({dic_len})\n')
      print(f'LENDO TODOS OS ARQUIVOS DA CELULA ({dic_len}):\n')
      for nome_arquivo, df in dic_dataframes.items():
        print_info_df(nome_arquivo, df)

    elif quant_a_ser_lida < dic_len:
      print(f"LENDO {quant_a_ser_lida} ARQUIVOS DA CELULA:\n")

      it = iter(dic_dataframes.items())
      counter = 0
      while counter < quant_a_ser_lida:
        nome_arquivo, df = next(it)
        print_info_df(nome_arquivo, df)
        counter +=1
  else:
    print(f'parametro quant_a_ser_lida invalido: {quant_a_ser_lida}')
    return


# cria a coluna capacidade segundo formula que pressupoe que os valores de corrente
# (coluna c) estão em amperes e os timestamps são, invariavelmente, de 10 em 10 segundos
def corrigir_soc(dic_dataframes, drop_soc_antigo=False):
  for df_name, dataframe in dic_dataframes.items():
    dataframe['capacidade'] = (dataframe['c'].cumsum() / 360)
    capacidade_total = dataframe['capacidade'].iloc[-1]

    dataframe['soc_corrigido'] = 100 - ((dataframe['capacidade'] / capacidade_total) * 100)

    if drop_soc_antigo:
      dataframe = dataframe.drop(columns=['soc'])

    dic_dataframes[df_name] = dataframe

  return dic_dataframes


# remover outliers do dataset utilizando a tecnica com IQR (usar se necessario)
def removerOutliers(df, features):
  mask = pd.Series(True, index=df.index)

  for feature in features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    feature_mask = (df[feature] >= limite_inferior) & (df[feature] <= limite_superior)

    mask = mask & feature_mask

  df_clean = df[mask].copy()
  return df_clean


# EXEMPLO DE USO // USAGE EXAMPLE:
if __name__ == "__main__":

    TODAS_AS_CELULAS = ("c3023", "c3024",
                        "c3025", "c3026",
                        "c3027")
    path_dataset_celulas = "/content/drive/MyDrive/DATASET/dataset"

    for celula in TODAS_AS_CELULAS:
        dic_dataframes = ler_csvs_de_uma_celula(celula, path_dataset_celulas)

        dfs_corrigidos = {
            nome: df.drop(columns='unnamed: 0')
            for nome, df in corrigir_soc(limpar_dataframes(dic_dataframes), drop_soc_antigo=True).items()
        }

        df_unificado = pd.concat(dfs_corrigidos.values(), ignore_index=True)
        diretorio_destino = f'/content/drive/MyDrive/DATASET/df_unificado_{celula}.parquet'
        df_unificado.to_parquet(diretorio_destino, compression='snappy')