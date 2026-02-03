import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

# FEITO PARA USO EM JUPYTER NOTEBOOK
# CODED FOR USAGE IN A JUPYTER NOTEBOOK


# METRICAS AVALIADAS:
# DE ERRO - MSE, RMSE, MAE, MAPE, erro máximo, erro médio e percentil 95 do erro.
# DE QUALIDADE - R2, R2 Ajustado, Variancia Explicada

# GRAFICOS GERADOS:
# Comparação entre predições e valores reais.
# Grafico para analise residual
# Visualização da distribuição de erros
# Visualização dos intervalos de predição
class AvaliadorModeloRegressao:


    def __init__(self):
      self.results = {}

    def avaliar_modelo(self, modelo, X_test, y_test, nome_modelo):

      y_pred = modelo.predict(X_test, verbose=0).flatten()
      y_test = np.array(y_test).flatten()

      residuals = y_test - y_pred

      mask = y_test != 0
      if np.any(mask):
        mape_score = np.mean(np.abs((y_test[mask] - y_pred[mask]) /
                                    y_test[mask])) * 100
      else:
        mape_score = 0.0

      metrics = {
              'MSE': mean_squared_error(y_test, y_pred),
              'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
              'MAE': mean_absolute_error(y_test, y_pred),
              'MAPE': mape_score,
              'R2': r2_score(y_test, y_pred),
              'R2 Ajustado': self._adjusted_r2(y_test, y_pred, X_test.shape[1]),
              'Variancia Explicada': explained_variance_score(y_test, y_pred),
              'Erro Medio': np.mean(residuals),
              'Desvio Padrao dos Erros': np.std(residuals),
              'Erro Maximo': np.max(np.abs(residuals)),
              'Percentil 95 do Erro': np.percentile(np.abs(residuals), 95)
          }

      self.results[nome_modelo] = {
              'model': modelo,
              'metrics': metrics,
              'y_true': y_test,
              'y_pred': y_pred,
              'residuals': residuals,
              'X_test': X_test
          }

      print(f"Modelo - '{nome_modelo}' - avaliado com sucesso.")
      print(f"#### VISÃO INICIAL DO MODELO DA CELULA {nome_modelo} ####")
      print(f"> Erro Médio - {metrics['Erro Medio']}")
      print(f"> Desvio Padrão dos Erros - {metrics['Desvio Padrao dos Erros']}")
      print(f"> Percentil 95 do Erro - {metrics['Percentil 95 do Erro']}\n")

    def _adjusted_r2(self, y_true, y_pred, n_features):
        n = len(y_true)
        if n <= n_features + 1:
          return np.nan

        r2 = r2_score(y_true, y_pred)
        adjusted = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted

    def plotar_comparacao_metricas(self, figsize=(18, 6)):
      if not self.results:
        print("Nenhum modelo avaliado ainda.")
        return

      metrics_df = pd.DataFrame({name: res['metrics']
                                 for name, res in self.results.items()}).T

      error_metrics = ['MSE', 'RMSE', 'MAE', 'Erro Maximo']
      quality_metrics = ['R2', 'R2 Ajustado', 'Variancia Explicada']
      mape_metric = ['MAPE']

      fig, axes = plt.subplots(1, 3, figsize=figsize)

      if all(m in metrics_df.columns for m in error_metrics):

            metrics_df[error_metrics].plot(kind='bar', ax=axes[0],
                                                width=0.8)
            axes[0].set_xlabel('Modelo', fontsize=12)
            axes[0].set_ylabel('Métrica de Erro', fontsize=12)
            axes[0].set_title('Métricas de Erro', fontsize=14,
                              fontweight='bold')
            axes[0].legend(title='Métricas', bbox_to_anchor=(1.05, 1),
                           loc='upper left')
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].tick_params(axis='x', rotation=45)

      if all(m in metrics_df.columns for m in quality_metrics):
        metrics_df[quality_metrics].plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_xlabel('Modelo', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Métricas de Qualidade do Ajuste',
                      fontsize=14, fontweight='bold')
        axes[1].legend(title='Métricas', bbox_to_anchor=(1.05, 1),
                        loc='upper left')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)

        values = metrics_df[quality_metrics].values.flatten()
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 0.05:
          margin = 0.01
          axes[1].set_ylim(max(0, vmin - margin), min(1.01, vmax + margin))
        else:
          axes[1].set_ylim([0, 1.1])

      if all(m in metrics_df.columns for m in mape_metric):
            metrics_df[mape_metric].plot(kind='bar', ax=axes[2], width=0.6,
                                         color='tab:red')
            axes[2].set_xlabel('Modelo', fontsize=12)
            axes[2].set_ylabel('MAPE (%)', fontsize=12)
            axes[2].set_title('Erro Percentual Médio Absoluto (MAPE)',
                              fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3, axis='y')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].legend().remove()

      plt.tight_layout()
      plt.show()

      return metrics_df

    def plotar_predicoes_vs_real(self, figsize=(16, 5)):
      if not self.results:
        print("Nenhum modelo avaliado ainda!")
        return

      n_models = len(self.results)
      cols = min(3, n_models)
      rows = (n_models + cols - 1) // cols

      fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
      axes = axes.flatten() if n_models > 1 else [axes]

      for idx, (name, result) in enumerate(self.results.items()):
        ax = axes[idx]
        ax.scatter(result['y_true'], result['y_pred'], alpha=0.6, s=30,
                    edgecolors='k', linewidth=0.5)

        min_val = min(result['y_true'].min(), result['y_pred'].min())
        max_val = max(result['y_true'].max(), result['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
                  linewidth=2, label='Predição Perfeita')

        r2 = result['metrics']['R2']
        ax.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Valores Reais', fontsize=11)
        ax.set_ylabel('Valores Preditos', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
      for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

      plt.tight_layout()
      plt.show()

    def plotar_analise_residual(self, figsize=(16, 10)):
      if not self.results:
        print("Nenhum modelo avaliado ainda.")
        return

      n_models = len(self.results)
      fig, axes = plt.subplots(n_models, 3, figsize=figsize)

      if n_models == 1:
        axes = axes.reshape(1, -1)

      for idx, (name, result) in enumerate(self.results.items()):
        axes[idx, 0].scatter(result['y_pred'], result['residuals'],
                               alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        axes[idx, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[idx, 0].set_xlabel('Valores Preditos', fontsize=10)
        axes[idx, 0].set_ylabel('Resíduos', fontsize=10)
        axes[idx, 0].set_title(f'{name} - Resíduos vs Predições', fontsize=11,
                               fontweight='bold')
        axes[idx, 0].grid(True, alpha=0.3)

        axes[idx, 1].hist(result['residuals'], bins=30,
                            edgecolor='black', alpha=0.7, color='skyblue')
        axes[idx, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[idx, 1].set_xlabel('Resíduos', fontsize=10)
        axes[idx, 1].set_ylabel('Frequência', fontsize=10)
        axes[idx, 1].set_title(f'{name} - Distribuição dos Resíduos',
                               fontsize=11, fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3, axis='y')

        from scipy import stats
        stats.probplot(result['residuals'], dist="norm", plot=axes[idx, 2])
        axes[idx, 2].set_title(f'{name} - Q-Q Plot', fontsize=11,
                               fontweight='bold')
        axes[idx, 2].grid(True, alpha=0.3)

      plt.tight_layout()
      plt.show()

    def plotar_distribuicao_erros(self, figsize=(14, 6)):
      if not self.results:
        print("Nenhum modelo avaliado ainda!")
        return

      fig, axes = plt.subplots(1, 2, figsize=figsize)

      error_data = [np.abs(res['residuals']) for res in self.results.values()]
      axes[0].boxplot(error_data, tick_labels=list(self.results.keys()))
      axes[0].set_ylabel('Erro Absoluto', fontsize=12)
      axes[0].set_title('Distribuição de Erros Absolutos', fontsize=14,
                        fontweight='bold')
      axes[0].grid(True, alpha=0.3, axis='y')
      axes[0].tick_params(axis='x', rotation=45)

      positions = range(1, len(self.results) + 1)
      parts = axes[1].violinplot(error_data, positions=positions,
                                  showmeans=True, showmedians=True)
      axes[1].set_xticks(positions)
      axes[1].set_xticklabels(list(self.results.keys()), rotation=45,
                              ha='right')
      axes[1].set_ylabel('Erro Absoluto', fontsize=12)
      axes[1].set_title('Distribuição de Erros (Violin Plot)', fontsize=14,
                        fontweight='bold')
      axes[1].grid(True, alpha=0.3, axis='y')

      plt.tight_layout()
      plt.show()

    def plotar_intervalos_predicao(self, confidence=0.95, figsize=(14, 6)):
      if not self.results:
        print("nenhum modelo avaliado ainda.")
        return

      n_models = len(self.results)
      cols = min(2, n_models)
      rows = (n_models + cols - 1) // cols

      fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
      axes = axes.flatten() if n_models > 1 else [axes]

      for idx, (name, result) in enumerate(self.results.items()):
        std_residuals = np.std(result['residuals'])
        z_score = 1.96 if confidence == 0.95 else 2.576

        sort_idx = np.argsort(result['y_true'])
        y_true_sorted = result['y_true'][sort_idx]
        y_pred_sorted = result['y_pred'][sort_idx]

        axes[idx].scatter(range(len(y_true_sorted)), y_true_sorted,
                            s=8, label='Real', color='black')
        axes[idx].plot(range(len(y_pred_sorted)), y_pred_sorted, alpha=0.7,
                          color='red', linewidth=2, label='Predito')

        axes[idx].fill_between(range(len(y_pred_sorted)),
                                y_pred_sorted - z_score * std_residuals,
                                y_pred_sorted + z_score * std_residuals,
                                alpha=0.3, color='blue',
                                label=f'IC {int(confidence*100)}%')

        axes[idx].set_xlabel('Amostras (ordenadas)', fontsize=11)
        axes[idx].set_ylabel('Valor', fontsize=11)
        axes[idx].set_title(f'{name} - Intervalos de Predição',
                            fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

      for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

      plt.tight_layout()
      plt.show()


# USE EXAMPLE
if __name__ == "__main__":

    # ____________ CONFIGURE ENVIROMENT ______________
    pasta_modelos = "/content/drive/MyDrive/some_path"
    pasta_scalers = "/content/drive/MyDrive/some_path"
    pasta_inputs = "/content/drive/MyDrive/some_path"
    CELULAS = ["c3023", "c3024",
               "c3025", "c3026", "c3027"]

    modelos_drive = {}
    scalers_drive = {}
    inputs_drive = {}
    for celula in CELULAS:
        modelos_drive[celula] = tf.keras.models.load_model(
            os.path.join(pasta_modelos, celula, f"modelo_local_{celula}.keras"))

        with open(f'{pasta_scalers}scalers_{celula}.pkl', 'rb') as f:
            scalers_drive[celula] = pickle.load(f)

        with open(f'{pasta_inputs}inputs_{celula}.pkl', 'rb') as f:
            inputs_drive[celula] = pickle.load(f)
    # ___________________________________________________________

    avaliador = AvaliadorModeloRegressao()

    for celula, modelo in modelos_drive.items():
        inputs = inputs_drive[celula]
        X_test = inputs['X_test_scaled']
        y_test = inputs['y_test']

        avaliador.avaliar_modelo(modelo, X_test, y_test, celula)

    # ______________________________
    # Gerar Visualização comparativa das metricas do modelo em
    # cada celula:
    avaliador.plotar_comparacao_metricas()
    # ______________________________
    # plota os graficos de comparacao entre predições
    # e valores reais do modelo para cada celula
    avaliador.plotar_predicoes_vs_real()
    # ______________________________
    # analise visual dos residuos do modelo para cada celula
    avaliador.plotar_analise_residual()
    # ______________________________
    # visualizar a distribuicao dos erros do modelo para cada celula
    avaliador.plotar_distribuicao_erros()
    # ______________________________
    # visualizar os intervalos de predicao do modelo para cada celula
    avaliador.plotar_intervalos_predicao()
    # ______________________________