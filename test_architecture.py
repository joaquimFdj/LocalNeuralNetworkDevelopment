import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam

import numpy as np
import json
import os
import pickle as pickle
import sys
from datetime import datetime

# CODIGO PARA USAR EM JUPYTER NOTEBOOK
# CODED FOR USAGE IN A JUPYTER NOTEBOOK


class ModeloLocal:

  # SEED USADA - 42
  def __init__(self, input_dim, celula, output_units=1, output_activation='linear',
           alpha=0.0001, random_state=42):

    self.input_dim = input_dim
    self.celula = celula

    self.output_units = output_units
    self.output_activation = output_activation
    self.alpha = alpha
    self.random_state = random_state

    tf.random.set_seed(self.random_state)
    np.random.seed(self.random_state)

    self.model = self._criar_modelo()

  def _criar_modelo(self):
    l2_reg = regularizers.l2(self.alpha)

    modelo = keras.Sequential([
        keras.Input(shape=(self.input_dim,)),

        # camada 1
        layers.Dense(64, kernel_regularizer=l2_reg,
                     kernel_initializer='he_normal', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Dropout(0.2),

        # camada 2
        layers.Dense(32, kernel_regularizer=l2_reg,
                     kernel_initializer='he_normal', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Dropout(0.15),

        # camada 3
        layers.Dense(16, kernel_regularizer=l2_reg,
                     kernel_initializer='he_normal', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.Dropout(0.1),

        # output linear
        layers.Dense(self.output_units,
                     activation=self.output_activation,
                     kernel_initializer='glorot_normal')
    ], name=self.celula)

    return modelo

  def compilar_modelo(self, loss='huber', optimizer=None, metrics=['mae', 'mse']):

        if optimizer is None:
            # optimizer usa gradient clipper para estabilidade
            optimizer = Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                clipnorm=1.0  # busca previnir gradientes explosivos
            )

        # huber loss - robusta a outliers - para tratar problema na celula c3023
        if loss == 'huber':
            loss_fn = keras.losses.Huber(delta=1.0)
        else:
            loss_fn = loss

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

  def _criar_callbacks(self):
    # callbacks inteligentes otimizam treinamento

    callbacks = [
        # para reduzir o learning rate quando parar de melhorar
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=25,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),

        # parar o treinamento se nao melhorar
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=80,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),

        # salvar o melhor
        keras.callbacks.ModelCheckpoint(
            f"best_model_{self.celula}.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
            mode='min'
        )
    ]

    return callbacks

  def treinar(self, X_train, y_train, epochs=1000, batch_size=128,
                validation_split=0.2, verbose=0, use_callbacks=True):
    # o treinamento é otimizado ao utilizar as callbacks
    # - callbacks: ReduceLR + EarlyStopping + ModelCheckpoint

    # - epochs: 1000 (mas o early stopping previne overfitting)
    # - batch_size: 128 (mais atualizações, melhor convergência)

    callbacks = self._criar_callbacks() if use_callbacks else []

    history = self.model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )

    return history

  def prever(self, X, verbose=0):
    return self.model.predict(X, verbose=verbose)

  def salvar_no_caminho(self, caminho):
    self.model.save(caminho)

  def carregar_de_caminho(self, caminho):
    self.model = keras.models.load_model(caminho)


if __name__ == "__main__":

    print("versao tf -", tf.__version__)
    print("versao keras -", keras.__version__)

    pasta_inputs = "/content/drive/MyDrive/some_path"
    pasta_scalers = "/content/drive/MyDrive/some_path"
    CELULAS = ["c3023", "c3024",
               "c3025", "c3026", "c3027"]

    inputs_drive = {}
    scalers_drive = {}
    for celula in CELULAS:
        with open(f'{pasta_inputs}inputs_{celula}.pkl', 'rb') as f:
            inputs_drive[celula] = pickle.load(f)

        with open(f'{pasta_scalers}scalers_{celula}.pkl', 'rb') as f:
            scalers_drive[celula] = pickle.load(f)

    # 3 FEATURES:
    print(inputs_drive[CELULAS[0]]['X_train_scaled'].shape)

###############################################################################

    print("----- INICIANDO CRIACAO E SALVAMENTO DOS MODELOS -----\n")

    CELULAS = ["c3023", "c3024",
               "c3025", "c3026", "c3027"]
    for celula in CELULAS:

        X_train = inputs_drive[celula]['X_train_scaled']
        y_train = inputs_drive[celula]['y_train']
        input_dim = inputs_drive[celula]['X_train_scaled'].shape[1]

        print(f"Iniciando para celula {celula}")
        modelo = ModeloLocal(input_dim, celula)
        modelo.compilar_modelo()
        print("criou e compilou com sucesso")

        print(f"Iniciando treinamento")
        history = modelo.treinar(X_train, y_train)
        print("treinou com sucesso")

        print(f"iniciando salvamento no Drive")

        caminho = ""
        try:
            if not os.path.exists(caminho):
                raise FileNotFoundError
        except FileNotFoundError:
            print("Caminho -  -",
                  "NAO ENCONTRADO")
            sys.exit()
        pasta_modelo = os.path.join(caminho, celula)
        try:
            os.mkdir(pasta_modelo)
        except FileExistsError:
            print(f"pasta {pasta_modelo} ja existe.")
            sys.exit()

        nome_modelo = f"Local{celula}.keras"
        modelo.salvar_no_caminho(os.path.join(pasta_modelo, nome_modelo))

        info_modelo = {
            "celula": celula,
            "data treino": str(datetime.now().date()),
            "arquitetura": "'robusta-funil v1'",
            "notebook": "ModelosLocais.ipynb",
            "salvo em": pasta_modelo
        }
        nome_json = f"info_modelo_{celula}.json"
        with open(os.path.join(pasta_modelo, nome_json),
                  'w', encoding='utf-8') as f:
            json.dump(info_modelo, f, ensure_ascii=False, indent=4)

        caminho_scaler = os.path.join(pasta_modelo, f"scaler_modelo_{celula}.pkl")
        caminho_inputs = os.path.join(pasta_modelo, f"inputs_modelo_{celula}.pkl")
        with open(caminho_scaler, 'wb') as f:
            pickle.dump(scalers_drive[celula], f)
        with open(caminho_inputs, 'wb') as f:
            pickle.dump(inputs_drive[celula], f)

        print("salvou modelo celula -", celula, "- com sucesso\n\n")
