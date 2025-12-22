# -*- coding: utf-8 -*-
import os
import sys
import re
from typing import List, Tuple
from pickle import dump, load

# For data and plotting 
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import numpy as np
import cv2

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.base import TransformerMixin

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from focal_loss import BinaryFocalLoss

# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "default"
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# para crear mapa de flujo pydot y graphviz
try:
    import pydot
except ImportError:
    print("pydot no esta instalado. Instalando...")
    os.system('pip install pydot')
    import pydot

try:
    import graphviz
except ImportError:
    print("graphviz no esta instalado. Instalando...")
    os.system('pip install graphviz')
    import graphviz

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

    def save_model(self, model_name):
        dump(self._scaler, open(model_name + '.bin', 'wb'))

    def load_model(self, model_path, X):
        self._scaler = load(open(model_path, 'rb'))
        self._orig_shape = X.shape[1:]

def r2_score(y_true, y_pred):
    # Suma de los cuadrados de los residuos
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))

    # Media de los valores reales
    mean_y_true = tf.reduce_mean(y_true)

    # Suma de los cuadrados totales
    sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))

    # Evitar division por cero ESTO LO HE PUESTO YO PGS
    epsilon = tf.keras.backend.epsilon()
    
    sum_squares = tf.maximum(sum_squares, epsilon)

    # Coeficiente de determinacion R2
    R2 = 1 - sum_squares_residuals / sum_squares
    return R2

class ResNet18:

    tf.keras.backend.clear_session()

    @staticmethod
    def identity_block(X, filters: List[int]):
        # unpack number of filters to be used for each conv layer
        f1, f2 = filters
        X_shortcut = X

         # first convolutional layer (plus batch norm & relu activation)
        X = Conv2D(f1, (5,5), strides=(1,1), padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X)

        # second convolutional layer
        X = Conv2D(f1, (5,5), strides=(1,1), padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.4, epsilon=0.001, scale=False)(X)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])
        
        # relu activation at the end of the block
        X = Activation(tf.keras.layers.LeakyReLU(alpha=0.1))(X)

        return X

    @staticmethod
    def convolutional_block(X, filters: List[int], strides: Tuple[int,int]=(2,2)):
        
        # unpack number of filters to be used for each conv layer
        f1, f2 = filters
        
        # the shortcut branch of the convolutional block
        X_shortcut = X

        # first convolutional layer
        X = Conv2D(f1, (5,5), strides=strides, padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X)

        # second convolutional layer
        X = Conv2D(f1, (5,5), strides=(1,1), padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.4, epsilon=0.001, scale=False)(X)

        # shortcut path
        X_shortcut = Conv2D(f2, (1,1), strides=strides, padding='same',
                            kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                            bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X_shortcut)                         
        X_shortcut = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X_shortcut)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])
        
        # nonlinearity
        X = Activation(tf.keras.layers.LeakyReLU(alpha=0.1))(X)

        return X

    @staticmethod 
    def build(input_size: Tuple[int,int,int], classes: int) -> Model:
    
        # tensor placeholder for the model's input
        X_input = Input(input_size)

        # convolutional layer, followed by batch normalization and relu activation
        X = Conv2D(16, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform',
                   activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X_input)
        X = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X)
        X = Activation(tf.keras.layers.LeakyReLU(alpha=0.1))(X)
        
        # max pooling layer to halve the size coming from the previous layer
        X = MaxPooling2D((4, 4), strides=(2,2), padding='same')(X)

        # Blocks
        X = ResNet18.convolutional_block(X, [16,16], strides=(1,1))
        X = ResNet18.identity_block(X, [16,16])

        X = ResNet18.convolutional_block(X, [32,32])
        X = ResNet18.identity_block(X, [32,32])

        X = ResNet18.convolutional_block(X, [64,64])
        X = ResNet18.identity_block(X, [64,64])

        X = ResNet18.convolutional_block(X, [128,128])
        X = ResNet18.identity_block(X, [128,128])

        # Pooling layers
        X = MaxPooling2D((4, 4), padding='same')(X)
        
        # Convert feature maps ? vector
        X = GlobalAveragePooling2D()(X)
        
        # Output layer
        X = Dense(classes, activation='sigmoid')(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet18_5x5')
        return model

# Crea el modelo a partir de las distintas branches / features creadas con cnn_branch 
def auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str):
    combinedInput = tf.keras.layers.concatenate([cnn_div.output, cnn_cov.output, cnn_dot.output, cnn_str.output])

    # layer 1
    layers = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="dense_5_1")(combinedInput)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_1")(layers)   
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    
    # layer 2
    layers = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01),
                                   name="dense_5_2")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_2")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    # layer 3
    layers = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01),
                                   name="dense_5_3")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_3")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    # layer 4
    layers = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="dense_5_4")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_4")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    # layer 5
    layers = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="dense_5_5")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_5")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)

    # layer 6
    predictions = tf.keras.layers.Dense(2, activation="sigmoid", name="output_5")(layers)

    model = tf.keras.Model(inputs=[cnn_div.input, cnn_cov.input, cnn_dot.input, cnn_str.input], outputs=predictions)
    
    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Compile model
    model.compile(loss="mse", optimizer=opt, metrics=[r2_score])
    return model

def testing_model(model_path, dataX_path, dataY_path):
    # Load X_test (tuple of 4 arrays) and Y_test
    X_test_tuple = np.load(dataX_path, allow_pickle=True) 
    Y_test = np.load(dataY_path).astype(np.float32) 

    # Combine channels in an array (N, H, W, 4)
    X_test = np.concatenate(X_test_tuple, axis=-1)  

    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    print(f"NaNs en X_test: {np.isnan(X_test).sum()}, Y_test: {np.isnan(Y_test).sum()}")

    # Load model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU(0.1), 'r2_score': r2_score})

    # Predictions
    predictions = model.predict(
        [X_test[:, :, :, 0], X_test[:, :, :, 1], X_test[:, :, :, 2], X_test[:, :, :, 3]], verbose=0)

    predictions = np.nan_to_num(predictions, nan=0)

    print("NaNs in predictions:", np.isnan(predictions).sum())
    print("predictions shape:", predictions.shape)

    # Calculate R2
    r2_initial = plot_results(Y_test[:, 0], predictions[:, 0], "StartingPos")
    r2_final = plot_results(Y_test[:, 1], predictions[:, 1], "EndingPos")
    print("R2 starting position:" + str(r2_initial))
    print("R2 ending position:" + str(r2_final))

    return predictions


def plot_results(real, predicted, name):
    print(f"Calculating R2 for {name}...")
    print(f"NaNs in real: {np.isnan(real).sum()}, NaNs in predicted: {np.isnan(predicted).sum()}")
    print(f"real shape: {real.shape}, predicted shape: {predicted.shape}")

    # Verificar que las formas de los datos coincidan
    if real.shape != predicted.shape:
        raise ValueError("Shapes of real and predicted do not match.")

    # Calcular el coeficiente de determinacion R^2
    r2 = r2_score(real, predicted)

    # Crear el grafico
    plt.figure(figsize=(8, 6))
    plt.scatter(real, predicted, color='blue', label='Datos')
    plt.plot(real, real, color='red', label=f'Referencia (y=x, R^2 = {r2:.2f})')
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.title('Grafico de dispersion con R^2 - ' + name)
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_' + name + '.png', bbox_inches='tight', dpi=500)

    return r2
