import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as r2_sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# === Clase NDStandardScaler ===
class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
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
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

    def save_model(self, model_name):
        dump(self._scaler, model_name + '.bin')

    def load_model(self, model_path, X):
        self._scaler = load(model_path)
        self._orig_shape = X.shape[1:]

# === Funcion r2_score para Keras ===
def r2_score(y_true, y_pred):
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y_true = tf.reduce_mean(y_true)
    sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))

    epsilon = tf.keras.backend.epsilon()
    sum_squares = tf.maximum(sum_squares, epsilon)

    return 1 - sum_squares_residuals / sum_squares

# === Argumentos con argparse ===
parser = argparse.ArgumentParser(description="SmartInspection Prediction Script")
parser.add_argument("mode", type=str, help="Execution mode (e.g. test)")
parser.add_argument("model_path", type=str, help="Path to trained Keras model")
parser.add_argument("scalerX_path", type=str, help="Path to saved scaler")
parser.add_argument("dataX_path", type=str, help="Path to X_test .npy file")
parser.add_argument("dataY_path", type=str, help="Path to Y_test .npy file")
args = parser.parse_args()

def calcular_y_graficar_r23(real, predicted, nombre):
    print(f"Calculating R2 for {nombre}...")
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
    plt.title('Grafico de dispersion con R^2 - ' + nombre)
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_' + nombre + '.png', bbox_inches='tight', dpi=500)

    return r2
    
def testing_models2(model_path, scalerx_path, dataX_path, dataY_path):

    # === Load test data ===
    X_test = np.load(dataX_path)
    X_test = X_test / 255.0
    X_test = X_test.astype(np.float32)
    
    Y_test = np.load(dataY_path).astype(np.float32)
    
    sample_names = np.array([f"Muestra_{i+1}" for i in range(len(Y_test))])
    
    print("X_test raw:", np.min(X_test), np.max(X_test))
    print("Y_test raw:", np.min(Y_test), np.max(Y_test))
    
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("NaNs in X_test:", np.isnan(X_test).sum())
    print("NaNs in Y_test:", np.isnan(Y_test).sum())

    # === Load scaler ===
    scalerX = NDStandardScaler()
    scalerX.load_model(scalerx_path, X_test)   # <-- solo la ruta

    # === Apply scaling ===
    X1_test_scl = scalerX.transform(X_test)
    
    print("NaNs in X1_test_scl after scaling:", np.isnan(X1_test_scl).sum())

    # === Load model ===
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'LeakyReLU': tf.keras.layers.LeakyReLU(0.1),
            'r2_score': r2_score
        }
    )

    # === Predict ===
    predictions = model.predict(
        [
            X1_test_scl[:, :, :, :, 0],
            X1_test_scl[:, :, :, :, 1],
            X1_test_scl[:, :, :, :, 2],
            X1_test_scl[:, :, :, :, 3]
        ],
        verbose=0
    )

    predictions = np.nan_to_num(predictions, nan=0)

    print("NaNs in predictions:", np.isnan(predictions).sum())
    print("predictions shape:", predictions.shape)

    r2_initial = calcular_y_graficar_r23(Y_test[:, 0], predictions[:, 0], "StartingPos")
    r2_final = calcular_y_graficar_r23(Y_test[:, 1], predictions[:, 1], "EndingPos")

    print("R2 starting position:", r2_initial)
    print("R2 ending position:", r2_final)

    # === Metrics ===
    metrics = {}
    for i in range(Y_test.shape[1]):
        mse = mean_squared_error(Y_test[:, i], predictions[:, i])
        mae = mean_absolute_error(Y_test[:, i], predictions[:, i])
        r2 = r2_sklearn(Y_test[:, i], predictions[:, i])
        metrics[f"Output_{i}"] = {"MSE": mse, "MAE": mae, "R2": r2}
        print(f"Output_{i} -> MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # === Save predictions ===
    df = pd.DataFrame({
        "Sample": sample_names,
        **{f"Real_Output{i}": Y_test[:, i] for i in range(Y_test.shape[1])},
        **{f"Pred_Output{i}": predictions[:, i] for i in range(Y_test.shape[1])}
    })
    df.to_csv("predicciones_metrics.csv", index=False)

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv("metrics_summary.csv", index=True)

    print("\nPredicciones guardadas en 'predicciones_metrics.csv'")
    print("Resumen de metricas guardado en 'metrics_summary.csv'")
      
testing_models2(args.model_path, args.scalerX_path, args.dataX_path, args.dataY_path)
