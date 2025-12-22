from Bio import SeqIO
import re, os, subprocess, zipfile, multiprocessing, sys, shutil
import argparse
from autotrim import generation_multiprocessing, create_dataset
from model import ResNet18, auto_trimming, NDStandardScaler, testing_model
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
   
   
def load_data(fasta_path, output_dir="./te_aid", dataset_dir="./dataset_autotrim"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    generation_multiprocessing(fasta_path, 20, output_dir)
    
    print("Procesamiento completo.")  
    
    if os.path.exists(os.path.abspath("genomes")):
        shutil.rmtree(os.path.abspath("genomes"))    
    
    os.makedirs(dataset_dir, exist_ok=True)   
    
    cmd = f"""
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate auto_trimming_agp
    python -c "from autotrim import create_dataset; create_dataset(r'{fasta_path}', r'{output_dir}', r'{dataset_dir}')"
    """

    result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    
    print(result.stdout)
    print(result.stderr)

def get_model(input_size=(256, 256, 1), classes=128):

    # Ramas por canal
    tf.keras.backend.clear_session()
    cnn_div = ResNet18.build(input_size, classes)
    cnn_cov = ResNet18.build(input_size, classes)
    cnn_dot = ResNet18.build(input_size, classes)
    cnn_str = ResNet18.build(input_size, classes)
                      
    # Final model
    model = auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str)
    print(model.summary())
    
    return(model)

def run_experiment(model, train_ds, dev_ds, num_epochs):

    # Crear carpeta "checkpoint" si no existe
    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filepath = os.path.join(checkpoint_dir, "model.weights.h5")

    # Reduce LR si no mejora val_loss
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.01,
        patience=10,
        verbose=1
    )

    # Guardar mejores pesos segun val_r2_score
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_r2_score",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )

    # Entrenamiento
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=num_epochs,
        callbacks=[checkpoint_callback, lr_scheduler, early_stopping],
        verbose=1
    )

    return history, checkpoint_callback

    
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Modo de ejecucion")
    parser.add_argument("--input_fasta", required=True, help="Archivo FASTA de libreria")
    parser.add_argument("--processes", type=int, default=20, help="Numero de procesos paralelos")
    parser.add_argument("--output_dir", default="te-aid", help="Directorio de salida")
    parser.add_argument("--dataset_dir", default="dataset_autotrim", help="Directorio del dataset")
    args = parser.parse_args()
    
    db_dir = os.path.abspath("db")

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    if args.mode == "train":
    
        load_data(args.input_fasta, args.output_dir, args.dataset_dir)

        batch_size = 16        # IMPORTANTE
        num_epochs = 50
        input_size=(256, 256, 1)
        classes=128
    
        x_str = os.path.join(args.dataset_dir, "features_data.npy")
        y_str = os.path.join(args.dataset_dir, "labels_data.npy")
    
        # Cargar SOLO con mmap
        x = np.load(x_str, mmap_mode="r")
        y = np.load(y_str, mmap_mode="r")
    
        print(f"Loaded X shape: {x.shape}")
        print(f"Loaded Y shape: {y.shape}")
    
        # Split por INDICES (sin copiar datos)
        indices = np.arange(len(y))
        train_idx, test_dev_idx = train_test_split(indices, test_size=0.2, random_state=7)
        dev_idx, test_idx = train_test_split(test_dev_idx, test_size=0.5, random_state=7)  

        #Guardar scaler
        X_train_for_scaler = np.stack([
            (x[i].astype(np.float32) / 255.0) for i in train_idx
        ])
            
        scalerX = NDStandardScaler().fit(X_train_for_scaler)
        scalerX.save_model("scalerX")
            
        # Dataset generator (CLAVE)
        def make_dataset(indices, shuffle=False):
            def gen():
                for i in indices:
                    xi = x[i].astype(np.float32) / 255.0
                    xi = scalerX.transform(xi[np.newaxis, ...])[0].astype(np.float16)
                    yield (
                        xi[..., 0:1],
                        xi[..., 1:2],
                        xi[..., 2:3],
                        xi[..., 3:4],
                    ), y[i]
    
            return tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    (
                        tf.TensorSpec(input_size, tf.float16),
                        tf.TensorSpec(input_size, tf.float16),
                        tf.TensorSpec(input_size, tf.float16),
                        tf.TensorSpec(input_size, tf.float16),
                    ),
                    tf.TensorSpec((2,), tf.float32),
                )
            ).shuffle(512 if shuffle else 1)\
             .batch(batch_size)\
             .prefetch(tf.data.AUTOTUNE)
    
        train_ds = make_dataset(train_idx, shuffle=True)
        dev_ds   = make_dataset(dev_idx)
        test_ds  = make_dataset(test_idx)
    
        # Modelo (IGUAL QUE ANTES)
        tf.keras.backend.clear_session()
    
        model = get_model(input_size, classes)

        # Entrenamiento
        history, checkpoints = run_experiment(
            model,
            train_ds,
            dev_ds,
            num_epochs
        )

        # Guardar metricas
        for key in ['r2_score', 'val_r2_score', 'r2_starting', 'r2_ending', 'loss', 'val_loss']:
            if key in history.history:
                np.save(f"{key}.npy", history.history[key])
    
        # Plots
        plt.figure()
        for label in ['r2_score', 'val_r2_score', 'r2_starting', 'r2_ending']:
            if label in history.history:
                plt.plot(history.history[label], label=label)
        plt.xlabel('Epoch'); plt.ylabel('R2 Score'); plt.title('Epoch vs R2 Score')
        plt.legend(loc='lower right')
        plt.savefig('Train_Curve_R2_extended.png', bbox_inches='tight', dpi=500)
        plt.close()
    
        plt.figure()
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Epoch vs Loss')
        plt.legend(loc='upper right')
        plt.savefig('Train_Curve_Loss.png', bbox_inches='tight', dpi=500)
        plt.close()            
        
        # Guardar modelo entrenado
        model.save('trained_model.h5')
                       
        X_test = []
        Y_test = []
            
        for i in test_idx:
            xi = x[i].astype(np.float32) / 255.0
            xi = scalerX.transform(xi[np.newaxis, ...])[0].astype(np.float16)
            
            X_test.append((
                xi[..., 0:1],
                xi[..., 1:2],
                xi[..., 2:3],
                xi[..., 3:4],
            ))
            Y_test.append(y[i])
            
        # Separar canales
        ch0, ch1, ch2, ch3 = zip(*X_test)
        X_test = (
            np.stack(ch0, axis=0),
            np.stack(ch1, axis=0),
            np.stack(ch2, axis=0),
            np.stack(ch3, axis=0),
        )
            
        # Convertir Y_test a array
        Y_test = np.stack(Y_test)
            
        # Comprobar shapes
        for ch in X_test:
            print(ch.shape, ch.dtype)
        print("Y_test shape:", Y_test.shape, Y_test.dtype)
            
        # Guardar arrays
        np.save("X_test.npy", X_test, allow_pickle=True)
        np.save("Y_test.npy", Y_test)
    
    if args.mode == "test":
    
        testing_model("trained_model.h5", "X_test.npy", "Y_test.npy")
