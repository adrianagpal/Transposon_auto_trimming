import os, subprocess
import argparse
from dataset_library import generation_multiprocessing
from model_library import ResNet18, auto_trimming, NDStandardScaler, testing_model, plot_training_metrics, test_model, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
   
   
def load_data(fasta_path, dataset_dir="./dataset_autotrim"):
    
    # Transform sequences from fasta path into TEAid .pdfs
    generation_multiprocessing(fasta_path)
    
    print("Procesamiento completo.")  

    # Create directory to save the dataset    
    os.makedirs(dataset_dir, exist_ok=True)   
    
    # Create dataset
    cmd = f"""
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate auto_trimming_agp
    python -c "from dataset_library import create_dataset; create_dataset(r'{fasta_path}', r'{dataset_dir}')"
    """

    result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    
    print(result.stdout)
    print(result.stderr)

def get_model(input_size=(256, 256, 1), num_classes=128):

    # Ramas por canal
    tf.keras.backend.clear_session()
    cnn_div = ResNet18.build(input_size, num_classes)
    cnn_cov = ResNet18.build(input_size, num_classes)
    cnn_dot = ResNet18.build(input_size, num_classes)
    cnn_str = ResNet18.build(input_size, num_classes)
                      
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
        patience=20,
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

# ====================
# MAIN
# ====================   
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "trimming"], required=True, help="Modo de ejecucion")
    parser.add_argument("--input_fasta", required=True, help="Archivo FASTA de libreria")
    parser.add_argument("--processes", type=int, default=20, help="Numero de procesos paralelos")
    parser.add_argument("--output_dir", default="te_aid", help="Directorio de salida")
    parser.add_argument("--dataset_dir", default="dataset_30000", help="Directorio del dataset")
    args = parser.parse_args()
    
    if args.mode == "train":
    
        #load_data(args.input_fasta, args.dataset_dir)

        batch_size = 16
        num_epochs = 200
        input_size=(256, 256, 1)
        classes=128
    
        x_str = os.path.join(args.dataset_dir, "features_data.npy")
        y_str = os.path.join(args.dataset_dir, "labels_data.npy")
    
        # Load data using NumPy memory mapping (not loading the full dataset)
        x = np.load(x_str, mmap_mode="r")
        y = np.load(y_str, mmap_mode="r")
    
        print(f"Loaded X shape: {x.shape}")
        print(f"Loaded Y shape: {y.shape}")
    
        # Divide data to create subdatasets for training, test and validation, and save indices
        indices = np.arange(len(y))
        train_idx, test_dev_idx = train_test_split(indices, test_size=0.2, random_state=7)
        dev_idx, test_idx = train_test_split(test_dev_idx, test_size=0.5, random_state=7)  

        # Save scaler
        X_train_for_scaler = np.stack([(x[i].astype(np.float32) / 255.0) for i in train_idx])            
        scalerX = NDStandardScaler().fit(X_train_for_scaler)
        scalerX.save_model("scalerX")
            
        # TensorFlow dataset generator
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
    
        # Create datasets for training, validation and test
        train_ds = make_dataset(train_idx, shuffle=True)
        dev_ds   = make_dataset(dev_idx)
        test_ds  = make_dataset(test_idx)
    
        # Model for training
        tf.keras.backend.clear_session()    
        model = get_model(input_size, classes)

        # Fit model on training data
        history, checkpoints = run_experiment(
            model,
            train_ds,
            dev_ds,
            num_epochs
        )
   
        # Plots
        plot_training_metrics(history)           
        
        # Guardar modelo entrenado
        model.save('trained_model.h5')

        # Save data for testing            
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
            
        # Save arrays
        np.save("X_test.npy", X_test, allow_pickle=True)
        np.save("Y_test.npy", Y_test)
    
    if args.mode == "test":
    
        dataset_dir = "./dataset_predictions"
        
        model = tf.keras.models.load_model(
          "./models/resnet18_dropout_nokernel/trained_model.h5",
          custom_objects={"r2_score": r2_score},
          compile = False
        )
    
        predictions = test_model("./models/resnet18_dropout_nokernel/trained_model.h5", "./models/resnet18_dropout_nokernel/scalerX.bin", dataset_dir)
        
        tf.keras.utils.plot_model(
        model,
        to_file='model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )

        print(predictions)

    if args.mode == "trimming":

        dataset_dir = "dataset_autotrim"
        output_fasta = "curated_seq.txt"
        #load_data(args.input_fasta, dataset_dir)

        dataX_path = os.path.join(dataset_dir, "features_data.npy")
        dataY_path = os.path.join(dataset_dir, "labels_data.npy")
        TE_ids = np.load(os.path.join(dataset_dir, "case_labels.npy"), allow_pickle=True)

        predictions = test_model("./models/resnet18_dropout_nokernel/trained_model.h5", "./models/resnet18_dropout_nokernel/scalerX.bin", dataset_dir)

        print(predictions)
        # Cargar secuencias originales
        sequences = list(SeqIO.parse(args.input_fasta, "fasta"))

        cut_records = []

        TE_size = 15000
        for i, pred in enumerate(predictions):
            TE_id = TE_ids[i]
            print({TE_id})
            start = int(pred[0] * TE_size)
            print({start})
            end = int(pred[1] * TE_size)
            print({end})

            # Get the SeqRecord with matching id
            record = next((rec for rec in sequences if rec.id.startswith(TE_id)), None)

            for rec in sequences[:5]:
                print(rec.id)
                print(rec.description)

            if record is None:
                print(f"TE_id {TE_id} not found in sequences")
                continue

            curated_seq = record.seq[start:end]

            new_record = SeqRecord(
                Seq(curated_seq),
                id=TE_id,
                description=f"cut from {start} to {end}"
            )
            cut_records.append(new_record)

        print(f"Saving {len(cut_records)} cut sequences to {output_fasta}...")
        SeqIO.write(cut_records, output_fasta, "fasta")
        print("Done!")
