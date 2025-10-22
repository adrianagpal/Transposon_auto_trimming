from Bio import SeqIO
import random
import os

try:

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  
    ruta_fasta = filedialog.askopenfilename(
        title="Selecciona un archivo FASTA", 
        filetypes=[("FASTA files", "*.fasta"), ("All files", "*.*")])

except Exception:
    import argparse

    parser = argparse.ArgumentParser(description="Selecciona un archivo FASTA para procesar.")
    parser.add_argument("--fasta", required=True, help="Ruta al archivo FASTA")
    args = parser.parse_args()

    ruta_fasta = args.fasta

if not ruta_fasta or not os.path.isfile(ruta_fasta):
    print("No se seleccionó ningún archivo. Saliendo...")
    exit()

# cadena de ADN aleatoria con la longitud deseada
def generar_cadena(longitud):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choices(bases, k=longitud))

# formato FASTA
def datos_entrenamiento(caso):
    total_length = 20000
    sequences = []
    
    with open(ruta_fasta, "r") as fasta_file:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))

    if not sequences:
        print("Error: No se encontraron secuencias en el archivo FASTA.")
        exit()

    min_length = 5000
    filtered_sequences = [seq for seq in sequences if len(seq) > min_length]

    if caso == 1:
        seq1_choice = random.choice(sequences)
        min_starting_pos_1 = min(len(seq1_choice.seq), total_length // 2)
        starting_pos_1 = random.randint(min_starting_pos_1, total_length // 2)
                
        seq2_choice = random.choice(sequences)
        min_starting_pos_2 = min(starting_pos_1 + len(seq1_choice.seq), total_length - len(seq1_choice.seq))
        starting_pos_2 = random.randint(min_starting_pos_2, total_length - int(total_length * 0.1))
        
        final_seq = generar_cadena(starting_pos_1) + str(seq1_choice.seq)
        final_seq += generar_cadena(starting_pos_2 - len(seq1_choice.seq))
        final_seq += str(seq2_choice.seq)

        if len(final_seq) < total_length:
            final_seq += generar_cadena(total_length - len(final_seq))

        final_seq = final_seq[:total_length]
        return f">Caso{caso}_{seq1_choice.description}_{starting_pos_1}_{len(seq1_choice.seq)}\n{final_seq}"

    elif caso == 2:
        if len(filtered_sequences) == 0:
            raise ValueError("No hay secuencias con longitud >5000 para generar el caso 2")

        len_obj = total_length // 3

        seq1_choice = random.choice(filtered_sequences)
        seq1_seq = str(seq1_choice.seq)

        if len(seq1_seq) < len_obj:
            sequence_1 = seq1_seq + generar_cadena(len_obj - len(seq1_seq))
        else:
            sequence_1 = seq1_seq[:len_obj]
        
        seq2_choice = random.choice(filtered_sequences)
        seq2_seq = str(seq2_choice.seq)

        if len(seq2_seq) < len_obj:
            sequence_2 = seq2_seq + generar_cadena(len_obj - len(seq2_seq))
        else:
            sequence_2 = seq2_seq[:len_obj]

        final_seq = sequence_1 + sequence_2 + sequence_1

        if len(final_seq) < total_length:
            final_seq += generar_cadena(total_length - len(final_seq))

        final_seq = final_seq[:total_length]
        return f">Caso{caso}_{seq1_choice.description}_{len(seq1_seq)}_{len(seq2_seq)}\n{final_seq}"

    elif caso == 3:
        monomer_length = random.randint(5, 100)
        monomer = generar_cadena(monomer_length)
        
        random_sequence = random.choice(sequences)
        sequence_str = str(random_sequence.seq)
        sequence_length = len(sequence_str)
        
        start_pos = random.randint(int(total_length * 0.4), int(total_length * 0.6))
        
        microsatelite = (monomer * (start_pos // monomer_length)) + monomer[:start_pos % monomer_length]           
        microsatelite += sequence_str
        
        remaining = total_length - len(microsatelite)
        microsatelite += (monomer * (remaining // monomer_length)) + monomer[:remaining % monomer_length]
        microsatelite = microsatelite[:total_length]

        return f">Caso{caso}_{random_sequence.description}_{start_pos}_{sequence_length}\n{microsatelite}"

    elif caso == 4: 
        if len(filtered_sequences) == 0:
            raise ValueError("No hay secuencias con longitud >5000 para generar el caso 4")
        
        sequence = random.choice(filtered_sequences)
        seq4_seq = str(sequence.seq)

        return f">Caso{caso}_{sequence.description}_0_{len(seq4_seq)}\n{seq4_seq}"

# Output
nombre_salida = "output.fasta"
with open(nombre_salida, "w") as archivo_salida:
    for caso in range(1, 5):
        for _ in range(3):
            archivo_salida.write(datos_entrenamiento(caso) + "\n")

print(f"Archivo generado: {nombre_salida}")