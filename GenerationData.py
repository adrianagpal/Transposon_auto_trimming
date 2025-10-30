from Bio import SeqIO
import random
import os
import multiprocessing
import argparse

# cadena de ADN aleatoria con la longitud deseada
def generate_string(length):
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(random.choices(nucleotides, k=length))

# formato FASTA
def generate_simulated_data(case, sequences, total_length=20000):
    min_length = 5000
    filtered_sequences = [seq for seq in sequences if len(seq) > min_length]

    if case == 1:
        seq1_choice = random.choice(sequences)
        min_starting_pos_1 = min(len(seq1_choice.seq), total_length // 2)
        starting_pos_1 = random.randint(min_starting_pos_1, total_length // 2)
        seq1_species = " ".join(seq1_choice.description.split(" ")[1:]).replace(" ", "_")
                
        seq2_choice = random.choice(sequences)
        min_starting_pos_2 = min(starting_pos_1 + len(seq1_choice.seq), total_length - len(seq1_choice.seq))
        starting_pos_2 = random.randint(min_starting_pos_2, total_length - int(total_length * 0.1))
        
        final_seq = generate_string(starting_pos_1) + str(seq1_choice.seq)
        final_seq += generate_string(starting_pos_2 - len(seq1_choice.seq))
        final_seq += str(seq2_choice.seq)

        if len(final_seq) < total_length:
            final_seq += generate_string(total_length - len(final_seq))

        final_seq = final_seq[:total_length]
        return f">Caso{case}_{seq1_choice.id}_{starting_pos_1}_{len(seq1_choice.seq)}_{seq1_species}\n{final_seq}"

    elif case == 2:
        if len(filtered_sequences) == 0:
            raise ValueError("No hay secuencias con longitud >5000 para generar el caso 2")

        len_obj = total_length // 3

        seq1_choice = random.choice(filtered_sequences)
        seq1_species = " ".join(seq1_choice.description.split(" ")[1:]).replace(" ", "_")
        seq1_seq = str(seq1_choice.seq)

        if len(seq1_seq) < len_obj:
            sequence_1 = seq1_seq + generate_string(len_obj - len(seq1_seq))
        else:
            sequence_1 = seq1_seq[:len_obj]
        
        seq2_choice = random.choice(filtered_sequences)
        seq2_seq = str(seq2_choice.seq)

        if len(seq2_seq) < len_obj:
            sequence_2 = seq2_seq + generate_string(len_obj - len(seq2_seq))
        else:
            sequence_2 = seq2_seq[:len_obj]

        final_seq = sequence_1 + sequence_2 + sequence_1

        if len(final_seq) < total_length:
            final_seq += generate_string(total_length - len(final_seq))

        final_seq = final_seq[:total_length]
        return f">Caso{case}_{seq1_choice.id}_0_{len(seq1_seq)}_{len_obj}_{len(seq2_seq)}_{seq1_species}\n{final_seq}"

    elif case == 3:
        monomer_length = random.randint(5, 100)
        monomer = generate_string(monomer_length)
        
        random_sequence = random.choice(sequences)
        seq_species = " ".join(random_sequence.description.split(" ")[1:]).replace(" ", "_")
        sequence_str = str(random_sequence.seq)
        sequence_length = len(sequence_str)
        
        start_pos = random.randint(int(total_length * 0.4), int(total_length * 0.6))
        
        microsatelite = (monomer * (start_pos // monomer_length)) + monomer[:start_pos % monomer_length]           
        microsatelite += sequence_str
        
        remaining = total_length - len(microsatelite)
        microsatelite += (monomer * (remaining // monomer_length)) + monomer[:remaining % monomer_length]
        microsatelite = microsatelite[:total_length]

        return f">Caso{case}_{random_sequence.id}_{start_pos}_{sequence_length}_{seq_species}\n{microsatelite}"

    elif case == 4: 
        if len(filtered_sequences) == 0:
            raise ValueError("No hay secuencias con longitud >5000 para generar el caso 4")
        
        sequence = random.choice(filtered_sequences)
        seq_species = " ".join(sequence.description.split(" ")[1:]).replace(" ", "_")
        seq4_seq = str(sequence.seq)

        return f">Caso{case}_{sequence.id}_0_{len(seq4_seq)}_{seq_species}\n{seq4_seq}"

# Output
def generation_multiprocessing(sequences, n, output_file): 
    results = [] 
    for case in range(1, 5):
        for _ in range(n): 
            results.append(generate_simulated_data(case, sequences)) 
            
        # write sequentially inside each process to avoid file conflicts 
        with open(output_file, "w") as simulated_data: 
            for r in results: 
                simulated_data.write(r + "\n")
       
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="Archivo FASTA de entrada")
    parser.add_argument("--seq_per_case", type=int, default=1000, help="Número de secuencias por caso a generar ")
    parser.add_argument("--processes", type=int, default=20, help="Número de procesos por nodo")
    args = parser.parse_args()

    if not os.path.isfile(args.fasta):
        raise FileNotFoundError(f"No se encontró {args.fasta}")

    with open(args.fasta, "r") as fasta_file:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))

    if not sequences:
        raise ValueError("No se encontraron secuencias en el FASTA")

    # Dividir el trabajo entre procesos
    total_to_generate = args.seq_per_case
    seqs_per_process = total_to_generate // args.processes
    processes = []

    for i in range(args.processes): 
        start_idx = i * seqs_per_process 
        end_idx = start_idx + seqs_per_process 
        
        if i == args.processes - 1: 
            end_idx = total_to_generate 
            
        output_file = f"simulated_data_part{i + 1}.fasta" 
        p = multiprocessing.Process(
            target=generation_multiprocessing, 
            args=(sequences, end_idx-start_idx, output_file)
        ) 
        processes.append(p) 
        p.start() 
        
    for p in processes: 
        p.join()

    with open("simulated_data_merged.fasta", "w") as outfile:
        for i in range(args.processes):
            part_file = f"simulated_data_part{i + 1}.fasta"
            with open(part_file, "r") as infile:
                outfile.write(infile.read())
            os.remove(part_file)  # elimina los archivos temporales

print("✅ Secuencias generadas correctamente.")
