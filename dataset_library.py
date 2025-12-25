# -*- coding: utf-8 -*-
import subprocess
import zipfile
import os
import re
import multiprocessing
import shutil
from Bio import SeqIO
import fitz
import numpy as np
import cv2

# Function that downloads a specific genome
def download_genome(species_genome, zip_file, env_name="MC_helper_agp"):
    try:
        # First try with --reference
        print(f"Trying downloading {species_genome} with --reference...")

        subprocess.run(["conda", "run", "-n", env_name, "datasets", "download", 
            "genome", "taxon", species_genome, "--reference", "--filename", zip_file],
            check=True, capture_output=True, text=True
        )

        print(f"Download with --reference completed for {species_genome}.")

    except subprocess.CalledProcessError as e:
        try:

            # Second try without --reference
            print(f"Error with --reference. Retrying without --reference for {species_genome}...")

            subprocess.run(["conda", "run", "-n", env_name, "datasets", "download", 
                "genome", "taxon", species_genome, "--filename", zip_file],
                check=True, capture_output=True, text=True
            )

            print(f"Download without --reference completed for {species_genome}.")

        except subprocess.CalledProcessError:
            # If also fails, it will write it on the log file
            with open(failure_log, "a") as f:
                f.write(f"Error: Download not possible for {species_genome}\n")
            print(f"Error: Download not possible for {species_genome}. Registered in {failure_log}.")
            print(f"STDERR:\n{e.stderr}")
            return False

    return True

# Function that unzips a genome .zip file 
def unzip_genome(zip_file, genome_dir):

    if not os.path.exists(zip_file):
        print(f"Error: File {zip_file} was not found")
        return False

    try:
        # Create output folder it if doesnt exist
        os.makedirs(genome_dir, exist_ok=True)
        
        # Unzip the file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(genome_dir)
        
        print(f"File {zip_file} was unzipped in {genome_dir}")
        return True

    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid .zip file or it is corrupted.")
        return False

    except Exception as e:
        print(f"Unexpected error while unzipping {zip_file}: {e}")
        return False

# Function that saves the path of the .fna genome file
def find_fna_file(genome_dir, species_name):

    fna_file = None

    if not os.path.exists(genome_dir):
        print(f"Error: genome directory {genome_dir} does not exist")
        return None

    # Search recursively the first .fna file
    for root, dirs, files in os.walk(genome_dir):
        for file in files:
            if file.endswith(".fna"):
                fna_file = os.path.join(root, file)
                break
        if fna_file:
            break

    if fna_file:
        print(f"Genome of {species_name} downloaded and unzipped.")
        print(f"File .fna found in: {fna_file}")
        return fna_file

    else:
        with open(failure_log, "a") as f:
            f.write(f"Error: File .fna for {species_name} couldnt be found.\n")
        print(f"Error: File .fna for {species_name} couldnt be found. Registered in {failure_log}.")
        return None

# Function to get fasta header and run TE-Aid
def run_extract_and_teaid(header, te_fasta, fna_file, output_dir, env_name="te_aid"):

    if not os.path.isfile("./TE-Aid"):
        print("TE-Aid no encontrado o no ejecutable")
        return False

    if not os.path.isfile(fna_file):
        print(f"Archivo genomico no encontrado: {fna_file}")
        return False

    # Verify if TE fasta is not empty and exists
    if not os.path.exists(te_fasta) or os.path.getsize(te_fasta) == 0:
        print(f"Secuencia no encontrada para {header}")
        return False

    print("Secuencia extraida con exito:")

    # Execute TE-Aid
    print(f"Ejecutando TE-Aid con {te_fasta} y {fna_file}...")
    try:
        result = subprocess.run(
            f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env_name} && ./TE-Aid -q {os.path.abspath(te_fasta)} -g {os.path.abspath(fna_file)} -o {os.path.abspath(output_dir)}",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )

        print(f"TE-Aid ejecutado correctamente. Resultados en: {output_dir}")

    except subprocess.CalledProcessError:
        print(f"Error inesperado al ejecutar TE-Aid: {e}")
        return False

    return True

# Create species dictionary with indexes     
def create_species_dict_from_fasta(input_fasta):
    species_dict = {}

    # Get headers of sequences
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    headers = [f">{sequence.description}" for sequence in sequences]

    for line_num, header in enumerate(headers, start=0):
        match = re.search(r'([A-Z][a-z]+_[a-z]+)', header)

        if match:
            species_name = match.group(1)
            species_dict.setdefault(species_name, []).append(line_num)

    print(f"Se detectaron {len(species_dict)} especies unicas en {input_fasta}")
    return species_dict

def process_species(species, sequences, positions, headers, output_dir="./te_aid", genomes_dir="./genomes", env_name="MC_helper_agp"):

    os.makedirs(genomes_dir, exist_ok=True)

    print(f"Especie detectada: {species}")
    species_safe = species.replace("_", " ")
    genome_dir = os.path.abspath(os.path.join(genomes_dir, f"{species}_genome"))
    zip_file = os.path.abspath(os.path.join(genomes_dir, f"{species}.zip"))   

    print(f"headers: {headers}")

    for position in positions:
        try:
            header = headers[position]
            match_case = re.match(r'^>?([^#\s]+)', header)
            case_name = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())

            # Create output directory for the case
            case_dir = os.path.join(output_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)
            
            new_pdf = os.path.join(output_dir, f"{case_name}.pdf")

            # TE FASTA file 
            te_fasta = os.path.join(case_dir, f"{case_name}.fasta")

            print(f"genome_dir: {genome_dir}")
            print(f"case_dir: {case_dir}")
            print(f"case_name: {case_name}")
            print(f"header: {header}")
            print(f"position: {position}")
            print(f"sequences[position].seq: {sequences[position].seq}")

            with open(te_fasta, "w") as f:
                f.write(f"{header}\n{sequences[position].seq}\n")

            # Check if pdf exists
            if os.path.exists(new_pdf):
                print(f"PDF ya existe: {new_pdf}")
                shutil.rmtree(case_dir)
                continue

            if not os.path.exists(zip_file):
                download_genome(species_safe, zip_file)
            
            if not os.path.exists(genome_dir):
                unzip_genome(zip_file, genome_dir)

            fna_file = find_fna_file(genome_dir, species)
            if not fna_file:
                print(f"No se encontro archivo .fna para {species}")
                continue

            # Execute TE-Aid
            run_extract_and_teaid(header, te_fasta, fna_file, case_dir, env_name="te_aid")

            original_pdf = os.path.join(case_dir, f"{case_name}.fasta.c2g.pdf")

            if os.path.exists(original_pdf):
                os.rename(original_pdf, new_pdf)
                shutil.rmtree(case_dir)
                print(f"PDF renombrado como: {new_pdf}")
            else:
                print(f"No se encontro el PDF esperado: {original_pdf}")

        except Exception as e:
            print(f"ERROR procesando especie {species}: {e}")

    # Removing genome and case 
    if genome_dir and os.path.exists(genome_dir) and new_pdf and os.path.exists(new_pdf):
        shutil.rmtree(genome_dir)
        if zip_file and os.path.exists(zip_file):
            os.remove(zip_file)
        print(f"Genoma {species} eliminado para liberar espacio.")

def generation_multiprocessing(input_fasta, n_processes=20, output_dir="./te_aid", genomes_dir="./genomes"):

    os.makedirs(genomes_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get headers of sequences
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    headers = [f">{sequence.description}" for sequence in sequences]

    total = len(headers)
    print(f"Se encontraron {total} secuencias en {input_fasta}")
    
    species_dict = create_species_dict_from_fasta(input_fasta)
    print(f"Se detectaron {len(species_dict)} especies en total.")
    
    print(f"{species_dict}")

    # Crear procesos
    processes = []
    for species, positions in species_dict.items():
        p = multiprocessing.Process(
            target=process_species,
            args=(species, sequences, positions, headers, output_dir, genomes_dir)
        )
        processes.append(p)

    # Ejecutar procesos en batches de n_processes
    for i in range(0, len(processes), n_processes):
        batch = processes[i:i + n_processes]
        
        print(f"Iniciando batch {i // n_processes + 1} de {(len(processes) + n_processes - 1) // n_processes} "
              f"({len(batch)} procesos en paralelo)...")

        for p in batch:
            p.start()

        for p in batch:
            p.join()

        print(f"Batch {i // n_processes + 1} completado.\n")

    print("Procesamiento completo.") 

    if os.path.exists(genomes_dir):
        shutil.rmtree(genomes_dir)     

    if os.path.exists("db"):
        shutil.rmtree("db")     

# Function to count number of good and bad pdfs generated by TEAid
def count_good_pdfs(teaid_dir):
    pdf_files = [f for f in os.listdir(teaid_dir) if f.endswith(".pdf")]

    count_good = 0
    count_bad = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(teaid_dir, pdf_file)
        try:
            # Verifies PDF file size is > 4 KB
            if os.path.getsize(pdf_path) <= 4 * 1024:
                count_bad += 1
                continue

            # Tries to open the PDF
            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                count_good += 1
            else:
                count_bad += 1

        except Exception as e:
            print(f"Error abriendo {pdf_path}: {e}")
            count_bad += 1

    return count_good, count_bad
    
def create_dataset(input_fasta, output_dir, teaid_dir="./te_aid", TE_size=15000):
    # Reads fasta file and return a list of SeqRecord objects with attributes like TE.id and TE.seq
    TEs = list(SeqIO.parse(input_fasta, "fasta"))
    
    good_pdfs, _ = count_good_pdfs(teaid_dir)
    
    # Create a matrix to store images data
    feature_data = np.zeros((good_pdfs, 256, 256, 4), dtype=np.uint8)
    
    # Create matrices to store labels, case and species data
    labels = np.zeros((good_pdfs, 2), dtype=np.float32)
    case_names = []
    species_names = []

    n = 0
    for TE in TEs:

        TE_name = TE.id.split("#")[0]
        case_id = TE.id.split("_")[0]
        species_match = re.search(r'([A-Z][a-z]+_[a-z]+)$', TE.id)
        species_name = species_match.group(0) if species_match else None
        
        pdf_path = os.path.join(teaid_dir, TE_name + '.pdf') 
        
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) <= 4 * 1024:
            print(f"PDF no encontrado para {TE_name}, se omite.")
            continue
                            
        print(f"Doing TE_name: {TE_name}")
        try:
                
            doc = fitz.open(pdf_path)
            image_path = os.path.join(teaid_dir, TE_name + ".fa.c2g.jpeg")

            # Saving pages in jpeg format
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(200/72,200/72))
                pix.save(image_path, 'JPEG')

            # Carga la imagen desde el archivo
            te_aid_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if te_aid_image is None:
                print("ERROR: No se pudo abrir la imagen de", TE_name)
                continue  # pasa al siguiente TE
                
            # Divide la imagen en distintos plots y les hace resize
            # Guarda la informacion de los plots en la matriz feature_data
            feature_data[n, :, :, 0] = cv2.resize(te_aid_image[150:1030, 150:1130], (256, 256)) # divergence_plot
            feature_data[n, :, :, 1] = cv2.resize(te_aid_image[150:1030, 1340:2320], (256, 256)) # coverage_plot
            feature_data[n, :, :, 2] = cv2.resize(te_aid_image[1340:2220, 150:1130], (256, 256)) # selfdot_plot
            feature_data[n, :, :, 3] = cv2.resize(te_aid_image[1340:2220, 1340:2320], (256, 256)) # structure_plot
            
            # Guarda el penultimo valor del TE.id (posicion de inicio)
            start_pos = int(TE.id.split("_")[-4])

            # Guarda el ultimo valor del TE.id (longitud del TE)
            TE_len = int(TE.id.split("_")[-3])

            # Guarda en las etiquetas la posicion de inicio y la de final, relativa a la longitud total
            labels[n, 0] = start_pos / TE_size
            labels[n, 1] = min((start_pos + TE_len) / TE_size, 1)
                       
            case_names.append(TE_name)
            species_names.append(species_name)
            
            print(f"n: {n}")
            n += 1

        except Exception as ex:
            print(f"Something wrong with {TE_name}: {ex}")

    np.save(output_dir + "/features_data.npy", feature_data[:n])
    np.save(output_dir + "/labels_data.npy", labels[:n])
    np.save(output_dir + "/case_labels.npy", np.array(case_names))
    np.save(output_dir + "/species_labels.npy", np.array(species_names))
