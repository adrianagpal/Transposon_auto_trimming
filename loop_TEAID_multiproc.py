# -*- coding: utf-8 -*-
import subprocess
import zipfile
import os
import re
import argparse
import multiprocessing
from functools import partial
import shutil

# Function that downloads a specific genome
# It should work with MC_helper_agp/MCHelper envs
def download_genome(species_genome, zip_file, failure_log):
    try:
        # First try with --reference
        print(f"Intentando descargar {species_genome} con --reference...")
        subprocess.run(["conda", "run", "-n", "MC_helper_agp", "datasets", "download", 
            "genome", "taxon", species_genome, "--reference", "--filename", zip_file],
            check=True, capture_output=True, text=True
        )
        print(f"Download with --reference completed for {species_genome}.")

    except subprocess.CalledProcessError as e:
        print(f"Fallo con --reference. Reintentando sin --reference para {species_genome}...")
        try:
            # Second try without --reference
            subprocess.run(["conda", "run", "-n", "MC_helper_agp", "datasets", "download", 
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
            print("STDOUT:\n{e.stdout}")
            return False

    return True

# Function that unzips a genome .zip file 
def unzip_genome(zip_file, output_dir):

    if not os.path.exists(zip_file):
        print(f"Error: File {zip_file} was not found")
        return False

    try:
        # Create output folder it if doesnt exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Unzip the file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print(f"File {zip_file} was unzipped in {output_dir}")
        return True

    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid .zip file or it is corrupted.")
        return False

    except Exception as e:
        print(f"Unexpected error while unzipping {zip_file}: {e}")
        return False

# Function that saves the path of the .fna genome file
def find_fna_file(genome_dir, species_name, failure_log):

    fna_file = None

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
def run_extract_and_teaid(header, input_fasta, fna_file, output_base_dir, env_name="te_aid"):
    # Extraer nombre del caso
    match_case = re.match(r'^>?([^#\s]+)', header)
    nombre_caso = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())

    # Crear carpeta de salida específica para el caso
    case_dir = output_base_dir
    os.makedirs(case_dir, exist_ok=True)

    # Archivo FASTA de la TE
    te_fasta = os.path.join(case_dir, f"{nombre_caso}.fasta")

    # Ejecutar extractfasta.sh
    print(f"Extrayendo secuencia {header} de {input_fasta}...")
    try:
        subprocess.run(
            ["./extractfasta.sh", "H", header, input_fasta],
            check=True,
            stdout=open(te_fasta, "w"),
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar extractfasta.sh: {e.stderr}")
        return False

    # Verificar si TE.fasta no está vacío
    if not os.path.exists(te_fasta) or os.path.getsize(te_fasta) == 0:
        print(f"Secuencia no encontrada para {header}")
        return False

    print("Secuencia extraida con exito:")
    with open(te_fasta) as f:
        print(f.read())

    # Ejecutar TE-Aid
    print(f"Ejecutando TE-Aid con {te_fasta} y {fna_file}...")
    try:
        result = subprocess.run(
            f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env_name} && ./TE-Aid -q {os.path.abspath(te_fasta)} -g {os.path.abspath(fna_file)} -o {os.path.abspath(case_dir)}",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"Error en TE-Aid:\n{result.stderr}")
            return False

        print(f"TE-Aid ejecutado correctamente. Resultados en: {case_dir}")
        return True

    except Exception as e:
        print(f"Error inesperado al ejecutar TE-Aid: {e}")
        return False
       
def create_species_dict_from_fasta(input_fasta):
    species_dict = {}

    with open(input_fasta, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line.startswith(">"):
                # Extraer especie del header, ejemplo: Genus_species
                match = re.search(r'([A-Z][a-z]+_[a-z]+)', line)
                if match:
                    species_name = match.group(1).replace("_", " ")
                    species_dict.setdefault(species_name, []).append(line_num)

    print(f"Se detectaron {len(species_dict)} especies unicas en {input_fasta}")
    return species_dict

def process_species(species, positions, headers, input_fasta, output_dir, failure_log, genomes_dir="./genomes"):

    for position in positions:
        try:
            header = headers[(position - 1) // 2]

            match = re.search(r'([A-Z][a-z]+_[a-z]+)', header)
            if not match:
                print(f"No se encontro especie en header: {header}")
                continue

            species_name = match.group(1).replace("_", " ")
            print(f"Especie detectada: {species_name}")

            if species_name != species:
                print(f"Especie del header no coincide con {species}")
                continue

            species_genome = species_name
            match_case = re.match(r'^>?([^#\s]+)', header)
            nombre_caso = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())

            safe_name = species_genome.replace(" ", "_")
            zip_file = os.path.abspath(os.path.join(genomes_dir, f"{safe_name}.zip"))
            genome_dir = os.path.abspath(os.path.join(genomes_dir, f"{safe_name}_genome"))
            species_output_dir = os.path.join(output_dir, nombre_caso)
            os.makedirs(species_output_dir, exist_ok=True)

            pdf_nuevo = os.path.join(output_dir, f"{nombre_caso}.pdf")

            print(f"Genome_directory: {genomes_dir}")
            print(f"Species: {species_genome}")

            if os.path.exists(pdf_nuevo):
                print(f"PDF ya existe: {pdf_nuevo}")
                continue

            # Comprobar si existe el genoma
            if os.path.exists(genome_dir):
                print(f"Genoma ya descomprimido: {genome_dir}")
            else:
                if os.path.exists(zip_file):
                    print(f"ZIP del genoma ya existe: {zip_file}")
                else:
                    print(f"Descargando genoma de {species_genome}...")
                    if not download_genome(species_genome, zip_file, failure_log):
                        print(f"Fallo en la descarga de {species_genome}")
                        continue

                # Descomprimir solo si falta la carpeta
                print("Descomprimiendo genoma...")
                if not unzip_genome(zip_file, genome_dir):
                    print(f"Fallo al descomprimir {zip_file}")
                    continue

            # Buscar .fna
            fna_path = find_fna_file(genome_dir, species_genome, failure_log)
            if not fna_path:
                print(f"No se encontro archivo .fna para {species_genome}")
                return

            # Ejecutar TE-Aid
            run_extract_and_teaid(header, input_fasta, fna_path, species_output_dir)

            pdf_original = os.path.join(species_output_dir, f"{nombre_caso}.fasta.c2g.pdf")

            if os.path.exists(pdf_original):
                os.rename(pdf_original, pdf_nuevo)
                shutil.rmtree(species_output_dir)
                print(f"PDF renombrado como: {pdf_nuevo}")
            else:
                print(f"No se encontro el PDF esperado: {pdf_original}")

        except Exception as e:
            print(f"ERROR procesando especie {species}: {e}")


    # Limpieza final del genoma si todo fue bien
    if genome_dir and os.path.exists(genome_dir) and pdf_nuevo and os.path.exists(pdf_nuevo):
        shutil.rmtree(genome_dir)
        if zip_file and os.path.exists(zip_file):
            os.remove(zip_file)
        print(f"Genoma {species_genome} eliminado para liberar espacio.")

# Output
def generation_multiprocessing(input_fasta, n_processes, output_dir, failure_log, genomes_dir="./genomes"):
    # Leer las cabeceras del archivo FASTA
    with open(input_fasta, "r") as fasta:
        headers = [line.strip() for line in fasta if line.startswith(">")]

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
            args=(species, positions, headers, input_fasta, output_dir, failure_log, genomes_dir)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True, help="Archivo FASTA de libreria")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida")
    parser.add_argument("--processes", type=int, default=20, help="Numero de procesos paralelos")
    parser.add_argument("--failure_log", default="descargas_fallidas.log", help="Archivo para guardar fallos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generation_multiprocessing(
        args.input_fasta,
        args.processes,
        args.output_dir,
        args.failure_log
    )
