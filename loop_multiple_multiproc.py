# -*- coding: utf-8 -*-
import subprocess
import zipfile
import os
import re
import argparse
import multiprocessing
from functools import partial
import shutil

def download_genome(species_genome, zip_file, failure_log):
    try:
        # Primer intento con --reference
        print(f"Intentando descargar {species_genome} con --reference...")
        subprocess.run(["conda", "run", "-n", "MCHelper", "datasets", "download", 
            "genome", "taxon", species_genome, "--reference", "--filename", zip_file],
            check=True, capture_output=True, text=True
        )
        print(f"Descarga con --reference completada para {species_genome}.")

    except subprocess.CalledProcessError as e:
        print(f"Fallo con --reference. Reintentando sin --reference para {species_genome}...")
        try:
            # Segundo intento sin --reference
            subprocess.run(["conda", "run", "-n", "MCHelper", "datasets", "download", 
                "genome", "taxon", species_genome, "--filename", zip_file],
                check=True, capture_output=True, text=True
            )
            print(f"Descarga sin --reference completada para {species_genome}.")

        except subprocess.CalledProcessError:
            # Si tambien falla escribir en el log
            with open(failure_log, "a") as f:
                f.write(f"Error: No se pudo descargar el genoma de {species_genome}\n")
            print(f"? Error: No se pudo descargar el genoma de {species_genome}. Registrado en {failure_log}.")
            return False

    return True
    
def unzip_genome(zip_file, output_dir):

    if not os.path.exists(zip_file):
        print(f"Error: No se encontro el archivo {zip_file}")
        return False

    try:
        # Crear carpeta de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Descomprimir el archivo
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print(f"Archivo {zip_file} descomprimido correctamente en {output_dir}")
        return True

    except zipfile.BadZipFile:
        print(f"Error: {zip_file} no es un archivo ZIP valido o esta corrupto.")
        return False

    except Exception as e:
        print(f"Error inesperado al descomprimir {zip_file}: {e}")
        return False
        
def find_fna_file(genome_dir, species_name, failure_log):

    fna_file = None

    # Buscar recursivamente el primer archivo .fna
    for root, dirs, files in os.walk(genome_dir):
        for file in files:
            if file.endswith(".fna"):
                fna_file = os.path.join(root, file)
                break
        if fna_file:
            break

    if fna_file:
        print(f"Genoma de {species_name} descargado y descomprimido correctamente.")
        print(f"Archivo .fna encontrado en: {fna_file}")
        return fna_file
    else:
        with open(failure_log, "a") as f:
            f.write(f"Error: No se encontro archivo .fna para {species_name}\n")
        print(f"Error: No se encontro archivo .fna para {species_name}. Registrado en {failure_log}.")
        return None
        
def run_extract_and_teaid(header, input_fasta, fna_file, output_dir, env_name="te_aid"):

    match_case = re.match(r'^>?([^#\s]+)', header)
    nombre_caso = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())

    te_fasta = f"{nombre_caso}.fasta"

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

    # Verificar si TE.fasta no est vaco
    if not os.path.exists(te_fasta) or os.path.getsize(te_fasta) == 0:
        print(f"Secuencia no encontrada para {header}")
        return False

    print("Secuencia extraida con exito:")
    with open(te_fasta) as f:
        print(f.read())

    # Ejecutar TE-Aid
    print(f"Ejecutando TE-Aid con {te_fasta} y {fna_file}...")

    os.makedirs(output_dir, exist_ok=True)

    try:

        result = subprocess.run(
            f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env_name} && ./TE-Aid -q {os.path.abspath(te_fasta)} -g {os.path.abspath(fna_file)} -o {os.path.abspath(output_dir)}",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )

        print(result.stdout)
    
        if result.returncode != 0:
            print(f"Error en TE-Aid:\n{result.stderr}")
            return False
    
        print(f"TE-Aid ejecutado correctamente. Resultados en: {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error inesperado al ejecutar TE-Aid: {e}")
        return False

def process_header(header, input_fasta, output_dir, failure_log, genomes_dir="./genomes"):
    try:
        # Extraer especie del header
        match = re.search(r'_([A-Z][a-z]+_[a-z]+)$', header)
        if not match:
            print(f"{multiprocessing.current_process().name} - No se encontro especie en header: {header}")
            return

        species = match.group(1).replace("_", " ")
        print(f"{multiprocessing.current_process().name} - Especie detectada: {species}")

        # Construcción de rutas
        species_genome = species
        safe_name = species_genome.replace(" ", "_")
        zip_file = os.path.abspath(os.path.join(genomes_dir, f"{safe_name}.zip"))
        genome_dir = os.path.abspath(os.path.join(genomes_dir, f"{safe_name}_genome"))

        # Crear subdirectorio de salida para la especie
        species_output_dir = os.path.join(output_dir, safe_name)
        os.makedirs(species_output_dir, exist_ok=True)

        # Descargar y procesar
        if not download_genome(species_genome, zip_file, failure_log):
            print(f"{multiprocessing.current_process().name} - Fallo en la descarga de {species_genome}")
            return

        if not unzip_genome(zip_file, genome_dir):
            print(f"{multiprocessing.current_process().name} - Fallo al descomprimir {zip_file}")
            return

        fna_path = find_fna_file(genome_dir, species_genome, failure_log)
        if not fna_path:
            print(f"{multiprocessing.current_process().name} - No se encontro archivo .fna para {species_genome}")
            return

        # Ejecutar TE-Aid en la carpeta de salida de la especie
        run_extract_and_teaid(header, input_fasta, fna_path, species_output_dir)
        
        # Renombrar PDF generado por TE-Aid
        match_case = re.match(r'^>?([^#\s]+)', header)
        nombre_caso = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())
        pdf_original = os.path.join(species_output_dir, f"{nombre_caso}.fasta.c2g.pdf")  # o .c2g.pdf según versión
        pdf_nuevo = os.path.join(species_output_dir, f"{nombre_caso}.pdf")

        if os.path.exists(pdf_original):
            os.rename(pdf_original, pdf_nuevo)
            print(f"PDF renombrado como: {pdf_nuevo}")
            
            # Eliminar genoma para liberar espacio
            if os.path.exists(genome_dir):
                shutil.rmtree(genome_dir)
                print(f"Genoma {species_genome} eliminado para liberar espacio.")
        else:
            print(f"No se encontro el PDF esperado: {pdf_original}")

    except Exception as e:
        print(f"{multiprocessing.current_process().name} - ERROR procesando {header}: {e}")

# Output
def generation_multiprocessing(input_fasta, n_processes, output_dir, failure_log):
    # Leer las cabeceras del archivo FASTA
    with open(input_fasta, "r") as fasta:
        headers = [line.strip() for line in fasta if line.startswith(">")]

    total = len(headers)
    print(f"Se encontraron {total} secuencias en {input_fasta}")

    # Crear procesos
    processes = []
    for i, header in enumerate(headers):
        p = multiprocessing.Process(
            target=process_header,
            args=(header, input_fasta, output_dir, failure_log)
        )
        processes.append(p)

    # Ejecutar procesos en batches de n_processes
    for i in range(0, len(processes), n_processes):
        batch = processes[i:i + n_processes]
        for p in batch:
            p.start()
        for p in batch:
            p.join()

def process_chunk(headers_chunk, input_fasta, tmp_output_dir):
    for header in headers_chunk:
        process_header(header, input_fasta, tmp_output_dir)            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True, help="Archivo FASTA de libreria")
    parser.add_argument("--output_dir", required=True, help="Archivo FASTA de libreria")
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
