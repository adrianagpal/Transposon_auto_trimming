import subprocess
import zipfile
import os

def download_genome(species_genome, zip_file, failure_log):
    try:
        # Primer intento con --reference
        print(f"Intentando descargar {species_genome} con --reference...")
        subprocess.run(
            ["datasets", "download", "genome", "taxon", species_genome, "--reference", "--filename", zip_file],
            check=True, capture_output=True, text=True
        )
        print(f"Descarga con --reference completada para {species_genome}.")

    except subprocess.CalledProcessError as e:
        print(f"Fallo con --reference. Reintentando sin --reference para {species_genome}...")
        try:
            # Segundo intento sin --reference
            subprocess.run(
                ["datasets", "download", "genome", "taxon", species_genome, "--filename", zip_file],
                check=True, capture_output=True, text=True
            )
            print(f"Descarga sin --reference completada para {species_genome}.")

        except subprocess.CalledProcessError:
            # Si tambien falla escribir en el log
            with open(failure_log, "a") as f:
                f.write(f"? Error: No se pudo descargar el genoma de {species_genome}\n")
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
        
        print(f"? Archivo {zip_file} descomprimido correctamente en {output_dir}")
        return True

    except zipfile.BadZipFile:
        print(f"Error: {zip_file} no es un archivo ZIP valido o esta corrupto.")
        return False

    except Exception as e:
        print(f"?? Error inesperado al descomprimir {zip_file}: {e}")
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

    te_fasta = "TE.fasta"

    # Ejecutar extractfasta.sh
    print(f"?? Extrayendo secuencia {header} de {input_fasta}...")
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
    
if __name__ == "__main__":
    
    species_genome = "Lineus longissimus"
    safe_name = species_genome.replace(" ", "_")
    zip_file = f"{safe_name}.zip"
    genome_dir = f"{safe_name}_genome"
    failure_log = "descargas_fallidas.log"

    header = ">Caso1_BEL-12_LiLo#CLASSI/LTR/BELPAO_9920_7608_Lineus_longissimus"
    input_fasta = "Drosophila_melanogaster2.fasta"
    output_dir = "results_Aphyosemion_australe"
   
    # Descargar
    resultado = download_genome(species_genome, zip_file, failure_log)

    if resultado:
        if unzip_genome(zip_file, genome_dir):
            # Buscar el .fna
            fna_path = find_fna_file(genome_dir, species_genome, failure_log)
            if fna_path:
                run_extract_and_teaid(header, input_fasta, fna_path, output_dir)
                print(f"Listo para procesar: {fna_path}")
        else:
            print(f"Descarga correcta, pero fallo al descomprimir {zip_file}")
    else:
        print(f"Fallo en la descarga de {species_genome}")
