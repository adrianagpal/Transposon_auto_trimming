#!/bin/bash

#SBATCH -o agp_desc_gen.out
#SBATCH -e agp_desc_gen.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J agp_desc_gen
#SBATCH --time 1-00:00:00
#SBATCH --partition gpu
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 200GB

input_fasta=$1
genomes="genomes"
output_dir="te_aid"
failure_log="failure.log"

# Create files and folders
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

if [ ! -d "$genomes" ]; then
    mkdir -p "$genomes"
fi

if [ ! -f "$failure_log" ]; then
    touch "$failure_log"
fi

# Iterate each header in the input fasta file
grep ">" "$input_fasta" | sed 's/\r$//' | while read -r header; do
    echo "Header: $header"
    
    # Load the conda environment    
    source /shared/home/sorozcoarias/anaconda3/bin/activate MCHelper

    # Extract from the header the part after the last '#'
    after_hash=$(echo "$header" | awk -F'#' '{print $NF}')

    # Extract the species name from the header (generally the last two words)
    species=$(echo "$after_hash" | awk -F'_' '{if (NF>=2) print $(NF-1)"_"$NF}')
    
    # Extract from the header the part before '#'
    outname=$(echo "$header" | sed 's/>//' | sed 's#[/\\]#_#g' | cut -d'#' -f1)

    # Checks if species variable has a value
    if [[ -n "$species" ]]; then
        # Substitutes "_" by an space (needed to look for the genome)
        species_genome=$(echo "$species" | tr '_' ' ')
        echo "Procesando especie: $species_genome"
        echo "$species_genome" >> species.txt
    fi
    
    # Name of the .zip genome file    
    zip_file="${genomes}/${species_genome}_dataset.zip"

    # Checks if the genome has been downloaded before, if not it tries to download it
    if [[ -f "$zip_file" ]]; then
        echo "Ya existe dataset para $species, saltando!..."
    else
        # Try to download with --reference first
        if datasets download genome taxon "${species_genome}" --reference --filename "$zip_file"; then
            echo "Descarga con --reference completada para $species_genome."
        else
            echo "Fallo con --reference. Reintentando sin --reference..."
        
            if datasets download genome taxon "${species_genome}" --filename "$zip_file"; then
                echo "Descarga sin --reference completada para $species_genome."
            else
                echo "? Error: No se pudo descargar el genoma de $species_genome" >> "$failure_log"
                continue  # continues to the next fasta header
            fi
        fi
       
        # Unzip the genome file
        echo "Descomprimiendo dataset de $species..."
        unzip -o "$zip_file" -d "${genomes}/${species}/" >/dev/null 2>&1 || {
            echo "Error: Fallo al descomprimir $species" >> "$failure_log"
            continue
        } 
        
    fi
       
    # Search .fna file and save the path
    fna_file=$(find "${genomes}/${species}/" -type f -name "*.fna" -print -quit)
    
    if [[ -z "$fna_file" ]]; then
        echo "Error: No se encontró .fna para $species" >> "$failure_log"
        continue
    fi

    echo "? Genoma de $species descargado y descomprimido correctamente."
    
    # Load the conda environment
    source /shared/home/sorozcoarias/anaconda3/bin/activate te_aid
    
    # Extract sequence in TE.fasta
    "./extractfasta.sh" "H" "$header" "$input_fasta" > TE.fasta

    # Checks if TE.fasta is not empty
    if [ ! -s TE.fasta ]; then
        echo "Secuencia no encontrada para $header"
        continue
    fi
    
    # Shows sequence 
    echo "Secuencia extraída:"
    cat TE.fasta

    # Runs TE-Aid on the sequence and corresponding genome
    ./TE-Aid -q TE.fasta -g "$fna_file" -o "$output_dir"
    
    # Renames generated PDF file and moves it to output folder
    if [ -f $output_dir/TE.fasta.c2g.pdf ]; then
        mv $output_dir/TE.fasta.c2g.pdf "$output_dir/${outname}.pdf"
    fi
    
    # Clean up the genomes directory
    rm -rf "${genomes:?}"/*
    
done

echo "---- Proceso completado ----"
echo "Errores registrados en $failure_log"
