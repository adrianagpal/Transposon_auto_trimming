#! /bin/bash

# Comprobar que se han pasado los argumentos necesarios
if [ $# -ne 2 ]; then
    echo "Uso: $0 archivo_headers.fasta libreria.fasta"
    exit 1
fi

input_fasta=$1     # Archivo que contiene todas las secuencias (de donde extraer headers)
library=$2         # Librería de TE que se pasa a TE-Aid
output_dir="te-aid"

# Crear carpeta de salida si no existe
mkdir -p "$output_dir"

# Obtener directorio del script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Iterar por cada header en el FASTA
grep ">" "$input_fasta" | sed 's/\r$//' | while read -r header; do

    echo "Header: $header"

    # Mantener el '>' en el header
    outname=$(echo "$header" | sed 's/>//' | sed 's#[/\\]#_#g' | sed 's/#/_/g')

    # Extraer la secuencia en TE.fasta
    $DIR/extractfasta.sh H "$header" "$input_fasta" > TE.fasta

    # Comprobar que TE.fasta no esté vacío
    if [ ! -s TE.fasta ]; then
        echo "Secuencia no encontrada para $header"
        continue
    fi

    # Mostrar la secuencia en consola
    echo "Secuencia extraída:"
    cat TE.fasta

    # Ejecutar TE-Aid sobre la secuencia extraída
    $DIR/TE-Aid -q TE.fasta -g "$library"

    # Renombrar el PDF generado y moverlo a la carpeta de salida
    if [ -f TE.fasta.c2g.pdf ]; then
        mv TE.fasta.c2g.pdf "$output_dir/${outname}.pdf"
    fi

    # Limpiar archivos temporales de TE-Aid
    rm -f TE.fasta TE.fasta.fai blastn.txt
done


echo ""
echo "-------------"
echo "( FINISHED!!! )"
echo "-------------"