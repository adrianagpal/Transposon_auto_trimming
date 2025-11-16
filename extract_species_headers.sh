#! /bin/bash
################################################################################
# extract_fasta_headers.sh
# Extrae los headers de un archivo FASTA y los guarda en headers.txt
# Uso: ./extract_fasta_headers.sh archivo.fasta
################################################################################

# Comprobar que se ha pasado un argumento
if [ $# -ne 1 ]; then
    echo "Uso: $0 archivo.fasta"
    exit 1
fi

input_fasta=$1
output_headers="species.txt"

# Extraer los headers, quedarnos con las dos últimas "palabras" separadas por "_",
# y reemplazar "_" por espacio
grep ">" "$input_fasta" \
  | sed 's/\r$//' \
  | awk -F'_' '{ print $(NF-1) " " $NF }' \
  > "$output_headers"

echo "Headers extraídos en: $output_headers"
