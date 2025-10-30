#!/bin/bash

#SBATCH -o generacion_datos.out
#SBATCH -e generacion_datos.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J agp_gen_datos
#SBATCH --time 1-00:00:00
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --mem 200GB

source /shared/home/sorozcoarias/anaconda3/bin/activate gpu

cd /shared/home/sorozcoarias/tagua_gen_ec/TransposonDLToolkit/auto_curation_v2/auto_trimming_Adriana

~/anaconda3/envs/gpu/bin/python3 GenerationData.py --fasta 1000species_dataset/r.1.5_all.fasta --seq_per_case 20 
