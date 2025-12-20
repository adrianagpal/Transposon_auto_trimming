#!/bin/bash

#SBATCH -o data_gen.out
#SBATCH -e data_gen.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J data_gen
#SBATCH --time 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 200GB

source /shared/home/sorozcoarias/anaconda3/bin/activate agp_gpu

~/anaconda3/envs/agp_gpu/bin/python3 GenerationData.py --fasta r.1.5_all.fasta --seq_per_case 10

