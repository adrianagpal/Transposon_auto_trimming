#!/bin/bash

#SBATCH -o agp_species.out
#SBATCH -e agp_species.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J agp_species
#SBATCH --time 1-00:00:00
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem 200GB

source /shared/home/sorozcoarias/anaconda3/bin/activate auto_trimming_agp

~/anaconda3/envs/auto_trimming_agp/bin/python3 get_species_np.py 
