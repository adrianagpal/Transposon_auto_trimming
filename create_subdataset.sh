#!/bin/bash

#SBATCH -o agp_dataset.out
#SBATCH -e agp_dataset.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J agp_dataset
#SBATCH --partition long
#SBATCH --time 1-00:00:00
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem 200GB

source /shared/home/sorozcoarias/anaconda3/bin/activate auto_trimming_agp

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/auto_trimming_agp/bin/python3 prueba3.py 
