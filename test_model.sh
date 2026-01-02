#!/bin/bash

#SBATCH -o test_model.out
#SBATCH -e test_model.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J test_model
#SBATCH --partition long
#SBATCH --time 1-00:00:00
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --mem 200GB

module load tensorflow-gpu/2.6.2

export TF_GPU_ALLOCATOR=cuda_malloc_async

source /shared/home/sorozcoarias/anaconda3/bin/activate training_agp

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/training_agp/bin/python3 test_model.py ./dataset_30000/trained_model.h5 ./dataset_30000/scalerX.bin ./dataset_predictions
