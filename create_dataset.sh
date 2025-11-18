#!/bin/bash

#SBATCH -o agp_dataset.out
#SBATCH -e agp_dataset.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J agp_dataset
#SBATCH --time 1-00:00:00
#SBATCH --partition gpu
#SBATCH --nodelist=gpu-node-03
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 200GB

module load tensorflow-gpu/2.6.2

export TF_GPU_ALLOCATOR=cuda_malloc_async

source /shared/home/sorozcoarias/anaconda3/bin/activate auto_trimming_agp

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/auto_trimming_agp/bin/python3 create_dataset_cases.py dataset ./pdf_generation/TEAid/simulated_data_80.fasta ./pdf_generation/TEAid ./dataset_casos123 

