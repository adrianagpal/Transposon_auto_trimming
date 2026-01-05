#!/bin/bash

#SBATCH -o resnet18_30000.out
#SBATCH -e resnet18_30000.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J resnet18_30000
#SBATCH --time 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 400GB

module load tensorflow-gpu/2.6.2

source /shared/home/sorozcoarias/anaconda3/bin/activate auto_trimming_agp

~/anaconda3/envs/training_agp/bin/python3 resnet18_30000.py test trained_model.h5 scalerX.bin X_test.npy Y_test.npy
