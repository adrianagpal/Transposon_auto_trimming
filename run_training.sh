#!/bin/bash

#SBATCH -o training.out
#SBATCH -e training.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J training
#SBATCH --time 1-00:00:00
#SBATCH --partition gpu
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 200GB

#module load tensorflow-gpu/2.6.2
source /shared/home/sorozcoarias/anaconda3/bin/activate gpu

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /shared/home/sorozcoarias/tagua_gen_ec/TransposonDLToolkit/auto_curation_v2/auto_trimming_Adriana

~/anaconda3/envs/gpu/bin/python3 NN_trainingV2.py train pablo_dataset/features_data.npy pablo_dataset/labels_data.npy