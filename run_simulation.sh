#!/bin/bash

#SBATCH -o simulation3.out
#SBATCH -e simulation3.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J simulation_old_dataset
#SBATCH --time 1-00:00:00
#SBATCH --partition gpu
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 200GB

module load tensorflow-gpu/2.6.2
source /shared/home/sorozcoarias/anaconda3/bin/activate gpu

cd /shared/home/sorozcoarias/tagua_gen_ec/TransposonDLToolkit/auto_curation_v2/auto_trimming_Adriana

~/anaconda3/envs/gpu/bin/python3 SimulationData.py --fasta pablo_dataset/dataset_peq.fasta