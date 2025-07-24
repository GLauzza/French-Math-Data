#!/bin/bash
#SBATCH --job-name=math
#SBATCH --output=[YOUR PATH TO OUTPUT LOGS]
#SBATCH --error=[YOUR PATH TO ERROR LOGS]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --constraint=a100
#SBATCH --account=[YOUR ACCOUNT]
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=02:00:00

module purge
module load miniforge/24.11.3
conda activate vllmath

set -x

echo "SLURM: ---------------- TRANSLATING QUESTION -----------------------"
srun python ~/git/French-Math-Data/infer.py --model [YOUR MODEL] --dataset [YOUR DATASET] --task translation_question --input question --name [YOUR DATASET]-FR-question --batch_size -1

echo "SLURM: ---------------- TRANSLATING ANSWER -----------------------"
srun python ~/git/French-Math-Data/infer.py --model [YOUR MODEL] --dataset [YOUR DATASET]-FR-question --task translation_answer --input question --name [YOUR DATASET]-FR --batch_size -1

echo "SLURM: ---------------- CLASSIFYING LANGUAGE -----------------------"
srun python ~/git/French-Math-Data/infer.py --model fasttext --dataset [YOUR DATASET]-FR --task language_classification --input question_fr --name Raw-Train-Math-FR --batch_size 64

echo "SLURM: ----------------- CREATING TRAIN DATA -----------------------"
srun python ~/git/French-Math-Data/create_datasets.py --datasets Train-Math-FR --rl
