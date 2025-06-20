#!/bin/bash
#SBATCH --job-name=math
#SBATCH --output=/linkhome/rech/genlor01/ukq43aj/logs/log_%j.out
#SBATCH --error=/linkhome/rech/genlor01/ukq43aj/logs/log_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --constraint=a100
#SBATCH --account=knb@a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:10:00

module purge
module load arch/a100
module load miniforge/24.11.3
conda activate vllmath

set -x

echo "SLURM: -------------------- GETTING DATA ---------------------------"
srun python ./process_data/get_data.py

echo "SLURM: ----------------- CREATING COT / EVAL -----------------------"
srun python ./create_datasets.py --datasets Fused-CoT Eval-Math-FR

echo "SLURM: ---------------- TRANSLATING QUESTION -----------------------"
srun python ./infer.py --model Qwen3-32B --dataset Fused-CoT --task translation --input question --name Fused-CoT-FR --batch-size 64




# echo "SLURM: ----------------------- SOLVING -----------------------------"
# srun python ./infer.py --model Qwen3-32B --dataset Fused-CoT-FR --task math_fr --input question_fr --name Fused-CoT-FR-Solved --batch-size 64

# echo "SLURM: ---------------- CLASSIFYING LANGUAGE -----------------------"
# srun python ./infer.py --model fasttext --dataset Fused-CoT-FR-Solved --task language_classification --input solution_fr_Qwen3-8B --name Raw-Train-Math-FR --batch-size 64

# echo "SLURM: ----------------- CREATING TRAIN DATA -----------------------"
# srun python ./create_datasets.py --datasets Train-Math-FR --model Qwen3-32B

# OR

echo "SLURM: ---------------- TRANSLATING SOLUTION -----------------------"
srun python ./infer.py --model Qwen3-32B --dataset Fused-CoT-FR --task translation --input solution --name Fused-CoT-FR-Solved --batch-size 64

echo "SLURM: ---------------- CLASSIFYING LANGUAGE -----------------------"
srun python ./infer.py --model fasttext --dataset Fused-CoT-FR-Solved --task language_classification --input solution_fr --name Raw-Train-Math-FR --batch-size 64

echo "SLURM: ----------------- CREATING TRAIN DATA -----------------------"
srun python ./create_datasets.py --datasets Train-Math-FR



echo "SLURM: ---------------------- TRAINING -----------------------------"
srun python ./train.py --model Qwen3-8B --dataset Train-Math-FR
# srun accelerate launch --config_file ./deepspeed.yaml ./train.py --model Qwen3-8B --dataset Train-Math-FR

echo "SLURM: --------------------- EVALUATING ----------------------------"
srun python ./eval.py --models Qwen3-8B --dataset Eval-Math-FR


