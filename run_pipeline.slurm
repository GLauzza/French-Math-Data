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
#SBATCH --time=02:00:00
##SBATCH --mail-user=gabriel.lauzzana@loria.fr
##SBATCH --mail-type=ALL

module purge
module load arch/a100
module load miniforge/24.11.3
conda activate vllmath

set -x

#echo "SLURM: ----------------- CREATING COT / EVAL -----------------------"
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Fused-CoT Eval-Math-FR

#echo "SLURM: ------------------ CLASSIFYING TOPIC ------------------------"
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT --task topic --input question --name Fused-CoT-Topic-question --batch_size -1
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-Topic-question --task topic --input solution --name Fused-CoT-Topic --batch_size -1

# TODO FILTERING / BALANCING

echo "SLURM: ---------------- TRANSLATING QUESTION -----------------------"
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-Topic --task translation_question --input question --name Fused-CoT-FR --batch_size -1
srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT --task translation_question --input question --name Fused-CoT-FR --batch_size -1


#echo "SLURM: ----------------------- SOLVING -----------------------------"
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-FR --task math_fr --input question_fr --name Fused-CoT-FR-Solved --batch_size -1

#echo "SLURM: ---------------- CLASSIFYING LANGUAGE -----------------------"
#srun python ~/git/French-Math-Data/infer.py --model fasttext --dataset Fused-CoT-FR-Solved --task language_classification --input solution_fr_Qwen3-8B --name Raw-Train-Math-FR --batch_size 64

#echo "SLURM: ----------------- CREATING TRAIN DATA -----------------------"
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Train-Math-FR --model Qwen3-32B-FP8-dynamic

# OR

echo "SLURM: ---------------- TRANSLATING SOLUTION -----------------------"
srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-FR --task translation_solution --input solution --name Fused-CoT-FR-Solved --batch_size -1 --chunk_size 512

echo "SLURM: ---------------- CLASSIFYING LANGUAGE -----------------------"
srun python ~/git/French-Math-Data/infer.py --model fasttext --dataset Fused-CoT-FR-Solved --task language_classification --input solution_fr --name Raw-Train-Math-FR --batch_size 64

echo "SLURM: ----------------- CREATING TRAIN DATA -----------------------"
srun python ~/git/French-Math-Data/create_datasets.py --datasets Train-Math-FR
#srun python ~/git/French-Math-Data/process_data/hf_to_nemo.py --model Qwen2.5-Math-7B-Instruct --dataset Train-Math-FR

#echo "SLURM: --------------------- EVALUATING ----------------------------"
#srun python ~/git/French-Math-Data/eval.py --models Qwen2.5-Math-7B-Instruct Qwen2.5-Math-7B-Instruct-SFT --dataset Eval-Math-FR --batch_size -1
