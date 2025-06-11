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

srun python ~/git/French-Math-Data/process_data/get_data.py
srun python ~/git/French-Math-Data/create_datasets.py
srun python ~/git/French-Math-Data/infer.py --model Qwen3-8B --dataset Fused-CoT --task translation
srun python ~/git/French-Math-Data/infer.py --model Qwen3-8B --dataset Fused-CoT-FR --task math_fr
srun python ~/git/French-Math-Data/train.py --model Qwen3-8B --dataset Fused-CoT-FR-Solved
# TODO filter right and french solution / reformat
srun python ~/git/French-Math-Data/eval.py --models Qwen3-8B Qwen3-8B-SFT-Fused-CoT-FR-Solved --dataset Eval-Math-FR

