#!/bin/bash
#SBATCH --job-name="Prepa Math"
#SBATCH --output=/lustre/fsn1/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --constraint=h100
#SBATCH --account=knb@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00

module purge
module load arch/h100
module load uv/0.8.3
source /lustre/fsn1/projects/rech/knb/ukq43aj/git/French-Math-Data/.venv/bin/activate

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export WANDB_MODE=offline
export UNSLOTH_COMPILE_LOCATION="/lustre/fsn1/projects/rech/knb/ukq43aj/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -x




#echo "SLURM: ----------------- CREATING COT / EVAL -----------------------"
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Eval-Math-FR
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Eval-Math-EN
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Fused-CoT --n -1

#echo "SLURM: --------------------- DEDUPLICATE ---------------------------"
#srun python ~/git/French-Math-Data/process_data/dedup.py --dataset Train-Math-en-v3 --model Qwen3-Embedding-8B --threshold 0.975 --n 1 --criterion already_sampled

#echo "SLURM: ---------------------- DIVERSIFY ----------------------------"
#srun python ~/git/French-Math-Data/diversify.py --dataset Fused-CoT-Topic-question --model Qwen3-Embedding-4B --action dist

#echo "SLURM: --------------------- CLASSIFYING ---------------------------"
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT --task quality --input solution --name Fused-CoT-quality --batch_size -1
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT --task topic --input question --name Fused-CoT-Topic-question --batch_size -1
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-Topic-question --task topic --input solution --name Fused-CoT-Topic --batch_size -1

#echo "SLURM: ---------------- TRANSLATING QUESTION -----------------------"
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-Topic --task translation_question --input question --name Fused-CoT-FR --batch_size -1
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Nemotron-Math-v2-FR --task translation_answer --input answer --name Nemotron-Math-v2-FR-answer --batch_size 128




#echo "SLURM: ----------------------- SOLVING -----------------------------"
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-FR --task math_fr --input question_fr --name Fused-CoT-FR-Solved --batch_size -1

#echo "SLURM: ---------------- CLASSIFYING LANGUAGE -----------------------"
#srun python ~/git/French-Math-Data/infer.py --model fasttext --dataset Fused-CoT-FR-Solved --task language_classification --input solution_fr_Qwen3-8B --name Raw-Train-Math-FR --batch_size 64

#echo "SLURM: ----------------- CREATING TRAIN DATA -----------------------"
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Train-Math-FR --model Qwen3-32B-FP8-dynamic

# OR

#echo "SLURM: ---------------- TRANSLATING SOLUTION -----------------------"
#srun python ~/git/French-Math-Data/infer.py --model Qwen3-32B-FP8-dynamic --dataset Fused-CoT-FR --task translation_solution --input solution --name Fused-CoT-FR-Solved --batch_size -1 --chunk_size 512

#echo "SLURM: ---------------- CLASSIFYING LANGUAGE -----------------------"
#srun python ~/git/French-Math-Data/infer.py --model fasttext --dataset Fused-CoT-FR-Solved --task language_classification --input solution_fr --name Raw-Train-Math-FR --batch_size 64

#echo "SLURM: ----------------- CREATING TRAIN DATA -----------------------"
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Train-Math-en --n 100000
#srun python ~/git/French-Math-Data/create_datasets.py --datasets Train-Math --fr_ratio 1
#srun python ~/git/French-Math-Data/process_data/hf_to_nemo.py --model Qwen2.5-7B-Instruct --dataset Train-Math-en-fr



#echo "SLURM: ------------------------ TRAIN -------------------------------"
#srun python ~/git/French-Math-Data/train.py --model Qwen3-4B-Thinking-2507 --dataset Train-Math-en-fr --id qwen3-en-fr-6

#echo "SLURM: --------------------- MERGE WANDB ----------------------------"
#srun python ~/git/French-Math-Data/merge_wandb.py --run_ids unsloth-5-epoch-en-fr-0 unsloth-5-epoch-en-fr-1 unsloth-5-epoch-en-fr-2 unsloth-5-epoch-en-fr-3 unsloth-5-epoch-en-fr-4 unsloth-5-epoch-en-fr-5 unslot>

#echo "SLURM: --------------------- EVALUATING ----------------------------"
#srun python ~/git/French-Math-Data/eval.py --model Qwen3-4B-Thinking-2507-SFT-3-nemorl-90 --dataset Eval-Math-FR --batch_size -1

#echo "SLURM: --------------------- PLOT EVAL ----------------------------"
#srun python ~/git/French-Math-Data/notebooks/plot_eval.py
