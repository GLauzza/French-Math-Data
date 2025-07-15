import sys
import os
import shutil

from huggingface_hub import snapshot_download
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config


def get_huggingface_data(repo_id, allow_patterns):
    print("FM - Downloading", repo_id)
    output_path = config.DATA_PATHS[1] + repo_id.split("/")[-1]
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_path,
        repo_type="dataset",
        allow_patterns=allow_patterns
    )
    shutil.copytree(output_path, config.DATA_PATHS[2] + repo_id.split("/")[-1], dirs_exist_ok=True)
    print("FM - Downloaded", repo_id)


def get_am_deepseek_distill():
    get_huggingface_data("a-m-team/AM-DeepSeek-Distilled-40M", ["math_r1_*.jsonl"]) # ~40GB


def get_big_math():
    get_huggingface_data("SynthLabsAI/Big-Math-RL-Verified", ["*.parquet"]) # ~30MB

def get_deepmath():
    get_huggingface_data("zwhe99/DeepMath-103K", ["*.parquet"]) # ~2GB

def get_limo():
    get_huggingface_data("GAIR/LIMO", ["*.jsonl"]) # ~20MB


def get_limr():
    get_huggingface_data("GAIR/LIMR", ["*.json"]) # ~480KB


def get_llama_nemotron():
    get_huggingface_data("nvidia/Llama-Nemotron-Post-Training-Dataset", ["SFT/math/math_v1.1.jsonl"]) # ~40GB


def get_math_lvl5_fr():
    get_huggingface_data("le-leadboard/MATH_LVL5_fr", ["*.json"]) # ~3MB


def get_mclm():
    get_huggingface_data("amphora/MCLM", ["*.parquet"]) # ~2MB


def get_metamath_qa():
    get_huggingface_data("meta-math/MetaMathQA", ["*.json"]) # ~400MB


def get_mgsm():
    os.system("mkdir " + config.DATA_PATHS[1] + "MGSM")
    os.system("wget https://raw.githubusercontent.com/google-research/url-nlp/refs/heads/main/mgsm/mgsm_fr.tsv -O " + config.DATA_PATHS[1] + "MGSM/mgsm_fr.tsv") # ~60KB
    os.system("mkdir " + config.DATA_PATHS[2] + "MGSM")
    os.system("wget https://raw.githubusercontent.com/google-research/url-nlp/refs/heads/main/mgsm/mgsm_fr.tsv -O " + config.DATA_PATHS[2] + "MGSM/mgsm_fr.tsv") # ~60KB


def get_msvamp():
    get_huggingface_data("Mathoctopus/MSVAMP", ["test_French.json"]) # ~400KB


def get_numinamath_1_5():
    get_huggingface_data("AI-MO/NuminaMath-1.5", ["data/*.parquet"]) # ~600MB


def get_open_r1_math():
    get_huggingface_data("open-r1/OpenR1-Math-220k", ["all/*.parquet"]) # ~8GB


def get_open_thoughts_2():
    get_huggingface_data("open-thoughts/OpenThoughts2-1M", ["data/*.parquet"]) # ~8GB


def get_pensez():
    get_huggingface_data("HoangHa/Pensez-v0.1", ["*.parquet"]) # ~20MB


def get_poly_math():
    get_huggingface_data("Qwen/PolyMath", ["fr/*.parquet"]) # ~120KB


def get_s1k_1_1():
    get_huggingface_data("simplescaling/s1K-1.1", ["data/*.parquet"]) # ~20MB


if __name__ == "__main__":
    # get_am_deepseek_distill() # Already on JZ
    get_big_math()
    get_deepmath()
    get_limo()
    get_limr()
    # get_llama_nemotron() # Already on JZ
    get_math_lvl5_fr()
    get_mclm() 
    get_metamath_qa()
    get_mgsm() 
    get_msvamp()
    get_numinamath_1_5()
    # get_open_r1_math() # Already on JZ
    # get_open_thoughts_2() # Already on JZ
    get_pensez()
    get_poly_math()
    get_s1k_1_1()