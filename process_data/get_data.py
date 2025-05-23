import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from huggingface_hub import snapshot_download
from datasets import load_dataset

import config


def get_huggingface_data(repo_id, allow_patterns):
    snapshot_download(
        repo_id=repo_id,
        local_dir=config.DATA_PATHS[1]+repo_id.split("/")[-1],
        repo_type="dataset",
        allow_patterns=allow_patterns
    )


def get_am_deepseek_distill():
    # get_huggingface_data("a-m-team/AM-DeepSeek-Distilled-40M", ["math_r1_*.jsonl"]) # ~40GB
    get_huggingface_data("a-m-team/AM-DeepSeek-Distilled-40M", ["math_r1_1pass.jsonl"]) # ~10GB


def get_big_math():
    get_huggingface_data("SynthLabsAI/Big-Math-RL-Verified", ["*.parquet"]) # ~30MB


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


def get_megamath_web_pro():
    # get_huggingface_data("LLM360/MegaMath", ["megamath-web-pro/*.parquet"]) # ~50GB
    get_huggingface_data("LLM360/MegaMath", ["megamath-web-pro/000_00000.parquet"]) # ~600MB


def get_metamath_qa():
    get_huggingface_data("meta-math/MetaMathQA", ["*.json"]) # ~400MB


def get_mgsm():
    os.system("mkdir " + config.DATA_PATHS[1] + "/MGSM")
    os.system("wget https://raw.githubusercontent.com/google-research/url-nlp/refs/heads/main/mgsm/mgsm_fr.tsv -O ./Data/MGSM/mgsm_fr.tsv") # ~60KB


def get_msvamp():
    get_huggingface_data("Mathoctopus/MSVAMP", ["test_French.json"]) # ~400KB


def get_numinamath_1_5():
    get_huggingface_data("AI-MO/NuminaMath-1.5", ["data/*.parquet"]) # ~600MB


def get_open_r1_math():
    # get_huggingface_data("open-r1/OpenR1-Math-220k", ["*.parquet"]) # ~8GB
    get_huggingface_data("open-r1/OpenR1-Math-220k", ["data/train-00000-of-00010.parquet"]) # ~200MB


def get_open_thoughts_2():
    # get_huggingface_data("open-thoughts/OpenThoughts2-1M", ["data/*.parquet"]) # ~8GB
    get_huggingface_data("open-thoughts/OpenThoughts2-1M", ["data/train-00000-of-00038.parquet"]) # ~200MB


def get_pensez():
    get_huggingface_data("HoangHa/Pensez-v0.1", ["*.parquet"]) # ~20MB


def get_poly_math():
    get_huggingface_data("Qwen/PolyMath", ["fr/*.parquet"]) # ~120KB


def get_s1k_1_1():
    get_huggingface_data("simplescaling/s1K-1.1", ["data/*.parquet"]) # ~20MB


def get_swallowmath():
    # get_huggingface_data("tokyotech-llm/swallow-math", ["*.jsonl"]) # ~13GB
    get_huggingface_data("tokyotech-llm/swallow-math", ["train-00002-of-00002.jsonl"]) # ~3GB


def main():
    # get_am_deepseek_distill()
    get_big_math()
    get_limo()
    get_limr()
    # get_llama_nemotron()
    get_math_lvl5_fr()
    get_mclm() 
    # get_megamath_web_pro()
    get_metamath_qa()
    get_mgsm() 
    get_msvamp()
    get_numinamath_1_5()
    # get_open_r1_math()
    # get_open_thoughts_2()
    get_pensez()
    get_poly_math()
    get_s1k_1_1()
    # get_swallowmath()


if __name__ == "__main__":
    main()