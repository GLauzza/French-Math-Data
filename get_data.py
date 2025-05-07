from huggingface_hub import snapshot_download
from datasets import load_dataset
import config


def get_huggingface_data(repo_id, allow_patterns):
    snapshot_download(
        repo_id=repo_id,
        local_dir=config.DATA_PATH+repo_id.split("/")[-1],
        repo_type="dataset",
        allow_patterns=allow_patterns
    )


def get_am_deepseek_distill():
    # get_huggingface_data("a-m-team/AM-DeepSeek-Distilled-40M", ["math_r1_*.jsonl"]) # ~40GB
    get_huggingface_data("a-m-team/AM-DeepSeek-Distilled-40M", ["math_r1_1pass.jsonl"]) # ~10GB


def get_big_math():
    get_huggingface_data("SynthLabsAI/Big-Math-RL-Verified", ["*.parquet"]) # ~32MB


def get_llama_nemotron():
    get_huggingface_data("nvidia/Llama-Nemotron-Post-Training-Dataset", ["SFT/math/math_v1.1.jsonl"]) # ~40GB


def get_megamath_web_pro():
    # get_huggingface_data("LLM360/MegaMath", ["megamath-web-pro/*.parquet"]) # ~50GB
    get_huggingface_data("LLM360/MegaMath", ["megamath-web-pro/000_00000.parquet"]) # ~600MB


def get_numinamath_1_5():
    # get_huggingface_data("AI-MO/NuminaMath-1.5", ["data/*.parquet"]) # ~600MB
    get_huggingface_data("AI-MO/NuminaMath-1.5", ["data/train-00000-of-00003.parquet"]) # ~200MB


def get_open_r1_math():
    # get_huggingface_data("open-r1/OpenR1-Math-220k", ["data/*.parquet"]) # ~2GB
    get_huggingface_data("open-r1/OpenR1-Math-220k", ["data/train-00000-of-00010.parquet"]) # ~200MB


def get_open_thoughts_2():
    # get_huggingface_data("open-thoughts/OpenThoughts2-1M", ["data/*.parquet"]) # ~8GB
    get_huggingface_data("open-thoughts/OpenThoughts2-1M", ["data/train-00000-of-00038.parquet"]) # ~200MB


def main():
    get_am_deepseek_distill()
    get_big_math()
    get_llama_nemotron()
    get_megamath_web_pro()
    get_numinamath_1_5()
    get_open_r1_math()
    get_open_thoughts_2()


if __name__ == "__main__":
    main()