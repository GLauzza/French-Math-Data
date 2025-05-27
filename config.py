import os

from transformers import AutoTokenizer

# Local
# DATA_PATHS = ["./Datasets/", "./Datasets/"]
# MODEL_PATHS = ["./Models/", "./Models/"]
# TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Jean Zay
DATA_PATHS = [os.environ["DSDIR"] + "/HuggingFace/", os.environ["WORK"] + "/Datasets/"]
MODEL_PATHS = [os.environ["DSDIR"] + "/HuggingFace_Models/", os.environ["WORK"] + "/Models/"]
TOKENIZER =  AutoTokenizer.from_pretrained(os.environ["DSDIR"] + "/HuggingFace_Models/Qwen/Qwen3-8B")


models_to_evaluate = [
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen/Qwen3-8B",
    "OpenLLM-France/Lucie-7B-Instruct-v1.1",
    "microsoft/Phi-4-mini-reasoning",
    "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # "open-r1/OpenR1-Distill-7B",
    "HoangHa/Pensez-v0.1-e5",
    "meta-llama/Llama-3.1-8B-Instruct",
]