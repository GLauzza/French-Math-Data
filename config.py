import os

from transformers import AutoTokenizer


# Local
DATA_PATHS = ["./Data/", "./Data/"]
MODEL_PATHS = ["./Models/", "./Models/"]
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Jean Zay
# DATA_PATHS = [os.environ["DSDIR"] + "/HuggingFace/", os.environ["WORK"] + "/Datasets/"]
# MODEL_PATHS = [os.environ["DSDIR"] + "/HuggingFace_Models/", os.environ["WORK"] + "/Models/"]
# TOKENIZER = AutoTokenizer.from_pretrained(os.environ["DSDIR"] + "/HuggingFace/Qwen/Qwen3-8B")
