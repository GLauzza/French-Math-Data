import os

from transformers import AutoTokenizer


DATA_PATHS = ["./Data/", "./Data/"] # Local
# DATA_PATHS = [os.environ["DSDIR"] + "/HuggingFace/", os.environ["WORK"] + "/Datasets/"] # Jean Zay
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
