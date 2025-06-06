import os

from transformers import AutoTokenizer

# Local
# DATA_PATHS = ["./Datasets/", "./Datasets/", "./Datasets/"]
# MODEL_PATHS = ["./Models/", "./Models/", "./Models/"]
# TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Jean Zay
DATA_PATHS = [os.environ["DSDIR"] + "/HuggingFace/", os.environ["WORK"] + "/Datasets/", os.environ["SCRATCH"] + "/Datasets/"]
MODEL_PATHS = [os.environ["DSDIR"] + "/HuggingFace_Models/", os.environ["WORK"] + "/Models/", os.environ["SCRATCH"] + "/Models/"]
TOKENIZER =  AutoTokenizer.from_pretrained(os.environ["DSDIR"] + "/HuggingFace_Models/Qwen/Qwen3-8B")
