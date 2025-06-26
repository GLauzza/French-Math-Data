import os

from transformers import AutoTokenizer

# Local
# parent = os.path.dirname(os.path.realpath(__file__))
# DATA_PATHS = [parent+"/Datasets/", parent+"/Datasets/", parent+"/Datasets/"]
# MODEL_PATHS = [parent+"/Models/", parent+"/Models/", parent+"/Models/"]
# TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Jean Zay
DATA_PATHS = [os.environ["DSDIR"] + "/HuggingFace/", os.environ["SCRATCH"] + "/Datasets/", os.environ["WORK"] + "/Datasets/"]
MODEL_PATHS = [os.environ["DSDIR"] + "/HuggingFace_Models/", os.environ["SCRATCH"] + "/Models/", os.environ["WORK"] + "/Models/"]
TOKENIZER =  AutoTokenizer.from_pretrained(os.environ["DSDIR"] + "/HuggingFace_Models/Qwen/Qwen3-8B")
