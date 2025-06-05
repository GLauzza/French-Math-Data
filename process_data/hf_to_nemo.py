import json
import sys
import os 

from tqdm import tqdm
import datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
from utils_data import *
from utils_model import *


def hf_to_nemo(dataset, output_path, chat_template_fun, tokenizer):
    os.makedirs(output_path, exist_ok=True) 
    with open(output_path + "/training.jsonl", "w") as f:
        for sample in tqdm(dataset):
            json.dump({
                "input": chat_template_fun(sample["question"]),
                "output": sample["solution"] + tokenizer.eos_token,
            }, f)
            f.write("\n")


if __name__ == "__main__":
    MODEL_NAME = "Qwen2.5-Math-7B-Instruct"
    DATASET_NAME = "Fused-CoT"

    print("FM - Getting Config")
    model_path, chat_template_fun, _ = get_config(MODEL_NAME)
    model, tokenizer = load_model(model_path)

    dataset =  load_data(DATASET_NAME)

    print("FM - Converting Dataset to Nemo")
    output_path = config.DATA_PATHS[1] + DATASET_NAME + "_NEMO"
    hf_to_nemo(dataset, output_path, chat_template_fun, tokenizer)
    print("FM - Dataset Converted to Nemo")