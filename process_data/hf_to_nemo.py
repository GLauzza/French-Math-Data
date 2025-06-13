import json
import sys
import os 
import shutil
import argparse

from tqdm import tqdm
import datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
from utils_data import *
from utils_model import *


def hf_to_nemo(dataset, output_path, chat_template_fun, tokenizer):
    print("FM - Converting Dataset to Nemo")
    os.makedirs(output_path, exist_ok=True) 
    with open(output_path + "/training.jsonl", "w") as f:
        for sample in tqdm(dataset):
            json.dump({
                "input": chat_template_fun(sample["question"]),
                "output": sample["solution"] + tokenizer.eos_token,
            }, f)
            f.write("\n")
    print("FM - Dataset Converted to Nemo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform a dataset from Huggingface format to Nemo finetuning format')
    parser.add_argument('--model', type=str, default="Qwen2.5-Math-7B-Instruct", help='Model to use for chat template')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to transform')
    args = parser.parse_args()

    model_path, chat_template_fun, _ = get_config(args.model)
    model, tokenizer = load_model(model_path)

    dataset = load_data(args.dataset)

    output_path = config.DATA_PATHS[2] + args.dataset + "-NEMO"
    
    hf_to_nemo(dataset, output_path, chat_template_fun, tokenizer)

    shutil.copytree(output_path, config.DATA_PATHS[1] + args.dataset + "-NEMO", dirs_exist_ok=True)