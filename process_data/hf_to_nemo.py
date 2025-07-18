import json
import sys
import os 
import shutil
import argparse

from tqdm import tqdm
import datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
from prepare_data import *
from utils_model import *


def hf_to_nemo(dataset, output_path, chat_template_fun, tokenizer):
    print("FM - Converting Dataset to Nemo")
    os.makedirs(output_path, exist_ok=True) 
    with open(output_path + "/training.jsonl", "w", encoding="utf-8") as f:
        for sample in tqdm(dataset):
            f.write(json.dumps({
                "input": chat_template_fun(sample["question"]),
                "output": sample["solution"] + tokenizer.eos_token,
            }) + "\n")
    with open(output_path + "/validation.jsonl", "w", encoding="utf-8") as f:
        f.write("\n")
    with open(output_path + "/test.jsonl", "w", encoding="utf-8") as f:
        f.write("\n")
    print("FM - Dataset Converted to Nemo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform a dataset from Huggingface format to Nemo finetuning format')
    parser.add_argument('--model', type=str, default="Qwen3-8B", help='Model to use for chat template')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to transform')
    args = parser.parse_args()

    model_path, chat_template_fun, _ = get_config(args.model, task="math")
    model, tokenizer = load_model(model_path)

    dataset = load_data(args.dataset)

    output_path = config.DATA_PATHS[1] + args.dataset + "-NEMO"
    
    hf_to_nemo(dataset, output_path, chat_template_fun, tokenizer)

    shutil.copytree(output_path, config.DATA_PATHS[2] + args.dataset + "-NEMO", dirs_exist_ok=True)