import shutil
import argparse
import sys
import os

from tqdm import tqdm

import config
from utils_model import *
from process_data.utils_data import *   


def infer(model, dataset, dataloader, output_name):
    outputs = []
    for x in tqdm(dataloader):
        request_outputs = model.generate(x)
        outputs += [request_output.outputs[0].text for request_output in request_outputs]

    dataset = dataset.add_column(
        output_name, outputs
    ).remove_columns(
        ["chat_input", "input_length"]
    )
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate dataset with model from english to french')
    parser.add_argument('--model', type=str, default="Qwen3-8B", help='Model to use for translation')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to use for inference')
    parser.add_argument('--task', type=str, default="translation", help='Task to prompt to the model')
    args = parser.parse_args()

    print("FM - Getting Config")
    model_path, chat_template_fun, sampling_params = get_config(args.model, task=args.task, n=1)
    if args.task == "translation":
        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.8, 
            top_k=10, 
            min_p=0, 
            max_tokens=1024,    
            seed=0
        )
    model = load_model(model_path, is_vllm=True)

    dataset = load_data(args.dataset).select(range(16))
    dataset, dataloader, _ = prepare_inference_data(
        dataset,
        chat_template_fun,
        batch_size=64,
        input_name="question" + "_fr"*(args.task[-3:] == "_fr"),
        use_only_input=False
        # use_only_input=True
    )

    if args.task == "translation":
        output_name = "question_fr"
        new_dataset_name = args.dataset + "-FR"
    elif args.task == "math":
        output_name = "solution_" + args.model
        new_dataset_name = args.dataset + "-Solved"
    elif args.task == "math_fr":    
        output_name = "solution_fr_" + args.model
        new_dataset_name = args.dataset + "-Solved"

    print("FM - Infering")
    dataset = infer(model, dataset, dataloader, output_name)
    # print("FM - Saving")
    # dataset.save_to_disk(config.DATA_PATHS[2] + new_dataset_name)
    # shutil.copytree(config.DATA_PATHS[2] + new_dataset_name, config.DATA_PATHS[1] + new_dataset_name, dirs_exist_ok=True)
    # print("FM - Saved")