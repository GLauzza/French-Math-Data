import shutil
import argparse
import sys
import os

from vllm import SamplingParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
from utils_model import *
from process_data.utils_data import *


def translate(model, chat_template_fun, dataset, batch_size=64):
    dataset, dataloader, _ = prepare_inference_data(chat_template_fun, dataset, batch_size)

    for x in tqdm(dataloader):
        request_outputs = model.generate(
            x["chat_input"],
            SamplingParams(
                temperature=0.6,
                top_p=0.95, 
                top_k=20, 
                min_p=0, 
                presence_penalty=0.5, 
                max_tokens=1024, 
                seed=0
            )
        )
        output_texts = [request_output.outputs[0].text for request_output in request_outputs]
        for sample, output_text in zip(x, output_texts):
            sample["question_fr"] = output_text
        break   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate dataset with model from english to french')
    parser.add_argument('--model', type=str, default="Qwen3-8B", help='Model to use for translation')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to translate')
    args = parser.parse_args()

    print("FM - Getting Config")
    model_path, chat_template_fun, _ = get_config(args.model)
    model, tokenizer = load_model(model_path)

    dataset = load_data(args.dataset).select(range(16))

    output_path = config.MODEL_PATHS[2] + args.dataset + "_FR"

    print("FM - Evaluating")
    translate(model, chat_template_fun, dataset)

    print("FM - Saving")
    dataset.save_to_disk(output_path)
    shutil.copytree(output_path, config.DATA_PATHS[1] + args.dataset + "_FR")
    print("FM - Saved")