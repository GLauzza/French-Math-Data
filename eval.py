import json
import shutil
import argparse

from tqdm import tqdm
from math_verify import parse, verify

import config
from process_data.extract_answer import *
from process_data.prepare_data import *
from utils_model import *


def eval_model(model, chat_template_fun, sampling_params, dataset_name, batch_size):
    dataset = load_data(dataset_name)
    dataset, dataloader, sources = prepare_inference_data(
        dataset,
        chat_template_fun,
        model.get_tokenizer(),
        batch_size=batch_size,
    )

    accuracies, cot_lengths, samples = {}, {}, {}
    for source in sources:
        accuracies[source] = 0
        cot_lengths[source] = 0
        samples[source] = []

    for x in tqdm(dataloader):
        request_outputs = model.generate(x["chat_input"], sampling_params)
        output_ids = [request_output.outputs[0].token_ids for request_output in request_outputs]
        output_texts = [request_output.outputs[0].text for request_output in request_outputs]
        for input_text, output_id, output_text, answer, source in zip(x["chat_input"], output_ids, output_texts, x["answer"], x["source"]):
            #pred = to_latex(extract_boxed_text(output_text))
            pred = output_text
            answer = to_latex(answer)
            parsed_pred = parse(pred)
            parsed_answer = parse(answer)
            is_valid = verify(parsed_answer, parsed_pred)
            cot_length = len(output_id)
            accuracies[source] += is_valid
            cot_lengths[source] += cot_length
            samples[source].append({
                "input": input_text,
                "generation": output_text,
                "cot_length": cot_length,
                "is_valid": is_valid,
                "answer": parsed_answer[1] if len(parsed_answer) > 0 else "",
                "pred": parsed_pred[1] if len(parsed_pred) > 0 else "",
            })

    for source in sources:
        n_source = len(dataset.filter(lambda x: x["source"] == source))
        accuracies[source] = accuracies[source] / n_source
        cot_lengths[source] = cot_lengths[source] / n_source

    free_vllm(model)

    return accuracies, cot_lengths, samples


def eval_models(models_configs, dataset_name, batch_size):
    print("FM - Loading Evaluation File")
    json_path = config.DATA_PATHS[1] + dataset_name + "/eval.json"
    if os.path.exists(json_path):
        with open(json_path, "r+") as f:
            output = json.load(f)
    else:
        output = {} 

    for model_path, chat_template_fun, sampling_params in models_configs:
        model = load_model(model_path, is_vllm=True)
        print("FM - Evaluating:", model_path, dataset_name)
        accuracies, cot_lengths, samples = eval_model(model, chat_template_fun, sampling_params, dataset_name, batch_size)
        output[model_path] = {
            "accuracies": accuracies,
            "cot_lengths": cot_lengths,
            "samples": samples,
        }
        
        print("FM - Dumping:", model_path, dataset_name)
        with open(json_path, 'w') as f:
            json.dump(output, f)
        print("FM - Dumped", output)

    shutil.copy(config.DATA_PATHS[1] + dataset_name + "/eval.json", config.DATA_PATHS[2] + dataset_name + "/eval.json")


if __name__ == "__main__":
    default_models = [
        "Qwen2.5-Math-7B-Instruct",
        "Qwen3-8B",
        "Lucie-7B-Instruct-v1.1",
        "Phi-4-mini-reasoning",
        # "deepseek-math-7b-instruct",
        "DeepSeek-R1-Distill-Qwen-7B",
        # "DeepSeek-R1-Distill-Llama-8B",
        # "OpenR1-Distill-7B",
        "Pensez-v0.1-e5",
        "Llama-3.1-8B-Instruct",
    ]

    parser = argparse.ArgumentParser(description='Evaluate a list of models on the dataset')
    parser.add_argument('--models', nargs='+', default=default_models, help ="Models to evaluate on")
    parser.add_argument('--dataset', type=str, default="Eval-Math-FR", help='Dataset to evaluate on')
    parser.add_argument('--n', type=int, default=1, help='Number of samples per example')
    parser.add_argument('--batch_size', type=int, default=-1, help='Batch size')
    args = parser.parse_args()

    models_configs = get_configs(args.models, task="math", n=args.n)

    eval_models(models_configs, args.dataset, args.batch_size)
