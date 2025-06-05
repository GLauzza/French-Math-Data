import json

from tqdm import tqdm
import torch
import datasets
from math_verify import parse, verify

import config
from process_data.utils_data import *
from utils_model import *


def to_latex(text):
    if (
        (text[0] == "$" and text[-1] == "$") or
        (text[0] == "[" and text[-1] == "]") or
        (text[:2] == "\\[" and text[-2:] == "\\]") or
        (text[:2] == "\\(" and text[-2:] == "\\)") or
        (text[:7] == "\\boxed{" and text[-1] == "}")
    ):
        return text
    else:
        return "$" + text + "$"


def prepare_data(chat_template_fun, dataset, batch_size):
    print("FM - Preparing Data")
    dataset = dataset.add_column(
        "chat_input",
        [chat_template_fun(question) for question in dataset["question"]]
    )
    dataset = dataset.add_column(
        "input_length",
        [len(question) for question in dataset["question"]]
    )
    dataset = dataset.sort("input_length")

    sources = set(dataset["source"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("FM - Prepared Data")
    return dataset, dataloader, sources


def eval_model(model, chat_template_fun, sampling_params, dataset_name, batch_size=64):
    dataset = load_data(dataset_name)
    dataset, dataloader, sources = prepare_data(chat_template_fun, dataset, batch_size)
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
            pred = to_latex(extract_boxed_text(output_text))
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
                "answer": (answer, str(parsed_answer)),
                "output": (pred, str(parsed_pred)),
            })

    for source in sources:
        n_source = len(dataset.filter(lambda x: x["source"] == source))
        accuracies[source] = accuracies[source] / n_source
        cot_lengths[source] = cot_lengths[source] / n_source

    return accuracies, cot_lengths, samples


def eval_models(models_configs, dataset_name):
    with open("eval.json", "r+") as f:
        print("FM - Loading Evaluation File")
        output = json.load(f)

        for model_path, chat_template_fun, sampling_params in models_to_evaluate:
            model = load_model(model_path, is_vllm=True)
            print("FM - Evaluating")
            accuracies, cot_lengths, samples = eval_model(model, chat_template_fun, sampling_params, dataset_name)
            print("FM - Dumping")
            output[model_path] = {
                "accuracies": accuracies,
                "cot_lengths": cot_lengths,
                "samples": samples,
            }
            f.seek(0)
            json.dump(output, f)
            f.truncate()
            print("FM - Dumped", output)


if __name__ == "__main__":
    MODELS_NAMES = [
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
    DATASET_NAME = "Eval-Math-FR"

    print("FM - Getting configs")
    models_configs = get_configs(MODELS_NAMES)
    eval_models(models_configs, DATASET_NAME)