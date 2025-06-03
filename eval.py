import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from math_verify import parse, verify
from vllm import LLM

import config
from process_data.utils_data import *
from model_configs import *


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def load_model(model_path):
    try:
        model = LLM(config.MODEL_PATHS[0]+model_path)
    except:
        model = LLM(config.MODEL_PATHS[1]+(model_path.split("/")[-1]))
    return model


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


def prepare_data(chat_template_fun, eval_dataset, batch_size):
    dataset = datasets.load_from_disk(config.DATA_PATHS[1]+eval_dataset)
    dataset = dataset.add_column(
        "chat_input",
        [chat_template_fun(x) for x in dataset["question"]]
    )
    dataset = dataset.add_column(
        "input_length",
        [len(x) for x in dataset["question"]]
    )
    dataset = dataset.sort("input_length")

    sources = set(dataset["source"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader, sources


def eval_model(model, chat_template_fun, sampling_params, batch_size=64, eval_dataset="Eval-Math-FR"):
    dataset, dataloader, sources = prepare_data(chat_template_fun, eval_dataset, batch_size)
    accuracies, cot_lengths, samples = {}, {}, {}

    for source in sources:
        accuracies[source] = 0
        cot_lengths[source] = 0
        samples[source] = []

    for x in dataloader:
        request_outputs = model.generate(x["chat_input"], sampling_params)
        output_ids = [request_output.outputs[0].token_ids for request_output in request_outputs]
        output_texts = [request_output.outputs[0].text for request_output in request_outputs]
        for output_id, output_text, answer, source in zip(output_ids, output_texts, x["answer"], x["source"]):
            pred = to_latex(extract_boxed_text(output_text))
            answer = to_latex(answer)
            parsed_pred = parse(pred)
            parsed_answer = parse(answer)
            is_valid = verify(parsed_answer, parsed_pred)
            cot_length = len(output_id)
            accuracies[source] += is_valid
            cot_lengths[source] += cot_length
            samples[source].append({
                "input": x["chat_input"],
                "generation": output_text,
                "cot_length": cot_length,
                "is_valid": is_valid,
                "answer": (answer, str(parsed_answer)),
                "output": (pred, str(parsed_pred)),
            })
        break

    for source in sources:
        n_source = len(dataset.filter(lambda x: x["source"] == source))
        accuracies[source] = accuracies[source] / n_source
        cot_lengths[source] = cot_lengths[source] / n_source

    return accuracies, cot_lengths, samples


def eval_models(models_to_evaluate):
    with open("eval.json", "r+") as f:
        output = json.load(f)

        for model_path, chat_template_fun, sampling_params in models_to_evaluate:
            model = load_model(model_path)
            accuracies, cot_lengths, samples = eval_model(model, chat_template_fun, sampling_params)
            output[model_path] = {
                "accuracies": accuracies,
                "cot_lengths": cot_lengths,
                "samples": samples,
            }
            f.seek(0)
            json.dump(output, f)
            f.truncate()
            print("Dumped:", output)


if __name__ == "__main__":
    models_to_evaluate = get_configs([
        # "Qwen2.5-Math-7B-Instruct",
        # "Qwen3-8B",
        "Lucie-7B-Instruct-v1.1",
        # "Phi-4-mini-reasoning",
        # # "deepseek-math-7b-instruct",
        # "DeepSeek-R1-Distill-Qwen-7B",
        # # "DeepSeek-R1-Distill-Llama-8B",
        # # "OpenR1-Distill-7B",
        # "Pensez-v0.1-e5",
        # "Llama-3.1-8B-Instruct",
    ])
    eval_models(models_to_evaluate)