import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from math_verify import parse, verify
from vllm import LLM, SamplingParams

import config
from process_data.utils_data import *


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def load_model(model_path):
    try:
        # tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[0]+model_path, padding_side='left')
        # model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[0]+model_path, device_map=device)
        model = LLM(config.MODEL_PATHS[0]+model_path)
    except:
        # tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), padding_side='left')
        # model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), device_map=device)
        model = LLM(config.MODEL_PATHS[1]+(model_path.split("/")[-1]))
    # return model, tokenizer
    return model, None


def to_chat_template_qwen_2_5(x):
    chat = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>user\n" + x + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return chat

def to_chat_template_qwen_3(x):
    chat = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>user\n" + x + "/think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return chat


def eval_model(model, tokenizer, chat_template_fun, batch_size=32, max_new_tokens=10000, eval_dataset="Eval-Math-FR"):
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

    # Ideal batch size is 32/64
    # for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    #     with_pad = 0
    #     without_pad = 0
    #     for i in range(0, len(dataset), bs):
    #         if i + bs > len(dataset):
    #             bs = len(dataset) - i
    #         with_pad += bs*dataset["input_length"][i+bs-1]
    #         without_pad += sum(dataset["input_length"][i:i+bs])
    #     print(f"Batch size: {bs}, With padding: {with_pad}, Without padding: {without_pad}, Padding ratio: {without_pad/with_pad}")

    sources = set(dataset["source"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accuracies, cot_lengths, samples = {}, {}, {}
    for source in sources:
        accuracies[source] = 0
        cot_lengths[source] = 0
        samples[source] = []
    for x in dataloader:
        request_outputs = model.generate(x["chat_input"], SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
        input_ids = [request_output.prompt_token_ids for request_output in request_outputs]
        output_ids = [request_output.outputs[0].token_ids for request_output in request_outputs]
        output_texts = [request_output.outputs[0].text for request_output in request_outputs]
        for input_id, output_id, output_text, answer, source in zip(input_ids, output_ids, output_texts, x["answer"], x["source"]):
            pred = "$" + extract_boxed_text(output_text) + "$"
            answer = "$" + answer + "$"
            parsed_pred = parse(pred)
            parsed_answer = parse(answer)
            is_valid = verify(parsed_answer, parsed_pred)
            cot_length = len(output_id) - len(input_id)
            accuracies[source] += is_valid
            cot_lengths[source] += cot_length
            samples[source].append({
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


def eval_models(models_to_evaluate):
    output = {}
    for model_path, chat_template_fun in models_to_evaluate:
        model, tokenizer     = load_model(model_path)
        accuracies, cot_lengths, samples = eval_model(model, tokenizer, chat_template_fun)
        output[model_path] = {
            "accuracies": accuracies,
            "cot_lengths": cot_lengths,
            "samples": samples,
        }
        print("Output:", output)
        with open("eval.json", "w") as f:
            json.dump(output, f)
            print("dumped")


if __name__ == "__main__":
    models_to_evaluate = [
        ("Qwen/Qwen2.5-Math-7B-Instruct", to_chat_template_qwen_2_5),
        # ("Qwen/Qwen3-8B", to_chat_template_qwen_3),
        # TODO: create chat template for every evaluated model
        # ("OpenLLM-France/Lucie-7B-Instruct-v1.1", to_chat_template_qwen_2_5),
        # ("microsoft/Phi-4-mini-reasoning", to_chat_template_qwen_2_5),
        # ("deepseek-ai/deepseek-math-7b-instruct", to_chat_template_qwen_2_5),
        # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", to_chat_template_qwen_2_5),
        # ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", to_chat_template_qwen_2_5),
        # ("open-r1/OpenR1-Distill-7B", to_chat_template_qwen_2_5),
        # ("HoangHa/Pensez-v0.1-e5", to_chat_template_qwen_2_5),
        # ("meta-llama/Llama-3.1-8B-Instruct", to_chat_template_qwen_2_5),
    ]
    eval_models(models_to_evaluate)