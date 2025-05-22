import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from math_verify import parse, verify

import config
from process_data.utils_data import *


models_to_evaluate = ["Qwen2.5-Math-1.5B-Instruct"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[0]+model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[0]+model_path, device_map=device)
    except:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[1]+model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[1]+model_path, device_map=device)
    return tokenizer, model


def to_chat_template(x):
    chat = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>user\n" + x + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return chat


def eval_model(model, tokenizer, batch_size=1, max_new_tokens=5000, eval_dataset="Eval-Math-FR"):
    dataset = datasets.load_from_disk(config.DATA_PATHS[1]+eval_dataset)
    dataset = dataset.add_column(
        "chat_input",
        [to_chat_template(x) for x in dataset["question"]]
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accuracy, cot_length = 0, 0
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            tokens = tokenizer(x["chat_input"], return_tensors='pt', padding=True).to(device)
            output_ids = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False, temperature=0).cpu()
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_id, output_text, answer in zip(output_ids, output_texts, x["answer"]):
                accuracy += verify(parse(answer), parse(extract_boxed_text(output_text)))
                cot_length += sum(output_id.view(-1) != tokenizer.pad_token_id)
            print("output", output_text)
            break
        accuracy = accuracy / len(dataloader)
        cot_length = cot_length / len(dataloader)
    return accuracy, cot_length


if __name__ == "__main__":
    for model_path in models_to_evaluate:
        tokenizer, model = load_model(model_path)
        accuracy, cot_length = eval_model(model, tokenizer)
        print("accuracy", accuracy)
        print("cot_length", cot_length)