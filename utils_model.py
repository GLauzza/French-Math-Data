import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

import config


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
TRANSLATION_INSTRUCTION = "Please Translate the following question in French."


def load_model(model_path, is_vllm=False):
    print("FM - Loading Model:", model_path)
    if is_vllm:
        try:
            model = LLM(config.MODEL_PATHS[0]+model_path)
        except:
            model = LLM(config.MODEL_PATHS[1]+(model_path.split("/")[-1]))
        print("FM - Loaded Model:", model_path)
        return model
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[0]+model_path, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[0]+model_path, device_map=device)
        except:
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), device_map=device)
        print("FM - Loaded Model:", model_path)
        return model, tokenizer


def to_chat_template_qwen_2_5(config_type):
    if config_type == "math":
        return (lambda x : (
            f"<|im_start|>system\n{MATH_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{x}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ))
    elif config_type == "translate":
        return (lambda x : (
            f"<|im_start|>system\n{TRANSLATION_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{x}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ))
    else:
        raise Exception("Config Type not supported")


def to_chat_template_qwen_3(config_type):
    if config_type == "math":
        return (lambda x : (
            f"<|im_start|>system\n{MATH_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{x}/think<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ))
    elif config_type == "translate":
        return (lambda x : (
            f"<|im_start|>system\n{TRANSLATION_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{x}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ))
    else:
        raise Exception("Config Type not supported")


def to_chat_template_lucie(config_type):
    if config_type == "math":
        return (lambda x : (
            f"<s><|start_header_id|>system<|end_header_id|>You are a helpful assistant. {MATH_INSTRUCTION}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>{x}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>"
        ))
    elif config_type == "translate":
        return (lambda x : (
            f"<s><|start_header_id|>system<|end_header_id|>You are a helpful assistant. {TRANSLATION_INSTRUCTION}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>{x}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>"
        ))
    else:
        raise Exception("Config Type not supported")


def to_chat_template_phi4(config_type):
    if config_type == "math":
        return (lambda x : (
            "<|system|>Your name is Phi, an AI math expert developed by Microsoft. {MATH_INSTRUCTION}<|end|>"
            "<|user|>{x}<|end|>"
            "<|assistant|>"
        ))
    elif config_type == "translate":
        return (lambda x : (
            "<|system|>Your name is Phi, an AI translator expert developed by Microsoft. {TRANSLATION_INSTRUCTION}<|end|>"
            "<|user|>{x}<|end|>"
            "<|assistant|>"
        ))
    else:
        raise Exception("Config Type not supported")


def to_chat_template_deepseek(config_type):
    if config_type == "math":
        return (lambda x : (
            f"{x}\n{MATH_INSTRUCTION}\n<think>\n"
        ))
    elif config_type == "translate":
        return (lambda x : (
            f"{x}\n{TRANSLATION_INSTRUCTION}\n"
        ))
    else:
        raise Exception("Config Type not supported")


DEFAULT_CHAT_TEMPLATE = to_chat_template_qwen_2_5
DEFAULT_SAMPLING_PARAMS = SamplingParams(n=5, temperature=0.6, top_p=0.95, top_k=30, presence_penalty=0.5, max_tokens=32768, seed=0)
def get_config(name, config_type="math"):
    if name == "Qwen2.5-Math-7B-Instruct":
        return (
            "Qwen/Qwen2.5-Math-7B-Instruct",
            to_chat_template_qwen_2_5(config_type),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name == "Qwen3-8B":
        return (
            "Qwen/Qwen3-8B",
            to_chat_template_qwen_3(config_type),
            SamplingParams(n=5, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=38912, seed=0)
        )
    elif name == "Lucie-7B-Instruct-v1.1":
        return (
            "OpenLLM-France/Lucie-7B-Instruct-v1.1", 
            to_chat_template_lucie(config_type),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name == "Phi-4-mini-reasoning":
        return (
            "microsoft/Phi-4-mini-reasoning", 
            to_chat_template_phi4(config_type),
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768, seed=0)
        )
    elif name == "deepseek-math-7b-instruct":
        return (
            "deepseek-ai/deepseek-math-7b-instruct", 
            to_chat_template_deepseek(config_type),
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768, seed=0)
        )
    elif name == "DeepSeek-R1-Distill-Qwen-7B":
        return (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            to_chat_template_deepseek(config_type),
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768, seed=0)
        )
    elif name == "DeepSeek-R1-Distill-Llama-8B":
        return (
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
            to_chat_template_deepseek(config_type),
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768, seed=0)
        )
    elif name == "OpenR1-Distill-7B":
        return (
            "open-r1/OpenR1-Distill-7B", 
            to_chat_template_deepseek(config_type),
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768, seed=0)
        )
    elif name == "Pensez-v0.1-e5":
        return (
            "HoangHa/Pensez-v0.1-e5", 
            DEFAULT_CHAT_TEMPLATE,
            SamplingParams(n=5, temperature=0.8, repetition_penalty=1.1, max_tokens=32768, seed=0)
        )
    elif name == "Llama-3.1-8B-Instruct":
        return (
            "meta-llama/Llama-3.1-8B-Instruct", 
            DEFAULT_CHAT_TEMPLATE,
            DEFAULT_SAMPLING_PARAMS
        )

def get_configs(names):
    configs = []
    for name in names:
        configs.append(get_config(name))
    return configs