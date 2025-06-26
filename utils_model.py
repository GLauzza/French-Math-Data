import torch
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import fasttext

import config
# from quantize import quantize


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

import os
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"]="+dynamo"

SYSTEM_INSTRUCTIONS = {
    "math": (
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
    "math_fr": (
        "S'il-te-plaît raisonne étape par étape, et écrit ta réponse finale à l'intérieur de \\boxed{}."
        "Tu doit penser et répondre uniquement en français."
    ),
    "translation": (
        ""
    ),
    "topic": (
        ""
    ),
}

USER_INSTRUCTIONS = {
    "math": (
        ""
    ),
    "math_fr": (
        ""
    ),
    "translation": (
        "Please translate sentence by sentence the full following text in French."
        "Only output the translation."
        "Don't summarize."
        "Preserve any mathematical formula formatting.\n"
        "Text:\n"
    ),
    "topic": (
        "Please classify the math topics of the following text."
        "Only output the topics as a single list separated by commas."
        "The text should be classified into 1 to 10 topics."
        "(e.g 'Linear Algebra, Inequalities, Computer Science, Projections')"
    ),
}


def load_model(model_path, is_vllm=False):
    print("FM - Loading Model:", model_path)
    if model_path == "facebook/fasttext-language-identification":
        return fasttext.load_model(config.MODEL_PATHS[0]+model_path+"/model.bin")
    elif is_vllm:
        try:
            model = LLM(
                config.MODEL_PATHS[0]+model_path,
                    # config.MODEL_PATHS[1]+(model_path.split("/")[-1])+"/Qwen3-32B-Q4_K_M.gguf", 
                    # dtype=torch.bfloat16,
                    # trust_remote_code=True,
                    # quantization="modelopt",
                    # quantization="bitsandbytes",
                    # quantization="AWQ",   
                    # kv_cache_dtype="fp8",
                    # calculate_kv_scales=True,
            )
        except:
            try:
                model = LLM(
                    config.MODEL_PATHS[1]+(model_path.split("/")[-1]), 
                    # config.MODEL_PATHS[1]+(model_path.split("/")[-1])+"/Qwen3-32B-Q4_K_M.gguf", 
                    # dtype=torch.bfloat16,
                    # trust_remote_code=True,
                    # quantization="modelopt",
                    # quantization="bitsandbytes",
                    # quantization="AWQ",
                    # kv_cache_dtype="fp8",
                    # calculate_kv_scales=True
                )
            except:
                model = LLM(
                    config.MODEL_PATHS[2]+(model_path.split("/")[-1]),
                    # config.MODEL_PATHS[1]+(model_path.split("/")[-1])+"/Qwen3-32B-Q4_K_M.gguf", 
                    # dtype=torch.bfloat16,
                    # trust_remote_code=True,
                    # quantization="modelopt",
                    # quantization="bitsandbytes",
                    # quantization="AWQ",
                    # kv_cache_dtype="fp8",
                    # calculate_kv_scales=True,
                )
        print("FM - Loaded Model:", model_path)
        return model
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[0]+model_path, device_map=device)
        except:
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), padding_side='left')
                model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), device_map=device)
            except:
                tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[2]+(model_path.split("/")[-1]), padding_side='left')
                model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[2]+(model_path.split("/")[-1]), device_map=device)
        print("FM - Loaded Model:", model_path)
        return model, tokenizer


def free_vllm(model):
    destroy_model_parallel()
    destroy_distributed_environment()
    del model.llm_engine # should be model.llm_engine.model_executor but executor is not found
    del model
    torch.cuda.empty_cache()
    # torch.distributed.destroy_process_group() # might be useful for distributed, doesn't work yet
    gc.collect()


def to_chat_template_qwen_2_5(task):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    return (lambda x : (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTIONS[task]}<|im_end|>\n"
        f"<|im_start|>user\n{USER_INSTRUCTIONS[task]}{x}<|im_end|>\n"
        f"<|im_start|>assistant\n{language_forcing}"
    ))


def to_chat_template_qwen_3(task):
    is_thinking = "no_"*("math" not in task)
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    close_thinking = '\n</think>\n'*('math' not in task)
    return (lambda x : (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTIONS[task]}<|im_end|>\n"
        f"<|im_start|>user\n{USER_INSTRUCTIONS[task]}{x}/{is_thinking}think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{language_forcing}{close_thinking}"
    ))


def to_chat_template_lucie(task):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    return (lambda x : (
        f"<s><|start_header_id|>system<|end_header_id|>You are a helpful assistant. {SYSTEM_INSTRUCTIONS[task]}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>{USER_INSTRUCTIONS[task]}{x}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>{language_forcing}"
    ))


def to_chat_template_phi4(task):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    if task == "math" or task == "topic":
        introduction = "Your name is Phi, an AI math expert developed by Microsoft."
    elif task == "math_fr":
        introduction = "Ton nom est Phi, une IA experte en math françaises développée par Microsoft."
    elif task == "translation":
        introduction = "Your name is Phi, an AI translation expert developed by Microsoft."
    else:
        raise Exception("task not valid for this chat template")
    return (lambda x : (
        f"<|system|>{introduction} {SYSTEM_INSTRUCTIONS[task]}<|end|>"
        f"<|user|>{USER_INSTRUCTIONS[task]}{x}<|end|>"
        f"<|assistant|>{language_forcing}"
    ))


def to_chat_template_deepseek(task):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    start_thinking = "<think>\n"*("math" in task)
    return (lambda x : (
        f"{USER_INSTRUCTIONS[task]}{x}\n{SYSTEM_INSTRUCTIONS[task]}\n{start_thinking}{language_forcing}"
    ))


def get_config(name, task="math", n=1, max_length=1000000):
    print("FM - Getting Config:", name, task)
    DEFAULT_CHAT_TEMPLATE = to_chat_template_qwen_2_5(task)
    DEFAULT_SAMPLING_PARAMS = SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=30, presence_penalty=0.5, max_tokens=min(32768, max_length), seed=0)

    if name == "fasttext":
        return (
            "facebook/fasttext-language-identification",
            (lambda x: x),
            None,
        )
    elif name == "Qwen2.5-Math-7B-Instruct":
        return (
            "Qwen/Qwen2.5-Math-7B-Instruct",
            to_chat_template_qwen_2_5(task),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name == "Qwen3-8B":
        return ( 
            "Qwen/Qwen3-8B",
            to_chat_template_qwen_3(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=min(38912, max_length), seed=0)
        )
    elif name == "Qwen3-32B":
        # quantize(config.MODEL_PATHS[0]+"Qwen/Qwen3-32B", config.MODEL_PATHS[1]+"Qwen3-32B-quantized", "compressor")
        return (
            "Qwen/Qwen3-32B",
            # "Qwen/Qwen3-32B-AWQ",
            # "Qwen/Qwen3-32B-bnb-4bit",
            # "Qwen/Qwen3-32B-unsloth-bnb-4bit", # Unsupported in vllm yet
            # "Qwen/Qwen3-32B-GPTQ-Int4",
            # "Qwen/Qwen3-32B-GPTQ-Int8",
            # "Qwen/Qwen3-32B-FP8-dynamic",
            # "Qwen/Qwen3-32B-quantized.w4a16", # Doesn't work
            # "Qwen/Qwen3-32B.w8a8",
            # "Qwen/Qwen3-32B-quantized",
            # "Qwen/Qwen3-32B-GGUF",
            # "Qwen/Qwen3-32B-FP8",
            # "Qwen/Qwen3-32B-FP8-KV",
            to_chat_template_qwen_3(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=min(38912, max_length), seed=0)
        )
    elif name == "Qwen3-30B-A3B":
        return (
            "Qwen/Qwen3-30B-A3B",
            to_chat_template_qwen_3(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=min(38912, max_length), seed=0)
        )
    elif name == "Lucie-7B-Instruct-v1.1":
        return (
            "OpenLLM-France/Lucie-7B-Instruct-v1.1", 
            to_chat_template_lucie(task),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name == "Phi-4-mini-reasoning":
        return (
            "microsoft/Phi-4-mini-reasoning", 
            to_chat_template_phi4(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name == "deepseek-math-7b-instruct":
        return (
            "deepseek-ai/deepseek-math-7b-instruct", 
            to_chat_template_deepseek(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name == "DeepSeek-R1-Distill-Qwen-7B":
        return (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            to_chat_template_deepseek(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name == "DeepSeek-R1-Distill-Llama-8B":
        return (
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
            to_chat_template_deepseek(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name == "OpenR1-Distill-7B":
        return (
            "open-r1/OpenR1-Distill-7B", 
            to_chat_template_deepseek(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name == "Pensez-v0.1-e5":
        return (
            "HoangHa/Pensez-v0.1-e5", 
            DEFAULT_CHAT_TEMPLATE,
            SamplingParams(n=n, temperature=0.8, repetition_penalty=1.1, max_tokens=min(32768, max_length), seed=0)
        )
    elif name == "Llama-3.1-8B-Instruct":
        return (
            "meta-llama/Llama-3.1-8B-Instruct", 
            DEFAULT_CHAT_TEMPLATE,
            DEFAULT_SAMPLING_PARAMS
        )

def get_configs(names, task="math", n=1, max_length=1000000):
    configs = []
    for name in names:
        configs.append(get_config(name, task, n, max_length))
    return configs