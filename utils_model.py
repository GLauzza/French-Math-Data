import torch
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import fasttext

import config


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

SYSTEM_INSTRUCTIONS = {
    "math": (
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
    "math_fr": (
        "S'il-te-plaît raisonne étape par étape, et écrit ta réponse finale à l'intérieur de \\boxed{}."
        "Tu doit penser et répondre uniquement en français."
    ),
    "translation_question": (
        ""
    ),
    "translation_solution": (
        ""
    ),
    "translation_answer": (
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
    "translation_question": (
        "Please translate sentence by sentence the full following question in French.\n"
        "- Only output the translation.\n"
        "- Don't solve the problem, only translate.\n"
        "- Preserve any mathematical formula formatting.\n"
        "Question:\n"
    ),
    "translation_solution": (
        "Please translate sentence by sentence the full following text in French.\n"
        "- Only output the translation.\n"
        # "- Don't summarize.\n"
        "- Don't solve the problem, only translate.\n"
        "- Preserve any mathematical formula formatting.\n"
        "- Don't translate what is inside \\boxed{}.\n"
        "Text:\n"
    ),
    "translation_answer": (
        "Please translate sentence by sentence the full following answer in French.\n"
        "- Only output the translation.\n"
        "- Preserve any mathematical formula formatting.\n"
        "Answer:\n"
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
                enable_prefix_caching=True,
                # dtype="float32",
                # enforce_eager=True,
                # tensor_parallel_size=1,
                # distributed_executor_backend="mp",  # multiprocessing, more isolated
                # # worker_use_ray=False,
                # disable_custom_all_reduce=True,  # Avoids custom NCCL kernels
                # # speculative_model=None, 
                # # num_speculative_tokens=0,
                seed=0,
            )
        except:
            try:
                model = LLM(
                    config.MODEL_PATHS[1]+(model_path.split("/")[-1]), 
                    enable_prefix_caching=True,
                    # dtype="float32",
                    # enforce_eager=True,
                    # tensor_parallel_size=1,
                    # distributed_executor_backend="mp",  # multiprocessing, more isolated
                    # # worker_use_ray=False,
                    # disable_custom_all_reduce=True,  # Avoids custom NCCL kernels
                    # # speculative_model=None, 
                    # # num_speculative_tokens=0,
                    seed=0,
                )
            except:
                model = LLM(
                    config.MODEL_PATHS[2]+(model_path.split("/")[-1]),
                    enable_prefix_caching=True,
                    # dtype="float32",
                    # enforce_eager=True,
                    # tensor_parallel_size=1,
                    # distributed_executor_backend="mp",
                    # # worker_use_ray=False,
                    # disable_custom_all_reduce=True,  # Avoids custom NCCL kernels
                    # # speculative_model=None, 
                    # # num_speculative_tokens=0,
                    seed=0,
                )
        print("FM - Loaded Model:", model_path)
        return model
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[0]+model_path, padding_side='left')
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
    elif "translation" in task:
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

    if name.startswith("fasttext"):
        return (
            f"facebook/fasttext-language-identification",   
            (lambda x: x),
            None,
        )
    elif name.startswith("Qwen2.5"):
        return (
            f"Qwen/{name}",
            to_chat_template_qwen_2_5(task),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name.startswith("Qwen3"):
        return (
            f"Qwen/{name}",
            to_chat_template_qwen_3(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=min(38912, max_length), seed=0)
        )
    elif name.startswith("Lucie"):
        return (
            f"OpenLLM-France/{name}", 
            to_chat_template_lucie(task),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name.startswith("Phi-4"):
        return (
            f"microsoft/{name}", 
            to_chat_template_phi4(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("deepseek-math") or name.startswith("DeepSeek-R1"):
        return (
            f"deepseek-ai/{name}", 
            to_chat_template_deepseek(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("OpenR1"):
        return (
            f"open-r1/{name}", 
            to_chat_template_deepseek(task),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("Pensez"):
        return (
            f"HoangHa/{name}", 
            DEFAULT_CHAT_TEMPLATE,
            SamplingParams(n=n, temperature=0.8, repetition_penalty=1.1, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("Llama"):
        return (
            f"meta-llama/{name}", 
            DEFAULT_CHAT_TEMPLATE,
            DEFAULT_SAMPLING_PARAMS
        )
    else:
        raise Exception(f"Model {name} not supported. Edit utils_model.py to add support.")

def get_configs(names, task="math", n=1, max_length=1000000):
    configs = []
    for name in names:
        configs.append(get_config(name, task, n, max_length))
    return configs
