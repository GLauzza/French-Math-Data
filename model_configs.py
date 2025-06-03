from vllm import SamplingParams


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


def to_chat_template_lucie(x):
    chat = (
        "<s><|start_header_id|>system<|end_header_id|>You are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{}.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>" + x + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    return chat


def to_chat_template_phi4(x):
    chat = (
        "<|system|>Your name is Phi, an AI math expert developed by Microsoft. Please reason step by step, and put your final answer within \\boxed{}.<|end|>"
        "<|user|>" + x + "<|end|>"
        "<|assistant|>"
    )
    return chat


def to_chat_deepseek(x):
    chat = (
        x + "\nPlease reason step by step, and put your final answer within \\boxed{}.<think>\n"
    )
    return chat


DEFAULT_CHAT_TEMPLATE = to_chat_template_qwen_2_5
DEFAULT_SAMPLING_PARAMS = SamplingParams(n=5, temperature=0.6, top_p=0.95, top_k=30, presence_penalty=0.5, max_tokens=32768)
def get_config(name):
    if name == "Qwen2.5-Math-7B-Instruct":
        return (
            "Qwen/Qwen2.5-Math-7B-Instruct",
            to_chat_template_qwen_2_5,
            DEFAULT_SAMPLING_PARAMS
        )
    elif name == "Qwen3-8B":
        return (
            "Qwen/Qwen3-8B",
            to_chat_template_qwen_3,
            SamplingParams(n=5, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=38912)
        )
    elif name == "Lucie-7B-Instruct-v1.1":
        return (
            "OpenLLM-France/Lucie-7B-Instruct-v1.1", 
            to_chat_template_lucie,
            DEFAULT_SAMPLING_PARAMS
        )
    elif name == "Phi-4-mini-reasoning":
        return (
            "microsoft/Phi-4-mini-reasoning", 
            to_chat_template_phi4,
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768)
        )
    elif name == "deepseek-math-7b-instruct":
        return (
            "deepseek-ai/deepseek-math-7b-instruct", 
            to_chat_deepseek,
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768)
        )
    elif name == "DeepSeek-R1-Distill-Qwen-7B":
        return (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            to_chat_deepseek,
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768)
        )
    elif name == "DeepSeek-R1-Distill-Llama-8B":
        return (
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
            to_chat_deepseek,
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768)
        )
    elif name == "OpenR1-Distill-7B":
        return (
            "open-r1/OpenR1-Distill-7B", 
            to_chat_deepseek,
            SamplingParams(n=5, temperature=0.6, top_p=0.95, max_tokens=32768)
        )
    elif name == "Pensez-v0.1-e5":
        return (
            "HoangHa/Pensez-v0.1-e5", 
            DEFAULT_CHAT_TEMPLATE,
            SamplingParams(n=5, temperature=0.8, repetition_penalty=1.1, max_tokens=32768)
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