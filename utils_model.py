# import unsloth

import torch
import gc

from sentence_transformers import SentenceTransformer
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
    "difficulty": (
        ""
    ),
    "knowledge": (
        ""
    ),
    "steps": (
        ""
    ),
    "quality": (
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
        "Please translate the full following question in French.\n"
        "- Only output the translation.\n"
        "- Only translate what is after <|Question|>.\n"
        "- Don't solve the problem, only translate.\n"
        "- Preserve any mathematical formula formatting.\n"
        "<|Question|>\n"
    ),
    "translation_solution": (
        "Please translate sentence by sentence the full following text in French.\n"
        "- Only output the translation.\n"
        "- Only translate what is after <|Text|>.\n"
        # "- Don't summarize.\n"
        "- Don't solve the problem, only translate.\n"
        "- Preserve any mathematical formula formatting.\n"
        "- Don't translate what is inside \\boxed{}.\n"
        "<|Text|>\n"
    ),
    "translation_answer": (
        "Please translate the full following answer in French.\n"
        "- Only output the translation.\n"
        "- Only translate what is after <|Answer|>.\n"
        "- Preserve any mathematical formula formatting.\n"
        "<|Answer|>\n"
    ),
    "topic": (
        "Please classify the math topics of the following text.\n"
        "Only output the topics as a single list separated by commas.\n"
        "The text should be classified into 1 to 10 topics."
        "(e.g 'Linear algebra, Inequalities, Computer science, Projections')\n"
    ),
    "difficulty": (
        "Please classify the math difficulty level of the following text.\n"
        "Only output the math difficulty level.\n"
        "The math difficulty level should be classified as either n-th grade, n-th year undergrad, Master, PhD, AMC-n, AIME, USAJMO, USAMO, MOP, IMO or Putnam.\n"
    ),
    "knowledge": (
        "We define a piece of knowledge as either a theorem, a fact, a relationship, a formula, a logical procedure or a definition.\n"
        "Please give the number of pieces of knowledge that are used in this solution.\n"
        "Don't count twice same pieces of knowledge, make sure every piece is unique and needed to solve the problem.\n"
        "Only output the number of pieces of knowledge.\n"
    ),
    "steps": (
        "We define a reasoning step as either a logical induction, a logical deduction, the use of a theorem, an equation step or a computation step.\n"
        "Please give the number of reasoning steps used in this solution.\n"
        "Don't count twice identic reasoning steps.\n"
        "Only output the number of reasoning steps.\n"
    ),
    "quality": (
        # "You are the admin of a math forum.\n"
        # "Your role is to give a decision about this solution based on its quality:\n"
        # "Upvote: The solution is extremely well formulated, >98%% factually correct, with no inconsistencies, fluent and easily understood by the targeted audience.\n"
        # "Keep: The solution is well formulated, >95%% factually correct, with few inconsistencies, mostly fluent and mostly understood by the targeted audience.\n"
        # "Downvote: The solution is mostly well formulated, >70%% factually correct, with some inconsistencies, not always fluent and not always understood by the targeted audience.\n"
        # "Remove: The solution is not well formulated, <70%% factually correct, with too much inconsistencies, not fluent or not well understood by the targeted audience.\n"
        # "Only output the decision.\n"
        """
        You are the strict admin of a math forum. Your job is to determine whether the solution is fully correct.

        A solution is **Correct** if:
        - All reasoning steps are logically valid and mathematically accurate.
        - The final answer is correct.
        - No critical errors (e.g., wrong formula, algebra mistake, invalid assumption).

        A solution is **Incorrect** if:
        - There is any significant error in logic, calculation, or method.
        - The final answer is wrong.
        - Key steps are missing or unjustified.
        - It contains hallucinations or nonsense.

        Be strict.

        Output Correct or Incorrect and then your justification of the label.
        """
    ),
}


def load_model(model_path, is_vllm=False, is_unsloth=False, accelerator=None, pc=None):
    print("FM - Loading Model:", model_path)
    if model_path == "facebook/fasttext-language-identification":
        return fasttext.load_model(config.MODEL_PATHS[0]+model_path+"/model.bin")
    elif is_vllm:
        if "embed" in model_path.lower():
            try:
                model = LLM(
                    config.MODEL_PATHS[0]+model_path,
                    enable_prefix_caching=True,
                    task="embed",
                    seed=0,
                )
            except:
                try:
                    model = LLM(
                        config.MODEL_PATHS[1]+(model_path.split("/")[-1]), 
                        enable_prefix_caching=True,
                        task="embed",
                        seed=0,
                    )
                except:
                    model = LLM(
                        config.MODEL_PATHS[2]+(model_path.split("/")[-1]),
                        enable_prefix_caching=True,
                        task="embed",
                        seed=0,
                    )
            print("FM - Loaded Model:", model_path)
            return model
        else:
           try:
               model = LLM(
                   config.MODEL_PATHS[0]+model_path,
                   enable_prefix_caching=True,
                   seed=0,
               )
           except:
               try:
                    model = LLM(
                        config.MODEL_PATHS[1]+(model_path.split("/")[-1]), 
                        enable_prefix_caching=True,
                        seed=0,
                    )
               except:
                   model = LLM(
                       config.MODEL_PATHS[2]+(model_path.split("/")[-1]),
                       enable_prefix_caching=True,
                       seed=0,
                   )
           print("FM - Loaded Model:", model_path)
        return model
    # elif is_unsloth:
    #     try:
    #         model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    #             config.MODEL_PATHS[0]+model_path, 
    #             load_in_4bit = False,
    #             load_in_8bit = False,
    #             full_finetuning=True, 
    #             max_seq_length=18000, 
    #             # device_map="balanced",
    #             # device_mesh=pc.build_device_mesh("cuda"), 
    #             # tp_plan="auto", 
    #             # use_cache=False,
    #         )
    #     except:
    #         try:
    #             model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    #                 config.MODEL_PATHS[1]+(model_path.split("/")[-1]), 
    #                 load_in_4bit = False,
    #                 load_in_8bit = False,
    #                 full_finetuning=True, 
    #                 max_seq_length=18000, 
    #                 # device_map="balanced",
    #                 # device_mesh=pc.build_device_mesh("cuda"), 
    #                 # tp_plan="auto", 
    #                 # use_cache=False,
    #             )
    #         except:
    #             model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    #                 config.MODEL_PATHS[2]+(model_path.split("/")[-1]), 
    #                 load_in_4bit = False,
    #                 load_in_8bit = False,
    #                 full_finetuning=True, 
    #                 max_seq_length=18000, 
    #                 # device_map="balanced",
    #                 # device_mesh=pc.build_device_mesh("cuda"), 
    #                 # tp_plan="auto", 
    #                 # use_cache=False,
    #             )
    #     print("FM - Loaded Model:", model_path)
    #     return model, tokenizer
    else:
        if "embed" in model_path.lower():
            try:
                model = SentenceTransformer(config.MODEL_PATHS[0]+model_path)
            except:
                try:
                    model = SentenceTransformer(config.MODEL_PATHS[1]+(model_path.split("/")[-1]))
                except:
                    model = SentenceTransformer(config.MODEL_PATHS[2]+(model_path.split("/")[-1]))
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[0]+model_path, padding_side='left')
                model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[0]+model_path, device_map="auto")
            except:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), padding_side='left')
                    model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[1]+(model_path.split("/")[-1]), device_map="auto")
                except:
                    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS[2]+(model_path.split("/")[-1]), padding_side='left')
                    model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATHS[2]+(model_path.split("/")[-1]), device_map="auto")
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


def to_chat_template_qwen_2_5(task, start_thinking):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    think = "<think>\n"*start_thinking
    return (lambda x : (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTIONS[task]}<|im_end|>\n"
        f"<|im_start|>user\n{USER_INSTRUCTIONS[task]}{x}<|im_end|>\n"
        f"<|im_start|>assistant\n{think}{language_forcing}"
    ))


def to_chat_template_qwen_3(task, start_thinking):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    think = "<think>\n"*(start_thinking or task.startswith("translation"))
    close_think = "\n</think>\n\n"*(task.startswith("translation"))
    return (lambda x : (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTIONS[task]}<|im_end|>\n"
        f"<|im_start|>user\n{USER_INSTRUCTIONS[task]}{x}<|im_end|>\n"
        f"<|im_start|>assistant\n{think}{language_forcing}{close_think}"
    ))


def to_chat_template_lucie(task, start_thinking):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    return (lambda x : (
        f"<s><|start_header_id|>system<|end_header_id|>You are a helpful assistant. {SYSTEM_INSTRUCTIONS[task]}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>{USER_INSTRUCTIONS[task]}{x}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>{language_forcing}"
    ))


def to_chat_template_phi4(task, start_thinking):
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


def to_chat_template_deepseek(task, start_thinking):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    start_thinking = "<think>\n"*("math" in task)
    return (lambda x : (
        f"{USER_INSTRUCTIONS[task]}{x}\n{SYSTEM_INSTRUCTIONS[task]}\n{start_thinking}{language_forcing}"
    ))


def to_chat_template_openai(task, start_thinking):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")

    return (lambda x : (
        f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Knowledge cutoff: 2024-06\nCurrent date: 2025-10-20\n\nReasoning: high\n\n"
        f"# Valid channels: analysis, final. Channel must be included for every message.<|end|>"
        f"<|start|>developer<|message|># Instructions\n\nreasoning language: French\n{SYSTEM_INSTRUCTIONS[task]}<|end|>"
        f"<|start|>user<|message|>{USER_INSTRUCTIONS[task]}{x}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>{language_forcing}"
    ))


def to_chat_template_mistral(task, start_thinking):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")

    return (lambda x : (
        f"First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.\n"
    ))

def to_chat_template_llama(task, start_thinking):
    return (lambda x : (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. {SYSTEM_INSTRUCTIONS[task]}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{USER_INSTRUCTIONS[task]}{x}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    ))

# no thinking yet and only math
def to_chat_template_luciole(task, start_thinking):
    language_forcing = "D'accord, laisse moi y réfléchir."*(task == "math_fr")
    think = "<think>\n"*start_thinking

    # Llama-nemotron-post-training v2 (post train) (0.2 GSM)
    return (lambda x : (
        f"<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n"
        f"<|im_start|>user\nSolve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{x}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    ))

    # Llama-nemotron-post-training v1 (0.05 GSM)
    # return (lambda x : (
    #     f"Question:\n{x}\nThoughts:\n"
    # ))

    # OpenMathInstruct (0.02 GSM)
    # return (lambda x : (
    #     f"{x}\n"
    # ))

    # Llama-nemotron-post-training v2 (0.11 GSM)
    # return (lambda x : (
    #     f"Question:\nSolve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{x}\n"
    # ))



def get_config(name, task="math", n=1, max_length=1000000, start_thinking=False):
    print("FM - Getting Config:", name, task)
    DEFAULT_CHAT_TEMPLATE = to_chat_template_qwen_2_5(task, start_thinking)
    DEFAULT_SAMPLING_PARAMS = SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=min(38192, max_length), seed=0)

    if name.startswith("fasttext"):
        return (
            f"facebook/fasttext-language-identification",   
            (lambda x: x),
            None,
        )
    elif name.startswith("Qwen2.5"):
        return (
            f"Qwen/{name}",
            to_chat_template_qwen_2_5(task, start_thinking),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name.startswith("Qwen3-Embedding"):
        return (
            f"Qwen/{name}",
            None,
            None
        )
    elif name.startswith("Qwen3"):
        return (
            f"Qwen/{name}",
            to_chat_template_qwen_3(task, start_thinking),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.5, max_tokens=min(38912, max_length), seed=0)
        )
    elif name.startswith("legml"):
        return (
            f"legmlai/{name}",
            to_chat_template_qwen_3(task, start_thinking),
            SamplingParams(n=n, temperature=0.6, top_p=0.9, max_tokens=min(38912, max_length), seed=0)
        )
    elif name.startswith("Lucie"):
        return (
            f"OpenLLM-France/{name}", 
            to_chat_template_lucie(task, start_thinking),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name.startswith("Phi-4"):
        return (
            f"microsoft/{name}", 
            to_chat_template_phi4(task, start_thinking),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("deepseek-math") or name.startswith("DeepSeek-R1"):
        return (
            f"deepseek-ai/{name}", 
            to_chat_template_deepseek(task, start_thinking),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("OpenR1"):
        return (
            f"open-r1/{name}", 
            to_chat_template_deepseek(task, start_thinking),
            SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("Pensez"):
        return (
            f"HoangHa/{name}", 
            to_chat_template_qwen_2_5(task, start_thinking),
            SamplingParams(n=n, temperature=0, max_tokens=min(32768, max_length), seed=0)
        )
    elif name.startswith("Llama"):
        return (
            f"meta-llama/{name}", 
            to_chat_template_llama(task, start_thinking),
            SamplingParams(n=n, temperature=0.7, top_p=0.95, max_tokens=min(38912, max_length), seed=0),
        )
    elif name.startswith("gpt-oss"):
        return (
            f"openai/{name}", 
            to_chat_template_openai(task, start_thinking),
            DEFAULT_SAMPLING_PARAMS
        )
    elif name.startswith("Magistral"):
        return (
            f"mistralai/{name}", 
            to_chat_template_mistral(task, start_thinking),
            SamplingParams(n=n, temperature=0.7, top_p=0.95, max_tokens=min(38912, max_length), seed=0),
        )
    elif "Luciole" in name:
        return (
            f"OpenLLM-France/{name}", 
            to_chat_template_luciole(task, start_thinking),
            SamplingParams(n=n, temperature=0.7, top_p=0.95, max_tokens=min(38912, max_length), seed=0),
        )
    else:
        raise Exception(f"Model {name} not supported. Edit utils_model.py to add support.")

def get_configs(names, task="math", n=1, max_length=1000000, start_thinking=False):
    configs = []
    for name in names:
        configs.append(get_config(name, task, n, max_length, start_thinking))
    return configs
