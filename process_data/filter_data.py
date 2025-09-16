import sys
import os
import re

from math_verify import parse, verify
from datasets import concatenate_datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
from process_data.extract_answer import *
from utils_model import *


def get_n_tokens(x):
    return config.TOKENIZER(x, return_length=True)["length"][0]


def filter_n_tokens(x, n_min, n_max):
    n_tokens = get_n_tokens(x)
    return n_tokens >= n_min and n_tokens <= n_max 


def similar_length(gold_length, pred_length, tol):
    return gold_length*(1+tol) >= pred_length and gold_length*(1-tol) <= pred_length


def filter_chinese(x, ratio=0.005):
    return (len(re.findall(r'[\u4e00-\u9fff]+', x))/len(x)) < ratio


def filter_boxed_format(x):
    n_boxed = x.count("\\boxed{")
    return n_boxed > 0 and n_boxed < 5


def filter_am_deepseek_distill(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["answer"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["ground_truth"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["answer"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["answer"]))

    # PPL too high
    dataset = dataset.filter(lambda x: x["ppl"] < 2.5)

    # Too hard
    dataset = dataset.filter(lambda x: x["pass_rate_r1"] > 0.025)

    # Invalid solution
    dataset = dataset.filter(lambda x: x["verify_score"] == 1)

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["answer"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_am_deepseek_r1_0528_distill(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["conversations"][0]["value"]) + x["conversations"][1]["value"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][0]["value"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][1]["value"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][1]["info"]["ground_truth"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["conversations"][1]["value"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["conversations"][1]["value"]))

    # PPL too high
    dataset = dataset.filter(lambda x: x["conversations"][1]["info"]["ppl"] < 3.25)

    # Invalid solution
    dataset = dataset.filter(lambda x: x["conversations"][1]["info"]["verify_score"] == 1)

    # Non-math question
    dataset = dataset.filter(lambda x: x["conversations"][0]["info"]["category"] == "math")

    # Answer in solution different (null)
    # dataset = dataset.filter(lambda x: verify(
    #     parse(to_latex(x["conversations"][1]["info"]["ground_truth"])),
    #     parse(x["conversations"][1]["value"])
    # ))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["conversations"][0]["value"]) + x["conversations"][1]["value"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_big_math(dataset):
    n_samples = dataset.num_rows

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 200))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 25))

    # Too hard
    dataset = dataset.filter(lambda x: x["llama8b_solve_rate"] is not None and x["llama8b_solve_rate"] > 0.025)

    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_deepmath(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 200))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["final_answer"], 0, 25))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["solution"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["solution"]))

    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(x["final_answer"])),
        parse(x["solution"])
    ))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_limo_v2(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))

    # Answer too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["solution"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["solution"]))

    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(x["answer"])),
        parse(x["solution"])
    ))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_limr(dataset):
    n_samples = dataset.num_rows

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["prompt"], 5, 384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 32))

    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_llama_nemotron(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["input"][0]["content"]) + x["output"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["input"][0]["content"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["output"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["output"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["output"]))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["input"][0]["content"]) + x["output"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_math_lvl5_fr_train(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["solution"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 512))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 30))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["solution"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["solution"]))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["solution"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_metamath_qa(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["query"]) + x["response"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["query"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["response"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["response"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["response"]))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["query"]) + x["response"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_nemotron_v1(dataset):
    n_samples = dataset.num_rows
    # # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["messages"][0]["content"]) + x["messages"][1]["content"]) for x in dataset])
    # # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["messages"][0]["content"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["messages"][1]["content"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(eval(x["metadata"])["expected_answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["messages"][1]["content"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["messages"][1]["content"]))

    # Non-math question
    dataset = dataset.filter(lambda x: x["category"] == "math")

    # Non-reasoning solution
    dataset = dataset.filter(lambda x: x["reasoning"] == "on")

    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(eval(x["metadata"])["expected_answer"])),
        parse(x["messages"][1]["content"])
    ))

    # # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["messages"][0]["content"]) + x["messages"][1]["content"]) for x in dataset])
    # # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_nemotron_v2(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["messages"][0]["content"]) + x["messages"][1]["content"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["messages"][0]["content"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["messages"][1]["content"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["metadata"]["expected_answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["messages"][1]["content"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["messages"][1]["content"]))

    # Non-math question
    dataset = dataset.filter(lambda x: x["category"] == "math")

    # Non-reasoning solution
    dataset = dataset.filter(lambda x: x["reasoning"] == "on")

    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(x["metadata"]["expected_answer"])),
        parse(x["messages"][1]["content"])
    ))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["messages"][0]["content"]) + x["messages"][1]["content"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_numinamath_1_5(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["solution"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 512))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["solution"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["solution"]))

    # Invalid problem
    dataset = dataset.filter(lambda x: x["problem_is_valid"] == "Yes")

    # Invalid solution
    dataset = dataset.filter(lambda x: x["solution_is_valid"] == "Yes")

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["solution"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_open_math_reasoning(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["generated_solution"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["generated_solution"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["expected_answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["generated_solution"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["generated_solution"]))

    # CoT format
    dataset = dataset.filter(lambda x: x["inference_mode"] == "cot")

    # Answer extracted
    dataset = dataset.filter(lambda x: x["problem_type"] == "has_answer_extracted")


    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(x["expected_answer"])),
        parse(x["generated_solution"])
    ))

    # Too hard
    dataset = dataset.filter(lambda x: x["pass_rate_72b_tir"] == "n/a" or float(x["pass_rate_72b_tir"]) > 0.0)

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["generated_solution"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_open_r1_math(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["generations"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["generations"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["generations"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["generations"]))

    # Reasoning not complete
    dataset = dataset.filter(lambda x: x["is_reasoning_complete"] != False)
    dataset = dataset.filter(lambda x: x["finish_reasons"] is None)

    # Invalid solution
    dataset = dataset.filter(lambda x: x["correctness_math_verify"] != False)
    dataset = dataset.filter(lambda x: x["correctness_llama"] != False)

    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(x["answer"])),
        parse(x["generations"])
    ))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["problem"]) + x["generations"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_open_thoughts_2(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["conversations"][0]["value"]) + x["conversations"][1]["value"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][0]["value"], 5, 512))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][1]["value"], 0, 16384))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["conversations"][1]["value"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["conversations"][1]["value"]))
    
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["conversations"][0]["value"]) + x["conversations"][1]["value"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_open_thoughts_3(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["conversations"][0]["value"]) + x["conversations"][1]["value"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][0]["value"], 5, 512))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 256))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][1]["value"], 0, 16384))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["conversations"][1]["value"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["conversations"][1]["value"]))

    # Answer in solution different
    # dataset = dataset.filter(lambda x: verify(
    #     parse(to_latex(x["answer"])),
    #     parse(x["conversations"][1]["value"])
    # ))

    # Non-math question
    dataset = dataset.filter(lambda x: x["domain"] == "math")
    
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["conversations"][0]["value"]) + x["conversations"][1]["value"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_s1k_1_1(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["deepseek_thinking_trajectory"] + x["deepseek_attempt"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 512))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["deepseek_thinking_trajectory"] + x["deepseek_attempt"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 1024))

    # Solution contains Chinese
    dataset = dataset.filter(lambda x: filter_chinese(x["deepseek_thinking_trajectory"] + x["deepseek_attempt"]))

    # Solution well formated
    dataset = dataset.filter(lambda x: filter_boxed_format(x["deepseek_thinking_trajectory"] + x["deepseek_attempt"]))

    # Invalid solution
    dataset = dataset.filter(lambda x: x["deepseek_grade"] == "Yes")

    # Non-Math data
    dataset = dataset.filter(lambda x: x["cot_type"] == "math")

    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(x["solution"])),
        parse(x["deepseek_thinking_trajectory"] + x["deepseek_attempt"])
    ))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["deepseek_thinking_trajectory"] + x["deepseek_attempt"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_train_math_fr(dataset, rl):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 512))

    # Solution too long
    if not rl:
        dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 512))

    # Failed inference
    if not rl:
        dataset = dataset.filter(lambda x: x["solution"][-11:] != "<DISCARDED>")

    # Invalid solution
    if not rl:
        dataset = dataset.filter(lambda x: x["valid"])

    # Translation length don't match
    dataset = dataset.filter(lambda x: (similar_length(get_n_tokens(x["question_en"]), get_n_tokens(x["question"]), 0.5)))
    if not rl:
        dataset = dataset.filter(lambda x: (similar_length(get_n_tokens(x["solution_en"]), get_n_tokens(x["solution"]), 0.5)))
    dataset = dataset.filter(lambda x: (similar_length(get_n_tokens(x["answer_en"]), get_n_tokens(x["answer"]), 0.2)))

    # Not French
    if rl:
        dataset = dataset.filter(lambda x : x["answer_fr_lang"][0] == "__label__fra_Latn" and x["answer_fr_lang_prob"][0] > 0.9)
    else:
        dataset = dataset.filter(lambda x : x["solution_fr_lang"][0] == "__label__fra_Latn" and x["solution_fr_lang_prob"][0] > 0.98)

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_train_math_en(dataset):
    n_samples = dataset.num_rows
    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens before filtering: {n_tokens/1000000000}B")

    # Question too short/long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 512))

    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))

    # Answer too long
    # dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 512))

    # n_tokens = sum([get_n_tokens(temp_chat_template_fun(config.TOKENIZER, x["question"]) + x["solution"]) for x in dataset])
    # print(f"Tokens after filtering: {n_tokens/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset
