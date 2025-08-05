import sys
import os

from math_verify import parse, verify

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
from process_data.extract_answer import *


def get_n_tokens(x):
    return config.TOKENIZER(x, return_length=True)["length"][0]


def filter_n_tokens(x, n_min, n_max):
    n_tokens = get_n_tokens(x)
    return n_tokens >= n_min and n_tokens <= n_max 


def similar_length(gold_length, pred_length, tol):
    return gold_length*(1+tol) >= pred_length and gold_length*(1-tol) <= pred_length


def filter_am_deepseek_distill(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 256))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 16384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["ground_truth"], 0, 50))
    # PPL too high
    dataset = dataset.filter(lambda x: x["ppl"] < 2.5)
    # Too hard
    dataset = dataset.filter(lambda x: x["pass_rate_r1"] > 0.025)
    # Invalid solution
    dataset = dataset.filter(lambda x: x["verify_score"] == 1)
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_big_math(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 200))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 25))
    # Too hard
    dataset = dataset.filter(lambda x: x["llama8b_solve_rate"] is not None and x["llama8b_solve_rate"] > 0.025)
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_deepmath(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 200))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 8192))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["final_answer"], 0, 25))
    # Answer in solution different
    dataset = dataset.filter(lambda x: verify(
        parse(to_latex(x["final_answer"])),
        parse(to_latex(extract_boxed_text(x["solution"])))
    ))
    print(f"Total tokens: {sum([get_n_tokens(sample['solution']) for sample in dataset])/1000000000}B")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_limo(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 256))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 16384))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_limr(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["prompt"], 5, 384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 32))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_llama_nemotron(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["input"][0]["content"], 5, 256))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["output"], 0, 16384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_math_lvl5_fr_train(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 512))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 2048))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 30))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_metamath_qa(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["query"], 5, 256))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["response"], 0, 1024))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_numinamath_1_5(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 512))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 2048))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))
    # Invalid problem
    dataset = dataset.filter(lambda x: x["problem_is_valid"] == "Yes")
    # Invalid solution
    dataset = dataset.filter(lambda x: x["solution_is_valid"] == "Yes")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_open_r1_math(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["problem"], 5, 256))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["generations"], 0, 16384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 50))
    # Reasoning not complete
    dataset = dataset.filter(lambda x: x["is_reasoning_complete"] != False)
    # Invalid solution
    dataset = dataset.filter(lambda x: x["correctness_math_verify"] != False)
    dataset = dataset.filter(lambda x: x["correctness_llama"] != False)
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_open_thoughts_2(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][0]["value"], 5, 512))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["conversations"][1]["value"], 0, 16384))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_s1k_1_1(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 512))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["deepseek_thinking_trajectory"], 0, 17500))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 1024))
    # Invalid solution
    dataset = dataset.filter(lambda x: x["deepseek_grade"] == "Yes")
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset


def filter_train_math_fr(dataset, rl):
    n_samples = dataset.num_rows
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
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 512))
    # Solution too long
    if not rl:
        dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 512))
    # Not French
    if rl:
        dataset = dataset.filter(lambda x : x["answer_fr_lang"][0] == "__label__fra_Latn" and x["answer_fr_lang_prob"][0] > 0.9)
    else:
        dataset = dataset.filter(lambda x : x["solution_fr_lang"][0] == "__label__fra_Latn" and x["solution_fr_lang_prob"][0] > 0.98)
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    if not rl:
        total_tokens = 0
        for sample in dataset:
            total_tokens += get_n_tokens(
                f"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>user\n{sample['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{sample['solution']}"
            )
        print(f"Total tokens: {total_tokens/1000000000}B")
    return dataset


def filter_train_math_en(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 512))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 512))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset
