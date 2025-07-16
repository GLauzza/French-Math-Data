import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

def get_n_tokens(x):
    return config.TOKENIZER(x, return_length=True)["length"][0]

def filter_n_tokens(x, n_min, n_max):
    n_tokens = get_n_tokens(x)
    return n_tokens >= n_min and n_tokens <= n_max 

def similar_length(n1, n2, tol):
    return n1 + n1*tol > n2 - n2*tol or n2 + n2*tol > n1 - n1*tol


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
    dataset = dataset.filter(lambda x: filter_n_tokens(x["r1_solution_1"], 0, 16384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["final_answer"], 0, 25))
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


def filter_train_math_fr(dataset):
    n_samples = dataset.num_rows
    # Question too short/long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["question"], 5, 512))
    # Solution too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["solution"], 0, 16384))
    # Answer too long
    dataset = dataset.filter(lambda x: filter_n_tokens(x["answer"], 0, 512))
    # Invalid solution
    dataset = dataset.filter(lambda x: x["valid"])
    # Not French
    dataset = dataset.filter(lambda x : x["solution_fr_lang"][0] == "__label__fra_Latn" and x["solution_fr_lang_prob"][0] > 0.98)
    # Translation length don't match
    dataset = dataset.filter(lambda x: (
        similar_length(get_n_tokens(x["question"]), get_n_tokens(x["question_en"]), 0.2) and
        similar_length(get_n_tokens(x["solution"]), get_n_tokens(x["solution_en"]), 0.2) and
        similar_length(get_n_tokens(x["answer"]), get_n_tokens(x["answer_en"]), 0.2)
    ))
    print(f"Filtered {100 * (n_samples - dataset.num_rows) / n_samples}% of the dataset")
    return dataset
