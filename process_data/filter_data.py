import config

def filter_n_tokens(x, n_min, n_max):
    n_tokens = config.TOKENIZER(x, return_length=True)["length"][0]
    return n_tokens >= n_min and n_tokens <= n_max 


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