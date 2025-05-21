import json
from datetime import datetime

from datasets import load_dataset, Dataset, Value
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from process_data.utils_data import *
from process_data.filter_data import *


# LOAD AND FILTER DATASETS ----------------------------------------------------------------------
am_deepseek_distill = load_data("AM-DeepSeek-Distilled-40M").select(range(10000))
am_deepseek_distill = filter_am_deepseek_distill(am_deepseek_distill)

big_math = load_data("Big-Math-RL-Verified").select(range(10000))
big_math = filter_big_math(big_math)

limo = load_data("LIMO")
limo = filter_limo(limo)

limr = load_data("LIMR")
limr = filter_limr(limr)

llama_nemotron = load_data("Llama-Nemotron-Post-Training-Dataset").select(range(10000))
llama_nemotron = llama_nemotron.add_column("answer", extract_boxed_text(llama_nemotron["output"]))
llama_nemotron = filter_llama_nemotron(llama_nemotron)

math_lvl5_fr_train = load_data("MATH_LVL5_fr")
math_lvl5_fr_test = load_data("MATH_LVL5_fr", split="test")
math_lvl5_fr_train = math_lvl5_fr_train.add_column("answer", extract_boxed_text(math_lvl5_fr_train["solution"]))
math_lvl5_fr_test = math_lvl5_fr_test.add_column("answer", extract_boxed_text(math_lvl5_fr_test["solution"]))
math_lvl5_fr_train = filter_math_lvl5_fr_train(math_lvl5_fr_train)

mclm = {"question": [], "answer": [], "source": [],}
for subset in ["m-imo", "mt-aime2024", "mt-math100"]:
    mclm_subset = load_data("MCLM", data_files=subset+".parquet")
    mclm_subset = mclm_subset.remove_columns(set(mclm_subset.features.keys()) - set(["fr", "answer"]))
    mclm_subset = mclm_subset.cast_column("answer", Value(dtype="string"))
    mclm["question"].extend(mclm_subset["fr"])
    mclm["answer"].extend(mclm_subset["answer"])
    mclm["source"].extend([subset] * len(mclm_subset["fr"]))
mclm = Dataset.from_dict(mclm)

mgsm = Dataset.from_pandas(pd.read_csv(
    config.DATA_PATHS[1]+"MGSM/mgsm_fr.tsv", sep="\t", header=None, names=["question", "answer"]
))

msvamp = load_data("MSVAMP", split="test")

megamath_web_pro = load_data("MegaMath").select(range(10000))
megamath_web_pro = megamath_web_pro.add_column(
    "timestamp_unformatted",
    [(datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ') - datetime(1970, 1, 1)).total_seconds() for ts in megamath_web_pro["timestamp"]]
)

metamath_qa = load_data("MetaMathQA").select(range(10000))
metamath_qa = metamath_qa.add_column(
    "answer",
    [x.split("The answer is: ")[-1] for x in metamath_qa["response"]],
)
metamath_qa = filter_metamath_qa(metamath_qa)

numinamath_1_5 = load_data("NuminaMath-1.5").select(range(10000))
numinamath_1_5 = filter_numinamath_1_5(numinamath_1_5)

open_r1_math = load_data("OpenR1-Math-220k").select(range(1000))
open_r1_math = flatten_features(open_r1_math, ['generations', 'is_reasoning_complete', 'correctness_math_verify', 'correctness_llama', 'finish_reasons'])
open_r1_math = filter_open_r1_math(open_r1_math)

open_thoughts_2 = load_data("OpenThoughts2-1M").select(range(10000))
open_thoughts_2 = filter_open_thoughts_2(open_thoughts_2)

#TODO: PENSEZ

polymath = load_data("PolyMath")

s1k_1_1 = load_data("s1K-1.1")
s1k_1_1 = filter_s1k_1_1(s1k_1_1)

swallowmath = load_data("swallow-math").select(range(10000))

# FUSION OF DATASETS ----------------------------------------------------------------------
cot_datasets = [
    {
        "name": "am-deepseek-distill",
        "dataset": am_deepseek_distill,
        "question": am_deepseek_distill["question"],
        "answer": am_deepseek_distill["ground_truth"],
        "solution": am_deepseek_distill["answer"],
        "source": ["am-deepseek-distill/" + source for source in am_deepseek_distill["question_source"]],
        "model": am_deepseek_distill["model_name"],
    },
    {
        "name": "big-math",
        "dataset": big_math,
        "question": big_math["problem"],
        "answer": big_math["answer"],
        "solution": [None] * len(big_math),
        "source": ["big-math/" + source for source in big_math["source"]],
        "model": [None] * len(big_math),
    },
    {
        "name": "limo",
        "dataset": limo,
        "question": limo["question"],
        "answer": limo["answer"],
        "solution": limo["solution"],
        "source": ["limo/NuminaMath-CoT or AIME or MATH or other"] * len(limo),
        "model": ["DeepSeek R1 or DeepSeek-R1-Distill-Qwen-32B or Qwen2.5-32b-Instruct or other"] * len(limo)
    },
    {
        "name": "limr",
        "dataset": limr,
        "question": limr["prompt"],
        "answer": limr["answer"],
        "solution": [None] * len(limr),
        "source": ["limr/" + source for source in limr["source"]],
        "model": [None] * len(limr)
    },
    {
        "name": "llama-nemotron",
        "dataset": llama_nemotron,
        "question": [x[0]["content"] for x in llama_nemotron["input"]],
        "answer": llama_nemotron["answer"],
        "solution": llama_nemotron["output"],
        "source": ["llama-nemotron/AoPS"] * len(llama_nemotron),
        "model": llama_nemotron["generator"],
    },
    {
        "name": "math-lvl5-fr",
        "dataset": math_lvl5_fr_train,
        "question": math_lvl5_fr_train["problem"],
        "answer": math_lvl5_fr_train["answer"],
        "solution": math_lvl5_fr_train["solution"],
        "source": ["math-lvl5-fr/math"] * len(math_lvl5_fr_train),
        "model": [None] * len(math_lvl5_fr_train),
    },
    {
        "name": "metamath-qa",
        "dataset": metamath_qa,
        "question": metamath_qa["query"],
        "answer": metamath_qa["answer"],
        "solution": metamath_qa["response"],
        "source": ["metamath-qa/" + subset for subset in metamath_qa["type"]],
        "model": ["unknown"] * len(metamath_qa),
    },
    {
        "name": "numinamath-1.5",
        "dataset": numinamath_1_5,
        "question": numinamath_1_5["problem"],
        "answer": numinamath_1_5["answer"],
        "solution": numinamath_1_5["solution"],
        "source": ["numina-math-1.5/" + source for source in numinamath_1_5["source"]],
        "model": ["unknown"] * len(numinamath_1_5),
    },
    {
        "name": "open-r1-math",
        "dataset": open_r1_math,
        "question": open_r1_math["problem"],
        "answer": open_r1_math["answer"],
        "solution": open_r1_math["generations"],
        "source": ["open-r1-math/" + source for source in open_r1_math["source"]],
        "model": ["open-r1"] * len(open_r1_math),
    },
    {
        "name": "open-thoughts-2",
        "dataset": open_thoughts_2,
        "question": [x[0]["value"] for x in open_thoughts_2["conversations"]],
        "answer": [None] * len(open_thoughts_2),
        "solution": [x[1]["value"] for x in open_thoughts_2["conversations"]],
        "source": ["open-thoughts-2/" + source if source is not None else None for source in open_thoughts_2["source"]],
        "model": ["unknown"] * len(open_thoughts_2),
    },
    {
        "name": "s1k-1.1",
        "dataset": s1k_1_1,
        "question": s1k_1_1["question"],
        "answer": s1k_1_1["solution"],
        "solution": s1k_1_1["deepseek_thinking_trajectory"],
        "source": ["s1k-1.1/" + source for source in s1k_1_1["source_type"]],
        "model": ["deepseek-r1"] * len(s1k_1_1),
    },
]
cot_dataset = fusion_datasets(cot_datasets)
cot_dataset.save_to_disk(config.DATA_PATHS[1] + "Fused-CoT")

eval_datasets = [
    {
        "name": "math-lvl5-fr",
        "dataset": math_lvl5_fr_test,
        "question": math_lvl5_fr_test["problem"],
        "answer": math_lvl5_fr_test["answer"],
        "source": ["math-lvl5-fr"] * len(math_lvl5_fr_test),
    },
    {
        "name": "mclm",
        "dataset": mclm,
        "question": mclm["question"],
        "answer": mclm["answer"],
        "source": ["mclm/" + source for source in mclm["source"]],
    },
    {
        "name": "mgsm",
        "dataset": mgsm,
        "question": mgsm["question"],
        "answer": mgsm["answer"],
        "source": ["mgsm"] * len(mgsm),
    },
    {
        "name": "msvamp",
        "dataset": msvamp,
        "question": msvamp["m_query"],
        "answer": msvamp["response"],
        "source": ["msvamp"] * len(msvamp),
    },
    {
        "name": "polymath",
        "dataset": polymath,
        "question": polymath["question"],
        "answer": polymath["answer"],
        "source": ["polymath/" + (identifier).split("-")[0] for identifier in polymath["id"]],
    }
]
eval_dataset = fusion_datasets(eval_datasets)
eval_dataset.save_to_disk(config.DATA_PATHS[1] + "Eval-Math-FR")
