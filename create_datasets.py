import json
from datetime import datetime
import argparse
import re

from datasets import load_dataset, Dataset, Value
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math_verify import parse, verify

from process_data.utils_data import *
from process_data.prepare_data import *
from process_data.filter_data import *
from process_data.extract_answer import *

def remove_instruct_translation(x):
    solution_begin = x["solution"][:500].lower()
    return not (("texte" in solution_begin) and ("trad" in solution_begin))
#        x["solution"] = re.split(r'texte[\s|>]*', solution, flags=re.IGNORECASE, maxsplit=1)[1]
#    return x


if __name__ == "__main__":
    default_datasets = [
        "Fused-CoT"
        "Eval-Math-FR",
    ]
    parser = argparse.ArgumentParser(description='Create a dataset')
    parser.add_argument('--datasets', nargs='+', default=default_datasets, help="Datasets to create")
    parser.add_argument('--model', type=str, default="", help="Which model solution to use for training dataset")
    parser.add_argument('--rl', default=False, action="store_true")
    parser.add_argument('--fr_ratio', type=float, default=0.5, help="Percentage of french for train dataset")
    parser.add_argument('--n', type=int, default=-1, help="Number of samples in the dataset")
    args = parser.parse_args()

    if "Fused-CoT" in args.datasets:    
        # am_deepseek_distill = load_data("a-m-team/AM-DeepSeek-Distilled-40M", data_files="math_r1_*.jsonl")
        # am_deepseek_distill = filter_am_deepseek_distill(am_deepseek_distill)

        am_deepseek_r1_0528_distill = load_data("a-m-team/AM-DeepSeek-R1-0528-Distilled").shuffle().select(range(100000))
        am_deepseek_r1_0528_distill = am_deepseek_r1_0528_distill.add_column(
            "answer",
            [extract_boxed_text(x["conversations"][1]["value"]) for x in am_deepseek_r1_0528_distill]
        )
        am_deepseek_r1_0528_distill = filter_am_deepseek_r1_0528_distill(am_deepseek_r1_0528_distill)

        # big_math = load_data("SynthLabsAI/Big-Math-RL-Verified")
        # big_math = filter_big_math(big_math)

        deepmath = load_data("zwhe99/DeepMath-103K").shuffle().select(range(100000))
        deepmath_solutions = list(deepmath["r1_solution_1"]) + list(deepmath["r1_solution_2"]) + list(deepmath["r1_solution_3"])
        deepmath = concatenate_datasets([deepmath] * 3).add_column(
            "solution",
            deepmath_solutions
        ).shuffle().select(range(100000))
        deepmath = filter_deepmath(deepmath)

        limo_v2 = load_data("GAIR/LIMO-v2")
        limo_v2 = filter_limo_v2(limo_v2)

        # limr = load_data("GAIR/LIMR")
        # limr = filter_limr(limr)

        # llama_nemotron = load_data("nvidia/Llama-Nemotron-Post-Training-Dataset", data_files="SFT/math/math_v1.1.jsonl")
        # llama_nemotron = llama_nemotron.add_column(
        #     "answer",
        #     [extract_boxed_text(x) for x in llama_nemotron["output"]]
        # )
        # llama_nemotron = filter_llama_nemotron(llama_nemotron)
        
        # math_lvl5_fr_train = load_data("le-leadboard/MATH_LVL5_fr")
        # math_lvl5_fr_train = math_lvl5_fr_train.add_column(
        #     "answer",
        #     [extract_boxed_text(x) for x in math_lvl5_fr_train["solution"]]
        # )
        # math_lvl5_fr_train = filter_math_lvl5_fr_train(math_lvl5_fr_train)

        # metamath_qa = load_data("meta-math/MetaMathQA")
        # metamath_qa = metamath_qa.add_column(
        #     "answer",
        #     [x.split("The answer is: ")[-1] for x in metamath_qa["response"]],
        # )
        # metamath_qa = filter_metamath_qa(metamath_qa)

        nemotron_v1 = load_data("nvidia/Nemotron-Post-Training-Dataset-v1", split="math").shuffle().select(range(100000))
        nemotron_v1 = filter_nemotron_v1(nemotron_v1)

        # # nemotron_v2 = load_data("nvidia/Nemotron-Post-Training-Dataset-v2")
        # # nemotron_v2 = filter_nemotron_v2(nemotron_v2)

        # # numinamath_1_5 = load_data("AI-MO/NuminaMath-1.5")
        # # numinamath_1_5 = filter_numinamath_1_5(numinamath_1_5)

        open_math_reasoning = load_data("nvidia/OpenMathReasoning", split="cot").shuffle().select(range(100000))
        open_math_reasoning = filter_open_math_reasoning(open_math_reasoning)

        open_r1_math = load_data("open-r1/OpenR1-Math-220k").shuffle().select(range(50000))
        open_r1_math = flatten_features(open_r1_math, ['generations', 'is_reasoning_complete', 'correctness_math_verify', 'correctness_llama', 'finish_reasons']).shuffle().select(range(100000))
        open_r1_math = filter_open_r1_math(open_r1_math)

        # open_thoughts_2 = load_data("open-thoughts/OpenThoughts2-1M", data_files="data/*.parquet")
        # open_thoughts_2 = filter_open_thoughts_2(open_thoughts_2)
        
        open_thoughts_3 = load_data("open-thoughts/OpenThoughts3-1.2M").shuffle().select(range(100000))
        open_thoughts_3 = open_thoughts_3.add_column(
            "answer",
            [extract_boxed_text(x["conversations"][1]["value"]) for x in open_thoughts_3]
        )
        open_thoughts_3 = filter_open_thoughts_3(open_thoughts_3)

        #TODO: PENSEZ

        s1k_1_1 = load_data("simplescaling/s1K-1.1")
        s1k_1_1 = filter_s1k_1_1(s1k_1_1)
        
        # open_thoughts_3_100k = load_data("mlfoundations-dev/openthoughts3_math_100k").shuffle()
        # open_thoughts_3_100k = open_thoughts_3_100k.add_column(
        #     "answer",
        #     [extract_boxed_text(x["final_reasoning_trace"]) for x in open_thoughts_3_100k]
        # )
        # open_thoughts_3_100k = filter_open_thoughts_3_100k(open_thoughts_3_100k)

        cot_datasets = [
            # {
            #     "name": "am-deepseek-distill",
            #     "dataset": am_deepseek_distill,
            #     "question": am_deepseek_distill["question"],
            #     "answer": am_deepseek_distill["ground_truth"],
            #     "solution": am_deepseek_distill["answer"],
            #     "source": ["am-deepseek-distill/" + source for source in am_deepseek_distill["question_source"]],
            #     "model": am_deepseek_distill["model_name"],
            # },
            {
                "name": "am-deepseek-r1-0528-distill",
                "dataset": am_deepseek_r1_0528_distill,
                "question": [x["conversations"][0]["value"] for x in am_deepseek_r1_0528_distill],
                "answer": am_deepseek_r1_0528_distill["answer"],
                "solution": [x["conversations"][1]["value"] for x in am_deepseek_r1_0528_distill],
                "source": ["am-deepseek-r1-0528-distill/" + x["conversations"][0]["info"]["source"] for x in am_deepseek_r1_0528_distill],
                "model": [x["conversations"][1]["info"]["model_name"] for x in am_deepseek_r1_0528_distill],
            },
            # {
            #     "name": "big-math",
            #     "dataset": big_math,
            #     "question": big_math["problem"],
            #     "answer": big_math["answer"],
            #     "solution": [None] * len(big_math),
            #     "source": ["big-math/" + source for source in big_math["source"]],
            #     "model": [None] * len(big_math),
            # },
            {
                "name": "deepmath",
                "dataset": deepmath,
                "question": deepmath["question"],
                "answer": deepmath["final_answer"],
                "solution": deepmath["solution"],
                "source": ["deepmath/(MMIQC or WebInstructSub or NuminaMath-CoT)"] * len(deepmath),
                "model": ["deepseek-r1"] * len(deepmath),
            },
            {
                "name": "limo_v2",
                "dataset": limo_v2,
                "question": limo_v2["question"],
                "answer": limo_v2["answer"],
                "solution": limo_v2["solution"],
                "source": ["limo_v2/(NuminaMath-CoT or DeepScaleR or AIME or MATH)"] * len(limo_v2),
                "model": ["DeepSeek R1 or DeepSeek-R1-Distill-Qwen-32B or QwQ-32b"] * len(limo_v2)
            },
            # {
            #     "name": "limr",
            #     "dataset": limr,
            #     "question": limr["prompt"],
            #     "answer": limr["answer"],
            #     "solution": [None] * len(limr),
            #     "source": ["limr/" + source for source in limr["source"]],
            #     "model": [None] * len(limr)
            # },
            # {
            #     "name": "llama-nemotron",
            #     "dataset": llama_nemotron,
            #     "question": [x[0]["content"] for x in llama_nemotron["input"]],
            #     "answer": llama_nemotron["answer"],
            #     "solution": llama_nemotron["output"],
            #     "source": ["llama-nemotron/AoPS"] * len(llama_nemotron),
            #     "model": llama_nemotron["generator"],
            # },
            # {
            #     "name": "math-lvl5-fr",
            #     "dataset": math_lvl5_fr_train,
            #     "question": math_lvl5_fr_train["problem"],
            #     "answer": math_lvl5_fr_train["answer"],
            #     "solution": math_lvl5_fr_train["solution"],
            #     "source": ["math-lvl5-fr/math"] * len(math_lvl5_fr_train),
            #     "model": [None] * len(math_lvl5_fr_train),
            # },
            # {
            #     "name": "metamath-qa",
            #     "dataset": metamath_qa,
            #     "question": metamath_qa["query"],
            #     "answer": metamath_qa["answer"],
            #     "solution": metamath_qa["response"],
            #     "source": ["metamath-qa/" + subset for subset in metamath_qa["type"]],
            #     "model": ["unknown"] * len(metamath_qa),
            # },
            {
                "name": "nemotron-v1",
                "dataset": nemotron_v1,
                "question": [x["messages"][0]["content"] for x in nemotron_v1],
                "answer": [eval(x["metadata"])["expected_answer"] for x in nemotron_v1],
                "solution": [x["messages"][1]["content"] for x in nemotron_v1],
                "source": ["nemotron-v1/" + eval(x["metadata"])["problem_source"] for x in nemotron_v1],
                "model": nemotron_v1["generator"],
            },
            # {
            #     "name": "numinamath-1.5",
            #     "dataset": numinamath_1_5,
            #     "question": numinamath_1_5["problem"],
            #     "answer": numinamath_1_5["answer"],
            #     "solution": numinamath_1_5["solution"],
            #     "source": ["numina-math-1.5/" + source for source in numinamath_1_5["source"]],
            #     "model": ["unknown"] * len(numinamath_1_5),
            # },
            {
                "name": "open-math-reasoning",
                "dataset": open_math_reasoning,
                "question": open_math_reasoning["problem"],
                "answer": open_math_reasoning["expected_answer"],
                "solution": open_math_reasoning["generated_solution"],
                "source": ["open-math-reasoning/" + source for source in open_math_reasoning["problem_source"]],
                "model": open_math_reasoning["generation_model"],
            },
            {
                "name": "open-r1-math",
                "dataset": open_r1_math,
                "question": open_r1_math["problem"],
                "answer": open_r1_math["answer"],
                "solution": open_r1_math["generations"],
                "source": ["open-r1-math/" + source for source in open_r1_math["source"]],
                "model": ["deepseek-r1"] * len(open_r1_math),
            },
            # {
            #     "name": "open-thoughts-2",
            #     "dataset": open_thoughts_2,
            #     "question": [x[0]["value"] for x in open_thoughts_2["conversations"]],
            #     "answer": [None] * len(open_thoughts_2),
            #     "solution": [x[1]["value"] for x in open_thoughts_2["conversations"]],
            #     "source": ["open-thoughts-2/" + source if source is not None else None for source in open_thoughts_2["source"]],
            #     "model": ["unknown"] * len(open_thoughts_2),
            # },
            {
                "name": "open-thoughts-3",
                "dataset": open_thoughts_3,
                "question": [x["conversations"][0]["value"] for x in open_thoughts_3],
                "answer": open_thoughts_3["answer"],
                "solution": [x["conversations"][1]["value"] for x in open_thoughts_3],
                "source": ["open-thoughts-3/" + source if source is not None else None for source in open_thoughts_3["source"]],
                "model": ["unknown"] * len(open_thoughts_3),
            },
            {
                "name": "s1k-1.1",
                "dataset": s1k_1_1,
                "question": s1k_1_1["question"],
                "answer": s1k_1_1["solution"],
                "solution": ["<think>" + thinking + "\n</think>\n<answer>\n" + attempt + "</answer>" for thinking, attempt in zip(s1k_1_1["deepseek_thinking_trajectory"], s1k_1_1["deepseek_attempt"])],
                "source": ["s1k-1.1/" + source for source in s1k_1_1["source_type"]],
                "model": ["deepseek-r1"] * len(s1k_1_1),
            },
        #     {
        #         "name": "open_thoughts_3_100k",
        #         "dataset": open_thoughts_3_100k,
        #         "question": open_thoughts_3_100k["instruction_seed"],
        #         "answer": open_thoughts_3_100k["answer"],
        #         "solution":open_thoughts_3_100k["final_reasoning_trace"],
        #         "source": ["open_thoughts_3_100k/" + source for source in open_thoughts_3_100k["_source"]],
        #         "model": ["QwQ-32b"] * len(open_thoughts_3_100k),
        #     },
        ]
        fused_cot = fusion_datasets(cot_datasets)
        fused_cot = fused_cot.filter(lambda x: x["solution"] is not None).shuffle(seed=0)
        if args.n != -1:
            fused_cot = fused_cot.select(range(args.n))

        fused_cot_en_big = load_data("me/Train-Math-en-big")
        already_translated = set()
        for example in fused_cot_en_big:
            if isinstance(example, dict):
                example_tuple = tuple(sorted(example.items()))
            else:
                example_tuple = example
            already_translated.add(example_tuple)

        def not_in_b_full(example):
            if isinstance(example, dict):
                example_tuple = tuple(sorted(example.items()))
            else:
                example_tuple = example
            return (example_tuple not in already_translated) or ("s1k-1.1" in example["source"]) or ("limo_v2" in example["source"])

        print("len before", len(fused_cot))
        fused_cot = fused_cot.filter(not_in_b_full)
        print("len after", len(fused_cot))

        fused_cot.save_to_disk(config.DATA_PATHS[1] + "Fused-CoT")
        fused_cot.save_to_disk(config.DATA_PATHS[2] + "Fused-CoT")

    if "Eval-Math-FR" in args.datasets:
        math_lvl5_fr_test = load_data("le-leadboard/MATH_LVL5_fr", split="test")
        math_lvl5_fr_test = math_lvl5_fr_test.add_column(
            "answer",
            [extract_boxed_text(x) for x in math_lvl5_fr_test["solution"]]
        )

        mclm = {"question": [], "answer": [], "source": [],}
        for subset in ["m-imo", "mt-aime2024", "mt-math100"]:
            mclm_subset = load_data("amphora/MCLM", data_files=subset+".parquet")
            mclm_subset = mclm_subset.remove_columns(set(mclm_subset.features.keys()) - set(["fr", "answer"]))
            mclm_subset = mclm_subset.cast_column("answer", Value(dtype="string"))
            mclm["question"].extend(mclm_subset["fr"])
            mclm["answer"].extend(mclm_subset["answer"])
            mclm["source"].extend([subset] * len(mclm_subset["fr"]))
        mclm = Dataset.from_dict(mclm)

        mgsm = Dataset.from_pandas(pd.read_csv(
            config.DATA_PATHS[2]+"MGSM/mgsm_fr.tsv", sep="\t", header=None, names=["question", "answer"]
        ))

        msvamp = load_data("Mathoctopus/MSVAMP", split="test")

        polymath = load_data("Qwen/PolyMath")

        eval_math_fr_datasets = [
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
        eval_math_fr = fusion_datasets(eval_math_fr_datasets).shuffle(seed=0)
        if args.n != -1:
            eval_math_fr = eval_math_fr.select(range(args.n))
        eval_math_fr.save_to_disk(config.DATA_PATHS[1] + "Eval-Math-FR")
        eval_math_fr.save_to_disk(config.DATA_PATHS[2] + "Eval-Math-FR")

    if "Eval-Math-EN" in args.datasets:
            aime2024 = load_data("HuggingFaceH4/aime_2024")
            aime2025 = load_data("yentinglin/aime_2025", data_files="data/*.parquet")
            amc23 = load_data("knoveleng/AMC-23")
            hmmt = load_data("MathArena/hmmt_feb_2025")
            math500 = load_data("HuggingFaceH4/MATH-500", split="test")

            eval_math_en_datasets = [
                {
                    "name": "aime2024",
                    "dataset": aime2024,
                    "question": aime2024["problem"],
                    "answer": aime2024["answer"],
                    "source": ["aime2024"] * len(aime2024),
                },
                {
                    "name": "aime2025",
                    "dataset": aime2025,
                    "question": aime2025["problem"],
                    "answer": aime2025["answer"],
                    "source": ["aime2025"] * len(aime2025),
                },
                {
                    "name": "amc23",
                    "dataset": amc23,
                    "question": amc23["problem"],
                    "answer": amc23["answer"],
                    "source": ["amc23"] * len(amc23),
                },
                {
                    "name": "hmmt",
                    "dataset": hmmt,
                    "question": hmmt["problem"],
                    "answer": hmmt["answer"],
                    "source": ["hmmt"] * len(hmmt),
                },
                {
                    "name": "math-500",
                    "dataset": math500,
                    "question": math500["problem"],
                    "answer": math500["answer"],
                    "source": ["math-500"] * len(math500),
                },
            ]
            eval_math_en = fusion_datasets(eval_math_en_datasets).shuffle(seed=0)
            if args.n != -1:
                eval_math_en = eval_math_en.select(range(args.n))
            eval_math_en.save_to_disk(config.DATA_PATHS[1] + "Eval-Math-EN")
            eval_math_en.save_to_disk(config.DATA_PATHS[2] + "Eval-Math-EN")

    if "Train-Math" in args.datasets:
        model_ext = "_"*(len(args.model) > 0) + args.model
        train_math_fr = load_data("Raw-Train-Math-FR")
        train_math_fr = train_math_fr.rename_column("question", "question_en")
        train_math_fr = train_math_fr.rename_column("question_fr", "question")

        if args.rl:
            train_math_fr = train_math_fr.rename_column("answer", "answer_en")
            train_math_fr = train_math_fr.rename_column("answer_fr", "answer")
        else:
            train_math_fr = train_math_fr.rename_column("answer", "answer_en")
            train_math_fr = train_math_fr.rename_column("answer_fr" + model_ext, "answer")

            train_math_fr = train_math_fr.rename_column("solution", "solution_en")
            train_math_fr = train_math_fr.rename_column("solution_fr" + model_ext, "solution")

            train_math_fr = train_math_fr.rename_column("valid_fr" + model_ext, "valid")

        train_math_fr = train_math_fr.filter(remove_instruct_translation)

        train_math_fr = filter_train_math_fr(train_math_fr, args.rl).shuffle(seed=0)
        if args.n != -1:
            train_math_fr = train_math_fr.select(range(int(args.n*args.fr_ratio)))
        train_math_en = load_data("Fused-CoT")
        used_question = set(train_math_fr["question_en"])
        train_math_en = train_math_en.filter(lambda x: x["question"] not in used_question)
        train_math_en = filter_train_math_en(train_math_en).shuffle(seed=0).select(range(int((len(train_math_fr)/args.fr_ratio)*(1-args.fr_ratio))))

        train_math_datasets = [
            {
                "name": "french",
                "dataset": train_math_fr,
                "question": train_math_fr["question"],
                "answer": train_math_fr["answer"],
                "solution": train_math_fr["solution"],
                "source": train_math_fr["source"],
                "model": train_math_fr["model"],
            },
            {
                "name": "english",
                "dataset": train_math_en,
                "question": train_math_en["question"],
                "answer": train_math_en["answer"],
                "solution": train_math_en["solution"],
                "source": train_math_en["source"],
                "model": train_math_en["model"],
            },
        ]

        train_math = fusion_datasets(train_math_datasets)

        train_math.save_to_disk(config.DATA_PATHS[1] + "Train-Math")
        train_math.save_to_disk(config.DATA_PATHS[2] + "Train-Math")

    if "Train-Math-en" in args.datasets:
        train_math_en = load_data("Fused-CoT-Dedup")
        train_math_en = filter_train_math_en(train_math_en).shuffle(seed=0)

        train_math_datasets = [
            {
                "name": "english",
                "dataset": train_math_en,
                "question": train_math_en["question"],
                "answer": train_math_en["answer"],
                "solution": train_math_en["solution"],
                "source": train_math_en["source"],
                "model": train_math_en["model"],
            },
        ]

        train_math = fusion_datasets(train_math_datasets)

        features = set(train_math_datasets[0].keys()) - set(["name", "dataset"])
        datamix = {}
        for feature in features:
            datamix[feature] = []
        
        for source, ratio in zip([
            "s1k-1.1",
            "open-thoughts-3",
            "open-r1-math",
            "open-math-reasoning",
            "nemotron-v1",
            "limo_v2",
            "deepmath",
            "am-deepseek-r1-0528-distill"
        ], [
            0.01,
            0.05,
            0.05,
            0.3,
            0.3,
            0.01,
            0.08,
            0.2
        ]):
            subset = train_math.filter(lambda x: x["source"].split("/")[0] == source)
            n_samples = int(args.n*ratio)
            for feature in features:
                datamix[feature].extend(sample_n_wo_repl(subset[feature], n_samples))
        
        datamix = Dataset.from_dict(datamix).shuffle(seed=0)
        datamix.save_to_disk(config.DATA_PATHS[1] + "Train-Math-en")
        datamix.save_to_disk(config.DATA_PATHS[2] + "Train-Math-en")

