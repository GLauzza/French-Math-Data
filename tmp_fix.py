import json
from datetime import datetime
import argparse

from datasets import load_dataset, Dataset, Value
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math_verify import parse, verify

from process_data.utils_data import *
from process_data.prepare_data import *
from process_data.filter_data import *
from process_data.extract_answer import *


if __name__ == "__main__":



    # datasets_splits = [load_data("me/Train-Math-fr-sol-"+str(i)) for i in range(8)]
    # datasets_json = [
    #         {
    #             "name": "data",
    #             "dataset": x,
    #             "question": x["question"],
    #             "answer": x["answer"],
    #             "solution": x['solution'],
    #             "source": x["source"],
    #             "model": x["model"],
    #             "solution_fr": x["solution_fr"],
    #             "answer_fr": x["answer_fr"],
    #             "valid_fr": x["valid_fr"],
    #         }
    #         for x in datasets_splits
    #         ]
    # fused_data = fusion_datasets(datasets_json)
    # fused_data.save_to_disk(config.DATA_PATHS[1] + "Train-Math-fr-sol")
    # fused_data.save_to_disk(config.DATA_PATHS[2] + "Train-Math-fr-sol")



    # train_math_en = load_data("me/Train-Math-en-big")
    # train_math_en_2 = load_data("me/Fused-CoT")
    # datasets_json = [
    #        {
    #            "name": "train_math_en",
    #            "dataset": train_math_en,
    #            "question": train_math_en["question"],
    #            "answer": train_math_en["answer"],
    #            "solution": train_math_en['solution'],
    #            "source": train_math_en["source"],
    #            "model": train_math_en["model"],
    #            "already_sampled": [True] * len(train_math_en)
    #        },
    #        {
    #            "name": "train_math_en_2",
    #            "dataset": train_math_en_2,
    #            "question": train_math_en_2["question"],
    #            "answer": train_math_en_2["answer"],
    #            "solution": train_math_en_2['solution'],
    #            "source": train_math_en_2["source"],
    #            "model": train_math_en_2["model"],
    #            "already_sampled": [False] * len(train_math_en_2)
    #       }
    #        ]
    # fused_data = fusion_datasets(datasets_json)
    # fused_data.save_to_disk(config.DATA_PATHS[1] + "Train-Math-en-v3")



    train_math_en = load_data("me/Train-Math-en-v3-Dedup")
    train_math_en = train_math_en.filter(lambda x: not x["already_sampled"])
    datasets_json = [
           {
               "name": "train_math_en",
               "dataset": train_math_en,
               "question": train_math_en["question"],
               "answer": train_math_en["answer"],
               "solution": train_math_en['solution'],
               "source": train_math_en["source"],
               "model": train_math_en["model"],
           },
           ]
    fused_data = fusion_datasets(datasets_json)
    fused_data.save_to_disk(config.DATA_PATHS[1] + "Fused-CoT-Dedup")