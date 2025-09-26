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
    # datasets = [load_data("me/Train-Math-fr-sol-"+str(i)) for i in range(8)]
    # datasets_json = [
    #         {
    #             "name": "data",
    #             "dataset": x,
    #             "question": x["question"],
    #             "answer": x["answer"],
    #             "solution": x['solution'],
    #             "source": x["source"],
    #             "model": x["model"],
    #         }
    #         for x in datasets
    #         ]
    # fused_data = fusion_datasets(datasets_json)
    # fused_data.save_to_disk(config.DATA_PATHS[1] + "Train-Math-fr-sol")
    # fusion_data.save_to_disk(config.DATA_PATHS[2] + "Train-Math-fr-sol")
    ot3 = load_data("mlfoundations-dev/openthoughts3_math_30k")
    ot3 = ot3.add_column(
        "answer",
        [extract_boxed_text(x["final_reasoning_trace"]) for x in ot3]
    )
    train_math_en = load_data("me/Train-Math-en")
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
            {
                "name": "ot3",
                "dataset": ot3,
                "question": ot3["instruction_seed"],
                "answer": ot3["answer"],
                "solution": ot3['final_reasoning_trace'],
                "source": ["OpenThoughts3_math_30k/" + source for source in ot3["_source"]],
                "model": ["QwQ-32b"] * len(ot3),
            }
            ]
    fused_data = fusion_datasets(datasets_json)
    fused_data.save_to_disk(config.DATA_PATHS[1] + "Train-Math-en-ot3")
