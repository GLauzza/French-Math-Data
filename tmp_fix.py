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
    data = load_data("me/Train-Math-en-big")
    new_answer = [
        extract_boxed_text(x["solution"]) if x["source"].startswith("am-deepseek-r1-0528-distill") or x["source"].startswith("open-thoughts-3") else x["answer"]
        for x in data
    ]

    data = data.remove_columns(["answer"])

    data = data.add_column(
        "answer",
        new_answer
    )
    data.save_to_disk(config.DATA_PATHS[1] + "Train-Math-en-big-v2")
    data.save_to_disk(config.DATA_PATHS[2] + "Train-Math-en-big-v2")