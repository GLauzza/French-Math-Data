import json

from datasets import load_dataset, Dataset
import numpy as np
import matplotlib.pyplot as plt

import config

def load_data(name):
    dataset = load_dataset(config.DATA_PATH+"/"+name, split="train")
    return dataset

def print_distribution(column, column_name):
    filtered = [x for x in column if x is not None]
    n_none = len(column) - len(filtered)
    if n_none > 0:
        print(f"Number of 'None' {column_name}: {n_none}")
    if len(filtered) == 0:
        return
    if type(filtered[0]) == list:
        filtered = [element for sample in filtered for element in sample]
    if type(filtered[0]) == dict:
        for k in filtered[0].keys():
            print_distribution([d[k] for d in filtered], column_name + "." + k)
        return
    if type(filtered[0]) == bool:
        filtered = [str(x) for x in filtered]

    if type(filtered[0]) == str:
        lengths = [len(x) for x in filtered]
        n_tokens = config.TOKENIZER(filtered, return_length=True)["length"]
        plt.figure(figsize=(20, 5))
        plt.hist(lengths, bins=min(len(set(lengths)), 1000))
        plt.title(column_name + " length distribution")
        plt.show()
        plt.figure(figsize=(20, 5))
        plt.hist(n_tokens, bins=min(len(set(n_tokens)), 1000))
        plt.title(column_name + " number of tokens distribution")
        plt.show()
        if sum(lengths) > 10000000:
            return
        unique_values = np.unique(filtered, return_counts=True)
        print(f"There is {len(unique_values[0])} unique {column_name}.")
        if len(unique_values[0]) > 100:
            return
        ordered_ind = np.argsort(-unique_values[1])
        plt.figure(figsize=(20, 5))
        plt.bar(unique_values[0][ordered_ind], unique_values[1][ordered_ind])
        plt.xticks(rotation='vertical')
    else:
        plt.figure(figsize=(20, 5))
        plt.hist(filtered, bins=1000)
    plt.title(column_name + " distribution")
    plt.show()

def print_distributions(dataset, column_names):
    print("Features:", list(dataset.features.keys()))
    print("Number of samples:", dataset.num_rows)
    print("Sample:", json.dumps(dataset[-1], indent=4, sort_keys=True))

    for column_name in column_names:
        print_distribution(dataset[column_name], column_name) 

def flatten_features(dataset, column_names):
    flat_dataset = Dataset.from_dict({k:[] for k in dataset.features.keys()})
    for sample in dataset:
        for i in range(len(sample[column_names[0]])):
            flat_sample = {}
            for k,v in sample.items():
                if type(v) == list and k in column_names:
                    flat_sample[k] = v[i]
                else:
                    flat_sample[k] = v
            flat_dataset = flat_dataset.add_item(flat_sample)
    return flat_dataset
