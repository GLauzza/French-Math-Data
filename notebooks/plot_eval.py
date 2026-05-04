import json
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
import fasttext

# sys.path.append(os.path.abspath(os.path.join("..")))
# from utils_model import * 

EVAL_PATH = "/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/Eval-Math-FR/eval.json"
# EVAL_PATH = "/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/MathAleaMCQ/eval.json"
# EVAL_PATH = "/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/Eval-Math-EN/eval.json"
# EVAL_PATH = "/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/GSMPLUS/eval.json"

with open(EVAL_PATH, "r") as f:
    line = f.readline()

json_string = '{"Qwen/Qwen3-4B-Thinking-2507"'+(line.split('"Qwen/Qwen3-4B-Thinking-2507"')[-1])
# json_string = '{"Qwen/Qwen2.5-7B-Instruct-SFT-unsloth-1250"'+(line.split('"Qwen/Qwen2.5-7B-Instruct-SFT-unsloth-1250"')[-1])
# json_string = line

results = json.loads(json_string)

datasets = list(list(results.values())[0]["accuracies"].keys())
n_models = len(results)


cutoff = 38700
for dataset in datasets:
    uncut_valids = []
    for model in ["Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-4B-Thinking-2507-SFT-2-nemorl-60", "Qwen/Qwen3-4B-Thinking-2507-SFT-3-nemorl-60"]:
        valid_long = 0
        valid_not_long = 0
        non_valid_long = 0
        non_valid_not_long = 0
        for sample in results[model]["samples"][dataset]:
            if sample["is_valid"]:
                if sample["cot_length"] > cutoff:
                    valid_long += 1
                else:
                    valid_not_long += 1
            else:
                if sample["cot_length"] > cutoff:
                    non_valid_long += 1
                else:
                    non_valid_not_long += 1
        total = valid_long + valid_not_long + non_valid_long + non_valid_not_long
        uncut_valids.append((total-non_valid_not_long)/total)
        print(f"{dataset}{" "*(17 - len(dataset))}| {model}{" "*(44 - len(model))}| valid_long: {valid_long} | valid_not_long: {valid_not_long} | non_valid_long: {non_valid_long} | non_valid_not_long: {non_valid_not_long} | cut:{non_valid_long/total:.0%} | acc:{(valid_long+valid_not_long)/total:.0%} | uncut valid:{total-non_valid_not_long}/{total} | length {int(results[model]["cot_lengths"][dataset])}")
    print()
    # print(f"{dataset} {" "*(30 - len(dataset))} | Before: {uncut_valids[0]:.0%} | After: {uncut_valids[1]:.0%} | Diff: {uncut_valids[0]-uncut_valids[1]:.0%}")

for model, result in results.items():
    accs = results[model]["accuracies"].values()
    results[model]["accuracies"]["avg"] = sum(accs) / len(accs)

    fig, ax = plt.subplots(figsize=(25,10))
for i, (model, result) in enumerate(results.items()):
    ax.bar(
        np.arange(len(datasets))*1.2 + (i-(0.5*(n_models-1)))/n_models,
        [result["accuracies"][dataset] if dataset in result["accuracies"] else 0.0 for dataset in datasets],
        width=0.9/n_models,
        label=model.split("/")[-1]
    )

ax.set_xticks(np.arange(len(datasets))*1.2)
ax.set_xticklabels([dataset.replace("/","\n") for dataset in datasets], fontsize=20)

ax.set_yticks(np.arange(5)/5)
ax.set_yticklabels(np.arange(5)/5, fontsize=20)

ax.set_ylabel("Accuracy", fontsize=30)

ax.set_title("Evaluation on French Math Data", fontsize=40)
ax.legend(fontsize=15)
plt.savefig("accuracy.png")
plt.show()

classifier = fasttext.load_model(os.environ["DSDIR"] + "/HuggingFace_Models/facebook/fasttext-language-identification/model.bin")

fig, ax = plt.subplots(figsize=(25,5))

top_languages = {}
for i, (model, result) in enumerate(results.items()):
    languages = {}
    top_languages_model = {}
    
    for dataset in datasets:
        languages_dataset = {}
        for sample in result["samples"][dataset]:
            lang_pred = classifier.predict(sample["generation"].replace("\n", " "), k=3)
            for lang, prob in zip(lang_pred[0], lang_pred[1]):
                lang = lang.split("__label__")[1]
                if lang in languages_dataset:
                    languages_dataset[lang] += prob
                else:
                    languages_dataset[lang] = prob
                if lang in languages:
                    languages[lang] += prob
                else:
                    languages[lang] = prob
        for lang in languages_dataset.keys():
            languages_dataset[lang] = languages_dataset[lang] / len(result["samples"][dataset])
    for lang in languages.keys():
        languages[lang] = languages[lang] / sum([len(result["samples"][dataset]) for dataset in datasets])
        if languages[lang] > 0.005:
            top_languages_model[lang] = languages[lang]

    top_languages[model] = top_languages_model

all_languages = list(set([language for top_languages_model in top_languages.values() for language in top_languages_model.keys()]))
for i, (model, result) in enumerate(results.items()):
    top_languages_model = [top_languages[model].get(language, 0) for language in all_languages]
    ax.bar(
        np.arange(len(top_languages_model))*1.2 + (i-(0.5*(n_models-1)))/n_models,
        list(top_languages_model),
        width=0.9/n_models,
        label=model
    )

ax.set_xticks(np.arange(len(all_languages))*1.2)
ax.set_xticklabels(all_languages, fontsize=20)

ax.set_yticks(np.arange(5)/5)
ax.set_yticklabels(np.arange(5)/5, fontsize=20)

ax.set_ylabel("Percentage", fontsize=30)
ax.set_title("CoT language", fontsize=40)
ax.legend(fontsize=15)
plt.savefig("language.png")
plt.show()