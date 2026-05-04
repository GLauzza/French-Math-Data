from datasets import Dataset, load_dataset

data = load_dataset("OpenLLM-BPI/MathAleaMCQ", "all", split="test")
data = data.add_column(
    "full_question",
    [sample["question"] + "\n" + "\n".join([chr(ord('A') + i) + ": " + choice for i, choice in enumerate(sample["choices"])]) for sample in data]
)
data = data.add_column(
    "source",
    ["mathalea" for sample in data]
)
data = data.add_column(
    "answer",
    [chr(ord('A') + sample["answerKey"]) for sample in data]
)
data = data.add_column(
    "answer_latex",
    [sample["choices"][sample["answerKey"]] for sample in data]
)
data = data.add_column(
    "answer_both",
    [chr(ord('A') + sample["answerKey"]) + ": " + sample["choices"][sample["answerKey"]] for sample in data]
)
data = data.remove_columns(["question"])
data = data.remove_columns(["choices"])
data = data.rename_column("full_question", "question")

print(data[0])

data.save_to_disk("/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/MathAleaMCQ")