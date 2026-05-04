from datasets import Dataset, load_dataset

# data = load_dataset("/lustre/fsn1/projects/rech/knb/ukq43aj/.cache/huggingface/hub/datasets--nvidia--Nemotron-Math-v2/snapshots/8e793210e175b6406c752a870f585f62de98c0d3", split="low")
data = load_dataset("/lustre/fsn1/projects/rech/knb/ukq43aj/.cache/huggingface/hub/datasets--nvidia--Nemotron-Math-v2/snapshots/8e793210e175b6406c752a870f585f62de98c0d3")

for split in data.keys():
    data[split] = data[split].rename_column("problem", "question")
    data[split] = data[split].rename_column("expected_answer", "answer")
    data[split] = data[split].add_column("solution", [sample["messages"][1]["content"] for sample in data[split]])

data.save_to_disk("/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/Nemotron-Math-v2")