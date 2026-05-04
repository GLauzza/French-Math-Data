from datasets import Dataset, load_dataset

data = load_dataset("ShadenA/MathNet", "France", split="train")
print(len(data))
data = data.filter(lambda x: len(x["images"]) == 0)
print(len(data))
data = data.filter(lambda x: x["final_answer"] is not None)
print(len(data))

data = load_dataset("ShadenA/MathNet", "all", split="train")
print(len(data))
data = data.filter(lambda x: x["country"] in ["Canada", "IMO", "Ireland", "New Zealand", "United States"] or x["language"] == "English")
print(len(data))
data = data.filter(lambda x: len(x["images"]) == 0)
print(len(data))
data = data.filter(lambda x: x["final_answer"] is not None)
print(len(data))