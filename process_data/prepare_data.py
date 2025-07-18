import sys
import os
import re

from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config


def load_data(path, data_files=None, split="train"):
    print("FM - Loading Dataset:", path)
    try:
        try:
            dataset = load_dataset(config.DATA_PATHS[0]+path, split=split, data_files=data_files)
        except FileNotFoundError:
            try:
                dataset = load_dataset(config.DATA_PATHS[1]+(path.split("/")[-1]), split=split, data_files=data_files)
            except FileNotFoundError:
                dataset = load_dataset(config.DATA_PATHS[2]+(path.split("/")[-1]), split=split, data_files=data_files)
    except ValueError:
        try:
            dataset = load_from_disk(config.DATA_PATHS[0]+path)
        except FileNotFoundError:
            try:
                dataset = load_from_disk(config.DATA_PATHS[1]+(path.split("/")[-1]))
            except FileNotFoundError:
                dataset = load_from_disk(config.DATA_PATHS[2]+(path.split("/")[-1]))
    print("FM - Loaded Dataset:", path)
    return dataset


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


def fusion_datasets(datasets):
    features = set(datasets[0].keys()) - set(["name", "dataset"])
    fused_dataset = {}
    for feature in features:
        fused_dataset[feature] = []

    for dataset in datasets:
        print(f"Processing dataset: {dataset['name']}")
        print(print_distributions(dataset["dataset"],[]))
        for feature in features:
            fused_dataset[feature].extend(dataset[feature])

    return Dataset.from_dict(fused_dataset)


def dirty_remove_math(text):
    letters = "a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ"
    # remove latex and math content
    text = re.sub(r"\$\$(.*?)\$\$", " ", text)
    text = re.sub(r"\$(.*?)\$", " ", text)    
    text = re.sub(r"\\\[(.*?)\\\]", " ", text)
    text = re.sub(r"\\\{(.*?)\\\}", " ", text)    
    text = re.sub(r"\{(.*?)\}", " ", text)
    text = re.sub(r"\[(.*?)\]", " ", text)
    text = re.sub(r"\((.*?)\)", " ", text)
    text = re.sub(rf"\\[{letters}]+\{{.*?\}}", " ", text)
    text = re.sub(rf"\\[{letters}]+", " ", text)
    # ... -> .
    text = re.sub(r"\s*\.\s*\.\s*\.\s*", ". ", text)
    # remove words containing non-words
    text = re.sub(rf"\b\w*[^{letters}0-9\.\s'’:\?\!,;-]+\w*\b", " ", text)
    # remove special chars
    text = re.sub(rf"[^{letters}\.\s'’:\?\!,;]", " ", text, flags=re.UNICODE)
    # remove isolated single character
    for _ in range(2):
        text = re.sub(rf"\b[{letters}0-9]['’]*\b(?:\s+\b[{letters}0-9]['’]*\b)+", " ", text)
        text = re.sub(rf"\s+[^{letters}0-9:\?\!;]\s+", " ", text)
        text = re.sub(r"\W(?:\s+\W)+", " ", text)
    # remove isolated numbers
    text = re.sub(r"(?:\s+(?:(?:mod)?[0-9]+[\.,]?)+){2,}\s+", " ", text)
    # strip
    text = re.sub(r"\s+", " ", text).strip()
    # remove repeated words
    text = re.sub(rf"\b((?:[{letters}0-9]+\s*){{1,5}})\b(?:\s+\1)+", r"\1", text)
    return text


def chunk(data, tokenizer, chunk_size):
    splitted = re.split(r"((?:(?<![\.:])[\.\?\!\n][\s+\n]\n*)(?!\s*-))", data)
    sentences = (
        [chunk + sep for chunk, sep in zip(splitted[0::2], splitted[1::2])] +
        [splitted[-1]]
    )

    chunks = []
    chunk_length = 0
    for sentence in sentences:
        n_tokens = tokenizer(sentence, return_length=True)["length"][0]
        if chunk_length + n_tokens > chunk_size:
            chunk_length = 0
        if chunk_length == 0:
            chunks.append(sentence)
        else:
            chunks[-1] += sentence
        chunk_length += n_tokens
    
    chunks_seps = []
    for i, chunk in enumerate(chunks):
        stripped_chunk = chunk.rstrip()
        chunks_seps.append(chunk[len(stripped_chunk):])
        chunks[i] = stripped_chunk
    
    return chunks, chunks_seps

 
def chunk_batch(data, tokenizer, input_name, chunk_size):
    print("FM - Chunking Data")
    chunked = []
    chunked_seps = []
    for sample in data:
        chunks, chunks_seps = chunk(sample, tokenizer, chunk_size)
        chunked.append(chunks)
        chunked_seps.append(chunks_seps)

    flattened_chunks = [chunk for chunks in chunked for chunk in chunks]
    flattened_chunks_seps = [chunk_sep for chunks_seps in chunked_seps for chunk_sep in chunks_seps]
    sample_ids = [i for i, chunks in enumerate(chunked) for chunk in chunks]
    chunk_ids = [i for chunks in chunked for i, chunk in enumerate(chunks)]

    dataset = Dataset.from_dict({
        input_name: flattened_chunks,
        "sep": flattened_chunks_seps,
        "sample_id": sample_ids,
        "chunk_id": chunk_ids,
        "id": list(range(len(flattened_chunks))),
    })
    print("FM - Chunked Data")
    return dataset


def prepare_inference_data(dataset, chat_template_fun, tokenizer, batch_size=-1, input_name="question", use_only_input=False, sortby=None, chunk_size=-1):
    print("FM - Preparing Data")
    if chunk_size == -1:
        dataset, dataloader, sources = prepare_sorted_inference_data(dataset, chat_template_fun, batch_size, input_name, use_only_input, sortby)
        print("FM - Prepared Data")
        return dataset, dataloader, sources

    chunked_dataset = chunk_batch(dataset[input_name], tokenizer, input_name, chunk_size)

    chunk_n_data = chunked_dataset.filter(lambda x : x["chunk_id"] == 0)
    _, dataloader, _ = prepare_sorted_inference_data(chunk_n_data, chat_template_fun, batch_size, input_name)

    print("FM - Prepared Data")
    return chunked_dataset, dataloader, None


def prepare_sorted_inference_data(dataset, chat_template_fun, batch_size=-1, input_name="question", use_only_input=False, sortby=None, answer_start=None):
    if sortby is None:
        sortby = input_name
    if answer_start is None:
        dataset = dataset.add_column(
            "chat_input",
            [chat_template_fun(input_field) for input_field in dataset[input_name]]
        )
    else:
        dataset = dataset.add_column(
            "chat_input",
            [chat_template_fun(input_field) + answer for input_field, answer in zip(dataset[input_name], answer_start)]
        )
    dataset = dataset.add_column(
        "length",
        [len(x) for x in dataset[sortby]]
    )
    dataset = dataset.sort("length")

    if batch_size == -1:
        batch_size = len(dataset)
    if use_only_input:
        dataloader = DataLoader(dataset["chat_input"], batch_size=batch_size, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sources = None
    if "source" in dataset:
        sources = set(dataset["source"])

    return dataset, dataloader, sources