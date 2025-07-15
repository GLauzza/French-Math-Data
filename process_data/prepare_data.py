import re

from datasets import Dataset
from torch.utils.data import DataLoader

def chunk(data, chunk_size):
    splitted = re.split(r"((?:(?<![\.:])[\.\?\!\n][\s+\n]\n*)(?!\s*-))", data)
    sentences = (
        [chunk + sep for chunk, sep in zip(splitted[0::2], splitted[1::2])] +
        [splitted[-1]]
    )

    chunks = []
    chunk_length = 0
    for sentence in sentences:
        if chunk_length + len(sentence) > chunk_size:
            chunk_length = 0
        if chunk_length == 0:
            chunks.append(sentence)
        else:
            chunks[-1] += sentence
        chunk_length += len(sentence)
    
    chunks_seps = []
    for i, chunk in enumerate(chunks):
        stripped_chunk = chunk.rstrip()
        chunks_seps.append(chunk[len(stripped_chunk):])
        chunks[i] = stripped_chunk
    
    return chunks, chunks_seps

 
def chunk_batch(data, input_name, chunk_size):
    print("FM - Chunking Data")
    chunked = []
    chunked_seps = []
    for sample in data:
        chunks, chunks_seps = chunk(sample, chunk_size)
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


def prepare_inference_data(dataset, chat_template_fun, batch_size=-1, input_name="question", use_only_input=False, sortby=None, chunk_size=-1):
    print("FM - Preparing Data")
    if chunk_size == -1:
        dataset, dataloader, sources = prepare_sorted_inference_data(dataset, chat_template_fun, batch_size, input_name, use_only_input, sortby)
        print("FM - Prepared Data")
        return dataset, dataloader, sources

    chunked_dataset = chunk_batch(dataset[input_name], input_name, chunk_size)

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