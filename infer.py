import shutil
import argparse

from tqdm import tqdm
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from math_verify import parse, verify   

import config
from utils_model import *
from process_data.prepare_data import *
from process_data.extract_answer import *
from topic import *


def classify(model, dataset, dataloader, output_name):
    print("FM - Classifying")
    outputs_lang = []   
    outputs_prob = []
    for x in tqdm(dataloader):
        for sample in x:
            non_math_text = dirty_remove_math(sample).split("\n")
            output = model.predict(non_math_text, k=5)
            unique_langs = set([lang for langs in output[0] for lang in langs])
            languages = {unique_lang:0 for unique_lang in unique_langs}
            for langs, probs in zip(output[0], output[1]):
                for lang, prob in zip(langs, probs):
                    languages[lang] += prob
            languages = {lang:(prob/len(non_math_text)) for lang, prob in languages.items()}
            sorted_languages = sorted(languages.items(), key=lambda item: -item[1])
            outputs_lang.append([lang for lang, prob in sorted_languages])
            outputs_prob.append([prob for lang, prob in sorted_languages])
        print(f"\n\n\nInputs:\n{non_math_text}\n\nOutputs_lang:\n{output[0]}\n\nOutputs_prob:\n{output[1]}\n\n\n")
    print("FM - Classified")

    dataset = dataset.add_column(
        output_name, outputs_lang
    ).add_column(
        output_name+"_prob", outputs_prob
    ).remove_columns(
        ["chat_input", "length"]
    )

    return dataset


def infer_chunked(model, raw_dataset, chunked_dataset, dataloader, output_name, sampling_params, chat_template_fun, batch_size, input_name, chunk_size):
    print("FM - Infering")

    max_chunks = max(chunked_dataset["chunk_id"])
    n_sample = max(chunked_dataset["sample_id"]) + 1
    inputs = [""] * n_sample
    outputs = [[]] * n_sample
    for i in tqdm(range(max_chunks)):
        for data in tqdm(dataloader):
            print(f"\n\n\nLens:{[len(sample) for sample in data['chat_input']]}\n\nInputs:\n{data['chat_input']}\n\n")
            request_outputs = model.generate(data["chat_input"], sampling_params)
            output = [request_output.outputs[0].text for request_output in request_outputs]
            output_toks = [request_output.outputs[0].token_ids for request_output in request_outputs]
            for sample_id, inp, out, out_toks, sep in zip(data["sample_id"], data[input_name], output, output_toks, data["sep"]):
                if len(out_toks) < 1.75*chunk_size and inputs[sample_id] != "<DISCARDED>":
                    if i == 0:
                        outputs[sample_id] = [out + sep]
                    else:
                        outputs[sample_id].append(out + sep)
                    inputs[sample_id] = inp + sep
                else:
                    if i == 0:
                        outputs[sample_id] = "<DISCARDED>"
                    else:
                        outputs[sample_id].append("<DISCARDED>")
                    inputs[sample_id] = "<DISCARDED>"
            print(f"\n\n\nLens:{[len(sample) for sample in output]}\n\nOutputs:\n{output}\n\n\n")

        if i < max_chunks:
            chunk_n_data = chunked_dataset.filter(lambda x : x["chunk_id"] == i+1)
            chunk_n_data = chunk_n_data.add_column(
                "concatenated_chunks",
                [inputs[sample["sample_id"]] + sample[input_name] for sample in chunk_n_data]
            )
            chunk_n_data = chunk_n_data.filter(lambda x : not x["concatenated_chunks"].startswith("<DISCARDED>"))
            answer_start = [outputs[sample_id][i] for sample_id in chunk_n_data["sample_id"]]
            _, dataloader, _ = prepare_sorted_inference_data(chunk_n_data, chat_template_fun, batch_size=batch_size, input_name="concatenated_chunks", answer_start=answer_start)
        
    outputs = [" ".join(output) for output in outputs]
    print("FM - Infered")

    raw_dataset = raw_dataset.add_column(output_name, outputs)
    return raw_dataset

    
def infer(model, dataset, dataloader, output_name, sampling_params):
    print("FM - Infering")
    outputs = []

    for data in tqdm(dataloader):
        request_outputs = model.generate(data, sampling_params)
        output = [request_output.outputs[0].text for request_output in request_outputs]
        outputs += output
        print(f"\n\n\nLens:{[len(sample) for sample in data]},{[len(sample) for sample in output]}\n\nInputs:\n{data}\n\nOutputs:\n{output}\n\n\n")
    print("FM - Infered")

    dataset = dataset.add_column(
        output_name, outputs
    ).remove_columns(
        ["chat_input", "length"]
    )

    return dataset


def extract_answer(dataset, output_name):
    pred_ext = output_name.split("solution")[-1]
    dataset = dataset.add_column(
        "answer" + pred_ext,
        [extract_boxed_text(solution) for solution in dataset[output_name]]
    )
    dataset = dataset.add_column(
        "valid" + pred_ext,
        [
            verify(parse(to_latex(answer)), parse(to_latex(pred)))
            for answer, pred in zip(dataset["answer"], dataset["answer" + pred_ext])
        ]
    )
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs a task on a dataset using a model')
    parser.add_argument('--model', type=str, default="Qwen3-8B", help='Model to use for inference')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to use for inference')
    parser.add_argument('--task', type=str, default="translation", choices=["translation", "math", "math_fr", "topic", "language_classification"], help='Task to prompt to the model')
    parser.add_argument('--input', type=str, default="question", help='Input to use for the task')
    parser.add_argument('--name', type=str, default=None, help='Name of the new dataset')
    parser.add_argument('--batch_size', type=int, default=-1, help='Batch size')
    parser.add_argument('--chunk_size', type=int, default=-1, help='Perform task on chunks of input')
    args = parser.parse_args()

    model_path, chat_template_fun, sampling_params = get_config(args.model, task=args.task, n=1)
    if args.task == "translation":
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            min_p=0,
            repetition_penalty=1.1,
            max_tokens=(2*args.chunk_size if args.chunk_size != -1 else 32768 if args.input == "solution" else 1024),
            seed=0
        )
    elif args.task == "topic":
        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.8, 
            top_k=10, 
            min_p=0, 
            max_tokens=64,
            seed=0,
            guided_decoding=GuidedDecodingParams(grammar=get_topic_grammar())
        )

    model = load_model(model_path, is_vllm=True)

    raw_dataset = load_data(args.dataset).shuffle(seed=42).select(range(20000))
    dataset, dataloader, _ = prepare_inference_data(
        raw_dataset,
        chat_template_fun,
        model.get_tokenizer() if args.chunk_size != -1 else None,
        batch_size=args.batch_size,
        input_name=args.input,
        use_only_input=True,
        sortby=("solution" if ("math" in args.task) else args.input),
        chunk_size=args.chunk_size
    )

    if args.task == "translation":
        output_ext = "fr"
        new_dataset_ext = "FR"
    elif args.task == "math" or args.task == "math_fr":
        output_ext = args.model
        new_dataset_ext = "Solved"
    elif args.task == "language_classification":    
        output_ext = "lang"
        new_dataset_ext = "Lang"
    elif args.task == "topic":    
        output_ext = "topic"
        new_dataset_ext = "Topic"
    output_name = args.input + "_" + output_ext
    new_dataset_name = args.dataset + "-" + new_dataset_ext
    if args.name:
        new_dataset_name = args.name

    if args.model == "fasttext":
        dataset = classify(model, dataset, dataloader, output_name)
    elif args.chunk_size != -1:
        dataset = infer_chunked(model, raw_dataset, dataset, dataloader, output_name, sampling_params, chat_template_fun, args.batch_size, args.input, args.chunk_size)
    else:
        dataset = infer(model, dataset, dataloader, output_name, sampling_params)

    if "solution" in output_name and args.task not in ["topic", "language_classification"]:
        dataset = extract_answer(dataset, output_name)

    print("FM - Saving")
    dataset.save_to_disk(config.DATA_PATHS[1] + new_dataset_name)
    shutil.copytree(config.DATA_PATHS[1] + new_dataset_name, config.DATA_PATHS[2] + new_dataset_name, dirs_exist_ok=True)
    print("FM - Saved")