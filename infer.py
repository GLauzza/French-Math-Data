import shutil
import argparse

from tqdm import tqdm
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from math_verify import parse, verify   

import config
from utils_model import *
from process_data.utils_data import *
from topic import *


def classify(model, dataset, dataloader, output_name):
    print("FM - Classifying")
    outputs_lang = []
    outputs_prob = []
    for x in tqdm(dataloader):
        output = model.predict([sample.replace("\n", " ") for sample in x], k=5)
        outputs_lang += output[0]
        outputs_prob += output[1]
        print(f"\n\n\nInputs:\n{x}\n\nOutputs_lang:\n{output[0]}\n\nOutputs_prob:\n{output[1]}\n\n\n")
    print("FM - Classified")

    dataset = dataset.add_column(
        output_name, outputs_lang
    ).add_column(
        output_name+"_prob", outputs_prob
    ).remove_columns(
        ["chat_input", "input_length"]
    )

    return dataset

    
def infer(model, dataset, dataloader, output_name, sampling_params):
    print("FM - Infering")
    outputs = []
    for x in tqdm(dataloader):
        request_outputs = model.generate(x, sampling_params)
        output = [request_output.outputs[0].text for request_output in request_outputs]
        outputs += output
        print(f"\n\n\nLens:{[len(sample) for sample in x]},{[len(sample) for sample in output]}\n\nInputs:\n{x}\n\nOutputs:\n{output}\n\n\n")
    print("FM - Infered")

    dataset = dataset.add_column(
        output_name, outputs
    ).remove_columns(
        ["chat_input", "length"]
    )

    if "solution" in output_name:
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
    parser.add_argument('--task', type=str, default="translation", help='Task to prompt to the model')
    parser.add_argument('--input', type=str, default="question", help='Input to use for the task')
    parser.add_argument('--name', type=str, default=None, help='Name of the new dataset')
    parser.add_argument('--batch-size', type=int, default=-1, help='Batch size')
    args = parser.parse_args()

    model_path, chat_template_fun, sampling_params = get_config(args.model, task=args.task, n=1)
    if args.task == "translation":
        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.8,
            top_k=10,
            min_p=0,
            max_tokens=(32768 if args.input == "solution" else 1024),
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

    dataset = load_data(args.dataset)
    dataset, dataloader, _ = prepare_inference_data(
        dataset,
        chat_template_fun,
        batch_size=args.batch_size,
        input_name=args.input,
        use_only_input=True,
        sortby=("solution" if ("solution" in args.input) or ("math" in args.task) else None)
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
    if args.name:
        new_dataset_name = args.name
    output_name = args.input + "_" + output_ext
    new_dataset_name = args.dataset + "-" + new_dataset_ext

    if args.model == "fasttext":
        dataset = classify(model, dataset, dataloader, output_name)
    else:
        dataset = infer(model, dataset, dataloader, output_name, sampling_params)

    print("FM - Saving")
    dataset.save_to_disk(config.DATA_PATHS[1] + new_dataset_name)
    shutil.copytree(config.DATA_PATHS[1] + new_dataset_name, config.DATA_PATHS[2] + new_dataset_name, dirs_exist_ok=True)
    print("FM - Saved")