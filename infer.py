import shutil
import argparse

from tqdm import tqdm

import config
from utils_model import *
from process_data.utils_data import *
from math_verify import parse, verify   


def infer(model, dataset, dataloader, output_name, sampling_params):
    print("FM - Infering")
    outputs = []
    for x in tqdm(dataloader):
        request_outputs = model.generate(x, sampling_params)
        outputs += [request_output.outputs[0].text for request_output in request_outputs]
    print("FM - Infered")

    dataset = dataset.add_column(
        output_name, outputs
    ).remove_columns(
        ["chat_input", "input_length"]
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
    model = load_model(model_path, is_vllm=True)

    dataset = load_data(args.dataset).select(range(16))
    dataset, dataloader, _ = prepare_inference_data(
        dataset,
        chat_template_fun,
        batch_size=64,
        input_name=args.input + "_fr"*(args.task[-3:] == "_fr"),
        use_only_input=True
    )

    if args.task == "translation":
        output_name = args.input + "_fr"
        new_dataset_name = args.dataset + "-FR"
    elif args.task == "math":
        output_name = "solution" + args.model
        new_dataset_name = args.dataset + "-Solved"
    elif args.task == "math_fr":    
        output_name = "solution_fr_" + args.model
        new_dataset_name = args.dataset + "-Solved"
    if args.name:
        new_dataset_name = args.name

    dataset = infer(model, dataset, dataloader, output_name, sampling_params)

    print("FM - Saving")
    dataset.save_to_disk(config.DATA_PATHS[2] + new_dataset_name)
    shutil.copytree(config.DATA_PATHS[2] + new_dataset_name, config.DATA_PATHS[1] + new_dataset_name, dirs_exist_ok=True)
    print("FM - Saved")