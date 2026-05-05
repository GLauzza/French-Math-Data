import shutil
import argparse
import os

import unsloth
from unsloth.chat_templates import train_on_responses_only
from unsloth import unsloth_train

from trl import SFTConfig, SFTTrainer

import config
from utils_model import *
from process_data.prepare_data import *

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin

import wandb

from transformers import TrainingArguments, TrainerCallback
from transformers.optimization import get_scheduler


def prepare_data(chat_template_fun, dataset, tokenizer):
    print("FM - Preparing Data")
    def preprocess_function(sample):
        return {
            "prompt": chat_template_fun(sample["question"]), 
            "completion": sample["solution"] + tokenizer.eos_token
        }
    print("FM - Prepared Data")
    dataset = dataset.map(preprocess_function, remove_columns=dataset.features)
    return dataset


def prepare_data_unsloth(chat_template_fun, dataset, tokenizer):
    print("FM - Preparing Data")
    dataset = dataset.add_column(
        "text",
        [chat_template_fun(sample["question"]) + sample["solution"] + tokenizer.eos_token for sample in dataset]
    )
    print("FM - Prepared Data")
    return dataset


def train(model, tokenizer, dataset, new_model_name, run_id):
    with wandb.init(
        dir=os.environ["SCRATCH"] + "/wandb", 
        entity="G-lauzzanaa", 
        project="french-cot-qwen3", 
        id=str(run_id), 
        resume="never"
    ):
        print("FM - Instantiate Training")

        training_args = SFTConfig(
                per_device_train_batch_size = 1,
                per_device_eval_batch_size = 1,
                gradient_accumulation_steps = 192, # Use GA to mimic batch size!
                eval_accumulation_steps = 192, # Use GA to mimic batch size!
                ddp_find_unused_parameters = False,
                gradient_checkpointing=True,    
                warmup_steps = 60,
                num_train_epochs = 3, # Set this for 1 full training run.
                learning_rate = 6e-5, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                save_strategy = "steps",
                output_dir=new_model_name,
                logging_dir=new_model_name + "/logs",
                save_steps = 50,
                run_name="french-cot-qwen3",
                optim = "adamw_torch_fused",
                weight_decay = 0.0,
                lr_scheduler_type = "cosine",
                seed = 0,
                dataloader_pin_memory=True,
                dataloader_num_workers=0,
                max_length=18000,
                dataset_num_proc=16,
                # packing=True,
                completion_only_loss=True,
                report_to="wandb",
                eval_strategy="steps",
                eval_steps=50,
                bf16=True,
                eval_on_start=True,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset = dataset.select(range(int(len(dataset)*0.975))),
            eval_dataset = dataset.select(range(int(len(dataset)*0.975), len(dataset))),
            processing_class=tokenizer,
            args=training_args,
        )

        print("FM - Training")
        trainer.train()
        #trainer.train(resume_from_checkpoint=True)


def train_unsloth(model, tokenizer, dataset, new_model_name, run_id):
    with wandb.init(
        dir=os.environ["SCRATCH"] + "/wandb", 
        entity="G-lauzzanaa", 
        project="french-cot-qwen3", 
        id=str(run_id), 
        resume="never"
    ):
        trainer = SFTTrainer(
            model = model,
            processing_class = tokenizer,
            train_dataset = dataset.select(range(int(len(dataset)*0.975))),
            eval_dataset = dataset.select(range(int(len(dataset)*0.975), len(dataset))),
            args = SFTConfig(
                per_device_train_batch_size = 1,
                per_device_eval_batch_size = 1,
                gradient_accumulation_steps = 192, # Use GA to mimic batch size!
                eval_accumulation_steps = 192, # Use GA to mimic batch size!
                ddp_find_unused_parameters = False,
                # gradient_checkpointing=True,    
                warmup_steps = 60,
                num_train_epochs = 3, # Set this for 1 full training run.
                learning_rate = 6e-05, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                save_strategy = "steps",
                output_dir=new_model_name,
                logging_dir=new_model_name + "/logs",
                save_steps = 50,
                run_name="french-cot-qwen3",
                optim = "adamw_torch_fused",
                weight_decay = 0.0,
                lr_scheduler_type = "cosine",
                seed = 0,
                dataloader_pin_memory=True,
                dataloader_num_workers=0,   
                max_seq_length=18000,
                dataset_num_proc=16,
                # packing=True,
                completion_only_loss=True,
                report_to="wandb",
                eval_strategy="steps",
                eval_steps=50,
                bf16=True,
                # eval_on_start=True,
            ),
            # callbacks=[ReplaceSchedulerCallback()],
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

        print("FM - Training")
        # unsloth_train(trainer)
        unsloth_train(trainer, resume_from_checkpoint=True)


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "french-cot-qwen3"

    parser = argparse.ArgumentParser(description='Train model with TRL')
    parser.add_argument('--model', type=str, default="Qwen2.5-Math-7B-Instruct", help='Model to train')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to train on')
    parser.add_argument('--use_unsloth', type=bool, default=True, help='Whether to use unsloth')
    parser.add_argument('--name', type=str, default=None, help='Name of the new model')
    parser.add_argument('--id', type=str, help='unique id to identify the wandb run', required=True)
    args = parser.parse_args()

    model_path, chat_template_fun, _ = get_config(args.model)

    if args.use_unsloth:
        model, tokenizer = load_model(model_path, is_unsloth=True)
    else:
        model, tokenizer = load_model(model_path)

    dataset = load_data(args.dataset).shuffle(seed=0)
    dataset = prepare_data(chat_template_fun, dataset, tokenizer)

    if args.name:
        new_model_name = args.name
    else:
        new_model_name = config.MODEL_PATHS[1] + args.model + "-SFT-unsloth-" + args.dataset

    if args.use_unsloth:
        train_unsloth(model, tokenizer, dataset, new_model_name, args.id)
    else:
        train_hf(model, tokenizer, dataset, new_model_name, args.id)

    print("FM - Saving")
    model.save_pretrained(config.MODEL_PATHS[1] + new_model_name, safe_serialization=False)
    tokenizer.save_pretrained(config.MODEL_PATHS[1] + new_model_name)
    shutil.copytree(config.MODEL_PATHS[1] + new_model_name, config.MODEL_PATHS[2] + new_model_name, dirs_exist_ok=True)
    print("FM - Saved")
