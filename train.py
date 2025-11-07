import shutil
import argparse
import os

import unsloth
from unsloth.chat_templates import train_on_responses_only

from trl import SFTConfig, SFTTrainer

import config
from utils_model import *
from process_data.prepare_data import *

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin

import wandb


# def prepare_data(chat_template_fun, dataset, tokenizer):
#     print("FM - Preparing Data")
#     def preprocess_function(sample):
#         return {
#             "prompt": chat_template_fun(sample["question"]), 
#             "completion": sample["solution"] + tokenizer.eos_token
#         }
#     print("FM - Prepared Data")
#     dataset = dataset.map(preprocess_function, remove_columns=dataset.features)
#     return dataset


def prepare_data(chat_template_fun, dataset, tokenizer):
    print("FM - Preparing Data")
    dataset = dataset.add_column(
        "text",
        [chat_template_fun(sample["question"]) + sample["solution"] + tokenizer.eos_token for sample in dataset]
    )
    print("FM - Prepared Data")
    return dataset


def train_hf(model, tokenizer, dataset, new_model_name):
    print("FM - Instantiate Training")

    training_args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 64,
        gradient_checkpointing=True,    
        warmup_steps = 25,
        num_train_epochs = 1,
        learning_rate = 5e-5,
        # bf16 = True,
        logging_first_step = True,
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "cosine",
        output_dir = config.MODEL_PATHS[1] + new_model_name,
        report_to = "tensorboard",
        save_safetensors=False,
        logging_dir=config.MODEL_PATHS[1] + new_model_name + "/logs",
        seed=0,
        # use_liger_kernel=True,
        max_length=18000,
        # max_length=1,
        # packing=True,
        # model_init_kwargs={"attn_implementation": "flash_attention_2"},
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        # torch_compile=True,
        # torch_compile_backend="inductor",
        # deepspeed="deepspeed.json",
        # packing=True,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # trainer = Trainer(
    #     model=model,
    #     train_dataset=dataset,
    #     processing_class=tokenizer,
    #     args=training_args,
    # )

    print("FM - Training")
    trainer.train()

def train_unsloth(model, tokenizer, dataset, new_model_name, run_id):
    with wandb.init(
        dir=os.environ["SCRATCH"] + "/wandb", 
        entity="G-lauzzanaa", 
        project="french-cot", 
        id=str(run_id), 
        # resume="must"
        resume="never"
    ):
        trainer = SFTTrainer(
            model = model,
            processing_class = tokenizer,
            train_dataset = dataset,
            args = SFTConfig(
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                ddp_find_unused_parameters = False,
                gradient_checkpointing=True,    
                warmup_steps = 25,
                num_train_epochs = 0.02, # Set this for 1 full training run.
                learning_rate = 5e-5, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                save_strategy = "steps",
                output_dir=new_model_name,
                logging_dir=new_model_name + "/logs",
                save_steps = 10,
                run_name="french-cot",
                optim = "adamw_8bit",
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
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

        print("FM - Training")
        trainer.train()
        # trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "french-cot"

    parser = argparse.ArgumentParser(description='Train model with TRL')
    parser.add_argument('--model', type=str, default="Qwen2.5-Math-7B-Instruct", help='Model to train')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to train on')
    parser.add_argument('--name', type=str, default=None, help='Name of the new model')
    parser.add_argument('--id', type=str, help='unique id to identify the wandb run', required=True)
    args = parser.parse_args()


    # accelerator = Accelerator(
    #     parallelism_config=ParallelismConfig(
    #         dp_shard_size=1, # Fully Sharded Data Parallel degree
    #         dp_replicate_size=1, # Data Parallel degree
    #         cp_size=1, # Context Parallel degree
    #         tp_size=4, # Tensor Parallel degree
    #     ),
    # )

    model_path, chat_template_fun, _ = get_config(args.model)
    model, tokenizer = load_model(model_path, is_unsloth=True)
    # model, tokenizer = load_model(model_path, accelerator=accelerator)

    dataset = load_data(args.dataset)
    dataset = prepare_data(chat_template_fun, dataset, tokenizer)

    if args.name:
        new_model_name = args.name
    else:
        new_model_name = config.MODEL_PATHS[1] + args.model + "-SFT-" + args.dataset

    train_unsloth(model, tokenizer, dataset, new_model_name, args.id)

    print("FM - Saving")
    model.save_pretrained(config.MODEL_PATHS[1] + new_model_name, safe_serialization=False)
    tokenizer.save_pretrained(config.MODEL_PATHS[1] + new_model_name)
    shutil.copytree(config.MODEL_PATHS[1] + new_model_name, config.MODEL_PATHS[2] + new_model_name, dirs_exist_ok=True)
    print("FM - Saved")
