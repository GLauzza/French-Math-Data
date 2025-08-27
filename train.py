import shutil
import argparse

from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from transformers import TrainingArguments, Trainer

import config
from utils_model import *
from process_data.prepare_data import *


def prepare_data(chat_template_fun, dataset, tokenizer):
    print("FM - Preparing Data")
    dataset = dataset.add_column(
        "chat_input",
        [chat_template_fun(sample["question"]) + sample["solution"] + tokenizer.eos_token for sample in dataset]
    )
    print("FM - Prepared Data")
    return dataset


def train(model, tokenizer, dataset, new_model_name):
    print("FM - Instantiate Training")
    # training_args = SFTConfig(
    #     dataset_text_field = "chat_input",
    #     per_device_train_batch_size = 1,
    #     gradient_accumulation_steps = 1,
    #     gradient_checkpointing=True,    
    #     warmup_steps = 13,
    #     num_train_epochs = 0.0001,
    #     learning_rate = 1e-5,
    #     bf16 = True,
    #     logging_first_step = True,
    #     logging_steps = 20,
    #     optim = "adamw_8bit",
    #     weight_decay = 0.00,
    #     lr_scheduler_type = "cosine",
    #     output_dir = config.MODEL_PATHS[1] + new_model_name,
    #     report_to = "tensorboard",
    #     save_safetensors=False,
    #     logging_dir=config.MODEL_PATHS[1] + new_model_name + "/logs",
    #     seed=0,
    #     use_liger_kernel=True,
    #     max_length=16384,
    #     # max_length=1,
    #     # packing=True,
    #     model_init_kwargs={"attn_implementation": "flash_attention_2"},
    #     dataloader_pin_memory=True,
    #     dataloader_num_workers=8,
    #     # torch_compile=True,
    #     # torch_compile_backend="inductor",
    #     # deepspeed="deepspeed.json"
    # )
    training_args = TrainingArguments(
        dataset_text_field = "chat_input",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        gradient_checkpointing=True,    
        warmup_steps = 13,
        num_train_epochs = 0.0001,
        learning_rate = 1e-5,
        bf16 = True,
        logging_first_step = True,
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        output_dir = config.MODEL_PATHS[1] + new_model_name,
        report_to = "tensorboard",
        save_safetensors=False,
        logging_dir=config.MODEL_PATHS[1] + new_model_name + "/logs",
        seed=0,
        use_liger_kernel=True,
        max_length=16384,
        # max_length=1,
        # packing=True,
        model_init_kwargs={"attn_implementation": "flash_attention_2"},
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        # torch_compile=True,
        # torch_compile_backend="inductor",
        # deepspeed="deepspeed.json"
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     processing_class=tokenizer,
    #     data_collator=collator,
    #     args=training_args,
    # )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        args=training_args,
    )

    print("FM - Training")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with TRL')
    parser.add_argument('--model', type=str, default="Qwen2.5-Math-7B-Instruct", help='Model to train')
    parser.add_argument('--dataset', type=str, default="Fused-CoT", help='Dataset to train on')
    parser.add_argument('--name', type=str, default=None, help='Name of the new model')
    args = parser.parse_args()

    model_path, chat_template_fun, _ = get_config(args.model)
    model, tokenizer = load_model(model_path)

    dataset = load_data(args.dataset)
    dataset = prepare_data(chat_template_fun, dataset, tokenizer)

    if args.name:
        new_model_name = args.name
    else:
        new_model_name = config.MODEL_PATHS[1] + args.model + "-SFT-" + args.dataset

    train(model, tokenizer, dataset, new_model_name)

    print("FM - Saving")
    model.save_pretrained(config.MODEL_PATHS[1] + new_model_name, safe_serialization=False)
    tokenizer.save_pretrained(config.MODEL_PATHS[1] + new_model_name)
    shutil.copytree(config.MODEL_PATHS[1] + new_model_name, config.MODEL_PATHS[2] + new_model_name, dirs_exist_ok=True)
    print("FM - Saved")