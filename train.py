import shutil
import argparse

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import config
from utils_model import *
from process_data.utils_data import *


def prepare_data(chat_template_fun, dataset, tokenizer):
    dataset = dataset.add_column(
        "chat_input",
        [chat_template_fun(sample["question"]) + sample["solution"] + tokenizer.eos_token for sample in dataset]
    )
    return dataset


def train(model, tokenizer, dataset, output_path):
    print("FM - Instantiate Training")
    training_args = SFTConfig(
        dataset_text_field = "chat_input",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 0.0001,
        learning_rate = 2e-4,
        bf16 = True,
        logging_first_step = True,
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        output_dir = output_path,
        report_to = "tensorboard",
        save_safetensors=False,
        logging_dir=output_path + "/logs",
        seed=0,
        use_liger_kernel=True,
        max_length=128,
        # padding_free=True,
        model_init_kwargs={"attn_implementation": "flash_attention_2"}
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer
    )

    trainer = SFTTrainer(
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
    args = parser.parse_args()

    print("FM - Getting Config")
    model_path, chat_template_fun, _ = get_config(args.model)
    model, tokenizer = load_model(model_path)

    dataset = load_data(args.dataset)
    print("FM - Preparing Data")
    dataset = prepare_data(chat_template_fun, dataset, tokenizer)

    output_path = config.MODEL_PATHS[2] + args.model + "_SFT_" + args.dataset
    train(model, tokenizer, dataset, output_path)

    print("FM - Saving")
    model.save_pretrained(output_path, safe_serialization=False)
    tokenizer.save_pretrained(output_path)
    shutil.copytree(output_path, config.MODEL_PATHS[1] + args.model + "_SFT_" + args.dataset)
    print("FM - Saved")