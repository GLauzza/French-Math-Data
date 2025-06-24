import modelopt.torch.quantization as mtq
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig
from llmcompressor import oneshot
from datasets import load_dataset, load_from_disk


def quantize(model_path, output_path, method):
    if method == "tensorRT":
        # Load the model from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Select the quantization config, for example, FP8
        config = mtq.FP8_DEFAULT_CFG
        # Define a forward loop function for calibration
        def forward_loop(model):
            pass
            # for data in calib_set:
            #     model(data)
        # PTQ with in-place replacement of quantized modules
        model = mtq.quantize(model, config, forward_loop)
        model.save_pretrained(output_path, safe_serialization=False)
        tokenizer.save_pretrained(output_path)

    elif method == "torchAO":   
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.save_pretrained(output_path, safe_serialization=False)
        tokenizer.save_pretrained(output_path)

    elif method == "compressor":

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)     
        tokenizer = AutoTokenizer.from_pretrained(model_path)   

        ds = load_from_disk("/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/Fused-CoT-FR").select(range(50))

        def process_and_tokenize(example):
            text = example["question_fr"]
            print("text", text)
            return tokenizer(text, padding=False, max_length=32184, truncation=True, add_special_tokens=False)

        ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

        print("DS", ds, ds[0])

        recipe = """
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    ignore: ["lm_head"]
                    config_groups:
                        group_0:
                            weights:
                                num_bits: 8
                                type: float
                                strategy: tensor
                                dynamic: false
                                symmetric: true
                            input_activations:
                                num_bits: 8
                                type: float
                                strategy: tensor
                                dynamic: false
                                symmetric: true
                            targets: ["Linear"]
                    kv_cache_scheme:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
        """
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=32184,
            num_calibration_samples=3,
        )
        model.save_pretrained(output_path, save_compressed=True)
        tokenizer.save_pretrained(output_path)