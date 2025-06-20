import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

def save(model, tokenizer, output_path):
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)


def quantize(model_path, output_path, method):
    if method == "tensorRT":
        # Load the model from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Select the quantization config, for example, FP8
        config = mtq.FP8_DEFAULT_CFG

        # Define a forward loop function for calibration
        def forward_loop(model):
            for data in calib_set:
                model(data)

        # PTQ with in-place replacement of quantized modules
        model = mtq.quantize(model, config, forward_loop)
        save(model, tokenizer, output_path) 
    elif method == "torchAO":   
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        save(model, tokenizer, output_path) 