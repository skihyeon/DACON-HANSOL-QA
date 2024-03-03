import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def initialize_model_and_tokenizer(base_model, device_map="cpu", torch_dtype=torch.float16, load_in_8bit=False, trust_remote_code=True):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
        trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    return model, tokenizer

def merge_and_save_model(model, peft_model_path, save_merged_model_path):
    peft_model = PeftModel.from_pretrained(model, peft_model_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(save_merged_model_path)
    return merged_model

def main(config_path):
    config = load_config(config_path)
    base_model = config["base_model"]
    peft_model_path = os.path.join("..", "models", config['peft_model'])
    save_merged_model_path = os.path.join("..", "model", config['save_merged_model'])

    model, tokenizer = initialize_model_and_tokenizer(base_model)
    merged_model = merge_and_save_model(model, peft_model_path, save_merged_model_path)
    tokenizer.save_pretrained(save_merged_model_path)

if __name__ == "__main__":
    config_path = '../configs/merge_config.json'
    main(config_path)