import json
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def initialize_model_and_tokenizer(model_name, device_map, bnb_config, peft_config):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model, tokenizer

def prepare_data(train_data_path):
    data = pd.read_csv(train_data_path)
    all_combinations = [{'question': data.at[i, q_col], 'answer': data.at[i, a_col]}
                        for i in range(len(data)) for q_col in ['질문_1', '질문_2']
                        for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']]
    return pd.DataFrame(all_combinations)

def create_dataset(processed_data, tokenizer):
    dataset = Dataset.from_pandas(processed_data)
    dataset_dict = DatasetDict({'train': dataset})
    dataset_dict = dataset_dict.map(lambda x: {'text': f"### 질문: {x['question']}\n\n### 답변: {x['answer']}"})
    dataset_dict = dataset_dict.map(lambda samples: tokenizer(samples["text"]), batched=True)
    return dataset_dict

def main(config_file):
    config = load_config(config_file)
    model, tokenizer = initialize_model_and_tokenizer(config["model"]["model_name"], config["model"]["device"], 
                                                      BitsAndBytesConfig(**config["bnb_config"]), LoraConfig(**config["peft_config"]))
    
    processed_data = prepare_data(config["train_data_path"])
    dataset_dict = create_dataset(processed_data, tokenizer)

    training_args = TrainingArguments(**config["train_config"])
    trainer = Trainer(model=model, train_dataset=dataset_dict["train"], args=training_args,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]
    trainer.train()
    trainer.model.save_pretrained(config["train_config"]["output_dir"])

if __name__ == "__main__":
    main('../configs/train_config.json')
