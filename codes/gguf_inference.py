import logging
import pandas as pd
from llama_cpp import Llama
from tqdm.auto import tqdm
import json
import os

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def initialize_llama(config):
    return Llama(
        n_batch=config["model"]["cfg_batch"],
        model_path=config["model"]["gguf_model_path"],
        n_ctx=config["model"]["cfg_ctx"],
        n_gpu_layers=-1,
    )

def generate_answers(llm, df_test, generation_kwargs):
    res_question, res_answer = [], []

    for row in tqdm(df_test.itertuples(), total=len(df_test)):
        question = f"### 질문: {getattr(row, '질문')}\n\n"
        try:
            res = llm(question, **generation_kwargs)
            answer = res["choices"][0]["text"]
            logging.info(f"질문: {question}\n답변: {answer}\n")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            answer = "Error generating response"
        res_question.append(question)
        res_answer.append(answer)

    return res_question, res_answer

def extract_answers(res_answer):
    return [x.split("### 답변: \n")[1] if "### 답변: \n" in x else x for x in res_answer]

def main():
    config = load_config("../configs/gguf_infer_config.json")
    llm = initialize_llama(config)

    generation_kwargs = {
        "max_tokens": config["generation"]["max_tokens"],
        "stop": [config["generation"]["stop"]],
        "echo": config["generation"]["echo"],
        "top_k": config["generation"]["top_k"],
        "repeat_penalty": config["generation"]["repeat_penalty"],
        "temperature": config["generation"]["temperature"],
    }

    df_test = pd.read_csv(config["test_data"]["test_file_path"])
    res_question, res_answer = generate_answers(llm, df_test, generation_kwargs)
    extracted_answers = extract_answers(res_answer)

    df_res = pd.DataFrame({'question': res_question, 'answer': extracted_answers})
    output_dir = os.path.dirname(config["output"]["result_file_path"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_res.to_csv(config["output"]["result_file_path"], index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()