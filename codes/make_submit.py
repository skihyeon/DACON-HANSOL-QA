from sentence_transformers import SentenceTransformer
import pandas as pd
import json

with open('../configs/submit_config.json', 'r') as config_file:
    config = json.load(config_file)

df_answers = pd.read_csv(config["answer_file_path"])
answers_list = df_answers['answer'].to_list()

modelEmb = SentenceTransformer('distiluse-base-multilingual-cased-v1')
pred_emb = modelEmb.encode(answers_list)

submit = pd.read_csv(config["sample_submit_path"])
submit.iloc[:,1:] = pred_emb
submit.to_csv(config["submit_file_path"], index=False)