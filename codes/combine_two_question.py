import random
import json


def clean_category(text):
    if "카테고리: " in text and " 질문: " in text:
        return "질문: " + text.split(" 질문: ", 1)[1]
    return text

def combine_questions(data):
    combined_data = data.copy()
    connectors = [" 그리고 ", " 또한, "]
    for i in range(len(data)):
        item = data[i]
        input_text = clean_category(item['input'])
        _, sample_2 = random.sample(data, 2)
        input_text_2 = clean_category(sample_2['input'])
        input_text_2 = input_text_2.replace("질문: ", "", 1)
        combined_input = input_text + random.choice(connectors) + input_text_2
        combined_output = f"{item['output']} {sample_2['output']}"
        combined_data.append({"input": combined_input, "output": combined_output})
    return combined_data

data_list = []
with open('../datas/alpaca_formatted_train_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line))

combined_data = combine_questions(data_list)

new_file_path = '../datas/combined_train_data_with_original.jsonl'
with open(new_file_path, 'w', encoding='utf-8') as new_file:
    for item in combined_data:
        new_file.write(json.dumps(item, ensure_ascii=False) + '\n')