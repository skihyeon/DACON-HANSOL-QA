import random
import json
import pandas as pd

train_file_path = '../datas/train_processed.csv'

def load_train_data(file_path):
    return pd.read_csv(file_path)

def format_data_with_instruction(data, instruction):
    formatted_data = []
    for _, row in data.iterrows():
        valid_answers = [row[f'답변_{i}'] for i in range(1, 6) if pd.notna(row[f'답변_{i}'])]
        for answer in valid_answers:
            for i in [1, 2]:
                question_key = f'질문_{i}'
                if pd.notna(row[question_key]):
                    formatted_data.append({
                        "instruction": instruction,
                        "question": row[question_key],
                        "answer": answer
                    })
    return formatted_data

def combine_questions(data, instruction, connectors=[" 그리고 ", " 또한, "], sample_fraction=1/3):
    combined_data = []
    sample_size = int(len(data) * sample_fraction)
    for _ in range(sample_size):
        item1, item2 = random.sample(data, 2)
        connector = random.choice(connectors)
        combined_question = f"{item1['question']}{connector}{item2['question']}"
        combined_answer = f"{item1['answer']} \n {item2['answer']}"
        combined_data.append({"instruction": instruction, "question": combined_question, "answer": combined_answer})
    return combined_data

def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

instruction = "제시된 각 질문에 대해 상세하고 명확한 설명을 제공하시오."
train_data = load_train_data(train_file_path)
formatted_data = format_data_with_instruction(train_data, instruction)
combined_data = combine_questions(formatted_data, instruction)
augmented_data = formatted_data + combined_data

new_file_path = '../datas/combined_train_data_with_original.jsonl'
save_to_jsonl(data=augmented_data, file_path=new_file_path)
