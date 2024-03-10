import pandas as pd

train_file_path = '../datas/train_processed.csv'

train_data = pd.read_csv(train_file_path)

def format_data_for_alpaca_separate_questions(row):
    valid_answers = [row[f'답변_{i}'] for i in range(1, 6) if pd.notna(row[f'답변_{i}'])]
    alpaca_formatted_data = []
    
    instruction = "제시된 각 질문에 대해 상세하고 명확한 설명을 제공하시오."
    
    for answer in valid_answers:
        if pd.notna(row['질문_1']):
            input_text_1 = f"{row['질문_1']}"
            alpaca_formatted_data.append({"instruction": instruction, "question": input_text_1, "answer": answer})
        
        if pd.notna(row['질문_2']):
            input_text_2 = f"{row['질문_2']}"
            alpaca_formatted_data.append({"instruction": instruction, "question": input_text_2, "answer": answer})
    
    return alpaca_formatted_data

alpaca_formatted_entries = []
train_data.apply(lambda row: alpaca_formatted_entries.extend(format_data_for_alpaca_separate_questions(row)), axis=1)

alpaca_formatted_df = pd.DataFrame(alpaca_formatted_entries)

alpaca_formatted_file_path = '../datas/alpaca_formatted_train_data.jsonl'
alpaca_formatted_df.to_json(alpaca_formatted_file_path, orient='records', lines=True, force_ascii=False)

print(f"Formatted dataset saved to {alpaca_formatted_file_path}")