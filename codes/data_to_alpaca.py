import pandas as pd

# 학습 데이터 파일 경로 설정 (실제 경로로 수정 필요)
train_file_path = '../datas/train_processed.csv'

# 학습 데이터를 불러옵니다.
train_data = pd.read_csv(train_file_path)

def format_data_for_alpaca_separate_questions(row):
    valid_answers = [row[f'답변_{i}'] for i in range(1, 6) if pd.notna(row[f'답변_{i}'])]
    alpaca_formatted_data = []
    
    for answer in valid_answers:
        if pd.notna(row['질문_1']):
            input_text_1 = f"카테고리: {row['category']} \n 질문: {row['질문_1']}"
            alpaca_formatted_data.append({"input": input_text_1, "output": answer})
        
        if pd.notna(row['질문_2']):
            input_text_2 = f"카테고리: {row['category']} \n 질문: {row['질문_2']}"
            alpaca_formatted_data.append({"input": input_text_2, "output": answer})
    
    return alpaca_formatted_data

# 전체 데이터셋에 대해 위의 함수를 적용합니다.
alpaca_formatted_entries = []
train_data.apply(lambda row: alpaca_formatted_entries.extend(format_data_for_alpaca_separate_questions(row)), axis=1)

# 결과를 데이터프레임으로 변환합니다.
alpaca_formatted_df = pd.DataFrame(alpaca_formatted_entries)

# 결과 파일을 JSON 형식으로 저장합니다. (실제 경로로 수정 필요)
alpaca_formatted_file_path = '../datas/alpaca_formatted_train_data.json'
alpaca_formatted_df.to_json(alpaca_formatted_file_path, orient='records', lines=True, force_ascii=False)

print(f"Formatted dataset saved to {alpaca_formatted_file_path}")