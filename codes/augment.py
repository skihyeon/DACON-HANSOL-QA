import pandas as pd
import os
from transformers import AutoTokenizer
from ktextaug import TextAugmentation
from KTACC.BERT_augmentation import BERT_Augmentation
from tqdm.auto import tqdm

# 파일 경로 설정
file_path = '../datas/train_processed.csv'
output_path = '../datas/augment_train_processed.csv'

# 데이터 불러오기
data = pd.read_csv(file_path)

if os.path.exists(output_path):
    combined_data = pd.read_csv(output_path)
else:
    combined_data = data.copy()  # 여기서 변경: 기존 데이터를 직접 combined_data에 할당


tokenizer = AutoTokenizer.from_pretrained('yanolja/KoSOLAR-10.7B-v0.2')
BERT_aug = BERT_Augmentation()

# 가장 많은 데이터를 가진 카테고리 식별
max_category = data['category'].value_counts().idxmax()

# 증강 작업 수행
rows_processed = 0  # 처리된 행 수를 추적
save_interval = 10  # 몇 개의 행마다 데이터를 저장할지 결정

already_processed_categories = combined_data['category'].unique()
agent = TextAugmentation(tokenizer=tokenizer, num_processes=1)
for category in tqdm(data['category'].unique(), desc="Processing categories"):
    if category != max_category and category not in already_processed_categories:
        category_data = data[data['category'] == category]
        
        for _, row in tqdm(category_data.iterrows(), total=category_data.shape[0], desc=f"Augmenting {category}"):
            new_row = row.copy()
            for col in ['질문_1', '질문_2', '답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                if pd.notnull(row[col]):
                    augmented_text = agent.generate(row[col])
                    new_row[col] = augmented_text
            combined_data = pd.concat([combined_data, pd.DataFrame([new_row])], ignore_index=True)
            rows_processed += 1
            
            # 일정한 행 수마다 파일 저장
            if rows_processed % save_interval == 0:
                combined_data.to_csv(output_path, index=False, encoding='utf-8-sig')



# 노이즈 추가가 전체 데이터셋에 대해 이루어집니다.
noisy_rows = []
for category in tqdm(combined_data['category'].unique(), desc="Adding noise"):
    category_data = combined_data[combined_data['category'] == category]
    noise_count = int(len(category_data) * 0.1)  # 10% 데이터에 노이즈 추가
    noise_samples = category_data.sample(n=noise_count)

    for _, row in noise_samples.iterrows():
        noisy_row = row.copy()
        noisy_row['id'] = f"{row['id']}_noise"  # 노이즈 추가된 데이터의 id 업데이트
        for col in ['질문_1', '질문_2', '답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
            if pd.notnull(row[col]):
                noisy_text = BERT_aug.random_masking_replacement(row[col], 0.15)
                noisy_row[col] = noisy_text
        noisy_rows.append(noisy_row)

# 노이즈 데이터를 DataFrame으로 변환하고 기존 데이터에 추가
noisy_df = pd.DataFrame(noisy_rows)
final_data = pd.concat([combined_data, noisy_df], ignore_index=True)

# 최종 데이터 저장
final_data.to_csv(output_path, index=False, encoding='utf-8-sig')
print("Data augmentation and noise addition complete and saved.")
