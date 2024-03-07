import pandas as pd
from transformers import AutoTokenizer
from ktextaug import TextAugmentation
from KTACC.BERT_augmentation import BERT_Augmentation
from tqdm.auto import tqdm

# 파일 경로 설정
file_path = '../datas/train_processed.csv'
output_path = '../datas/augment_train_processed.csv'

# 데이터 불러오기
data = pd.read_csv(file_path)

# Tokenizer 및 TextAugmentation 인스턴스 생성
tokenizer = AutoTokenizer.from_pretrained('yanolja/KoSOLAR-10.7B-v0.2')


# BERT Augmentation 인스턴스 생성
BERT_aug = BERT_Augmentation()

# 가장 많은 데이터를 가진 카테고리 식별
max_category = data['category'].value_counts().idxmax()

# 증강 대상 데이터 필터링
target_data = data[data['category'] != max_category]

# 증강 함수 정의
def augment_text(agent, text):
    
    res = agent.generate(text)
    
    return res

# 증강 작업 수행
augmented_rows = []
for _, row in tqdm(target_data.iterrows(), total=target_data.shape[0], desc="Augmenting"):
    new_row = row.copy()
    for col in ['질문_1', '질문_2', '답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
        if pd.notnull(row[col]):
            agent = TextAugmentation(tokenizer=tokenizer, num_processes=1)
            new_row[col] = augment_text(agent, row[col])
            del agent
    augmented_rows.append(new_row)

# 증강된 데이터를 DataFrame으로 변환
augmented_df = pd.DataFrame(augmented_rows, columns=data.columns)
combined_data = pd.concat([data, augmented_df], ignore_index=True)

# 각 카테고리별 데이터의 10% 비율만큼 노이즈 추가
noisy_rows = []
for category in combined_data['category'].unique():
    category_data = combined_data[combined_data['category'] == category]
    noise_count = int(len(category_data) * 0.1)  # 10% 데이터에 노이즈 추가
    noise_samples = category_data.sample(n=noise_count)
    
    for _, row in noise_samples.iterrows():
        noisy_row = row.copy()
        noisy_row['id'] = f"{row['id']}_noise"
        for col in ['질문_1', '질문_2', '답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
            noisy_text = BERT_aug.random_masking_replacement(row[col], 0.15)
            noisy_row[col] = noisy_text
        noisy_rows.append(noisy_row)

# 노이즈 데이터를 DataFrame으로 변환하고 기존 데이터에 추가
noisy_df = pd.DataFrame(noisy_rows, columns=combined_data.columns)
final_data = pd.concat([combined_data, noisy_df], ignore_index=True)

# 최종 데이터 저장
final_data.to_csv(output_path, index=False, encoding='utf-8-sig')
print("Data augmentation and noise addition complete and saved.")