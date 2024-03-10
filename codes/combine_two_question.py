import random
import json
import pandas as pd

# 가정: train_file_path 변수에 유효한 파일 경로가 설정되어 있음
train_file_path = '../datas/augment_train_processed.csv'

def load_train_data(file_path):
    return pd.read_csv(file_path)

def format_and_combine_data(data, instruction):
    formatted_data = []
    combined_questions_data = []
    combined_data = []

    # 단계 1: 각 ID에 대해 랜덤한 답변 3개씩 연결
    for _, row in data.iterrows():
        answers = [row[f'답변_{i}'] for i in range(1, 6) if pd.notna(row[f'답변_{i}'])]
        random_answers = random.sample(answers, len(answers))  # 랜덤하게 섞기

        for i in range(1, 3):  # 두 질문 모두 처리
            question_key = f'질문_{i}'
            if pd.notna(row[question_key]):
                for j in range(3):  # 각 질문에 대해 랜덤한 답변 3개 연결
                    formatted_data.append({
                        "id": row['id'],
                        "instruction": instruction,
                        "question": row[question_key],
                        "answer": random_answers[j]
                    })
                # 단계 2: 사용되지 않은 답변 2개 연결
                for unused_answer in random_answers[3:]:  # 남은 답변 사용
                    combined_questions_data.append({
                        "id": row['id'],
                        "instruction": instruction,
                        "question": row[question_key],
                        "answer": unused_answer
                    })

    # 단계 3: 2번에서 생성된 쌍을 랜덤하게 두 개 선택하여 질문과 답변 연결
    random.shuffle(combined_questions_data)  # 랜덤하게 섞기
    while combined_questions_data:
        if len(combined_questions_data) < 2:
            break  # 한 쌍 이하로 남은 경우 중단
        item1 = combined_questions_data.pop()
        for idx, item2 in enumerate(combined_questions_data):
            if item1['id'] != item2['id'] and item1['question'] != item2['question']:
                combined_questions_data.pop(idx)  # 두 번째 아이템 제거
                questions = [item1['question'], item2['question']]
                answers = [item1['answer'], item2['answer']]
                combined_question = random.choice([" 그리고 ", " 또한, "]).join(questions)
                combined_answer = " \n ".join(answers)
                combined_data.append({
                    "instruction": instruction,
                    "question": combined_question,
                    "answer": combined_answer
                })
                break

    return formatted_data + combined_data

# 데이터를 다시 로드하여 새로운 로직 적용

def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

instruction = "제시된 각 질문에 대해 상세하고 명확한 설명을 제공하시오."

train_data = load_train_data(train_file_path)
augmented_data = format_and_combine_data(train_data, instruction)

new_file_path = '../datas/aug_noise_train.jsonl'
save_to_jsonl(augmented_data, new_file_path)

# Return success message for user confirmation
"데이터가 성공적으로 처리되어 .jsonl 파일로 저장되었습니다."