# import json
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# import joblib
# import pandas as pd
#
#
# def load_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#     return data
#
#
# file_path = 'C:\\Users\\Muslim\\Downloads\\little2_dataset.jsonl'
#
# # # Загрузите ваш датасет
# # df = pd.read_json(file_path, lines=True)
# #
# # # Уменьшите датасет
# # reduced_df = df.sample(frac=0.25, random_state=42)  # 25% от исходного датасета
# #
# # # Сохраните уменьшенный датасет обратно в файл
# # reduced_df.to_json('C:\\Users\\Muslim\\Downloads\\little2_dataset.jsonl', orient='records', lines=True, force_ascii=False)
#
# train_data = load_jsonl(file_path)
#
# # Разделение набора данных на тренировочный и тестовый
# texts = [item['question'] for item in train_data]
# themes = [item['answer'] for item in train_data]
# X_train, X_test, y_train, y_test = train_test_split(texts, themes, test_size=0.2, random_state=42)
#
# # Создание и обучение модели
# vectorizer = TfidfVectorizer()
# classifier = LinearSVC(dual=False)
#
# model = make_pipeline(vectorizer, classifier)
# model.fit(X_train, y_train)
#
# # Теперь у вас есть обученная модель, которую вы можете использовать для классификации
# # Например:
# new_replica = "как дела?"
# predicted_answer = model.predict([new_replica])[0]
#
# print("Predicted Answer:", predicted_answer)
#
# joblib.dump(model, 'C:\\Users\\Muslim\\Downloads\\clear_dataset.pkl')

import json

def process_intents(json_data):
    processed_intents = {}

    for intent_name, intent_data in json_data["intents"].items():
        examples_set = set(intent_data["examples"])
        responses_set = set(intent_data["responses"])

        processed_intents[intent_name] = {
            "examples": list(examples_set),
            "responses": list(responses_set),
            "theme": ""
        }

    return {"intents": processed_intents}

def save_processed_json(processed_data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "your_input_file.json"  # Замените на путь к вашему входному файлу
    output_file = "processed_output.json"  # Замените на путь к выходному файлу

    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    processed_data = process_intents(json_data)
    save_processed_json(processed_data, output_file)

    print("Processing complete. Check the processed JSON file:", output_file)