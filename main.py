import re
import pandas as pd
import aiml
import nltk
import subprocess
import json
import joblib
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from autocorrect import Speller
from natasha import MorphVocab, Doc, NamesExtractor
from natasha import (Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser,
                     NewsNERTagger, DatesExtractor, MoneyExtractor, AddrExtractor)
from fuzzywuzzy import fuzz
from textblob import TextBlob

import speech_recognition as sr
from gtts import gTTS
import os
import random
from typing import Final

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN: Final = 'key'
current_directory = os.getcwd()

kernel = aiml.Kernel()

if os.path.isfile("bot_brain.brn"):
    kernel.bootstrap(brainFile="bot_brain.brn")
else:
    kernel.bootstrap(learnFiles="std-startup.xml", commands="LOAD AIML BOOK")
    #kernel.saveBrain("bot_brain.brn")


def load_bot_config(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        bot_config = json.load(file)
    return bot_config


# Загрузка данных из датасета по намерениям
processed_bot_config_path = current_directory + "\\processed_bot_config.json"
BOT_CONFIG = load_bot_config(processed_bot_config_path)


X_train = []
y_train = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_train.append(example.lower())
        y_train.append(intent.lower())

vectorizer_ml = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X_train_tfidf = vectorizer_ml.fit_transform(X_train)

# Создаем классификатор на основе нейронной сети
clf_ml = LinearSVC()

# Обучаем нейронную сеть
clf_ml.fit(X_train_tfidf, y_train)

# 9 ГБ модель, обученная на 38000 элементах (вопрос-ответ), больше не позволяла оперативная память
loaded_model = joblib.load('C:\\Users\\Muslim\\Downloads\\clear_dataset.pkl')

#nltk.download('stopwords')

patterns = "[A-Za-z!#$%&'()*+,./:;<=>?@[\\]^_`{|}~\"\\-]+"
stopwords_ru = set(stopwords.words("russian"))
morph = MorphAnalyzer()

spell = Speller('ru')


# Шаг 1: Обработка ввода
def clean_input(user_input):
    # Удаление лишнего
    cleaned_input = re.sub(patterns, ' ', user_input.strip()).lower()
    # Коррекция опечаток с использованием расстояния Левенштейна
    corrected_input = spell(cleaned_input)

    return corrected_input


# Шаг 2: Лемматизация
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)


def lemmatize_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmatized_tokens = [token.lemma if token.lemma is not None else "" for token in doc.tokens]
    filtered_tokens = [word for word in lemmatized_tokens if word not in stopwords_ru]
    lemmatized_text = " ".join(filtered_tokens)
    return lemmatized_text


def classify_intent(replica):
    # Используем нейронку для предсказания (прогноза) намерения
    intent = clf_ml.predict(vectorizer_ml.transform([replica]))[0]
    return intent


# Шаг 4: Извлечение сущностей
def extract_entities(text):
    names_matches = names_extractor(text)
    dates_matches = dates_extractor(text)
    moneys_matches = money_extractor(text)
    addrs_matches = addr_extractor(text)
    entities = ([match.fact for match in names_matches] +
                [match.fact for match in dates_matches] +
                [match.fact for match in moneys_matches] +
                [match.fact for match in addrs_matches])
    return entities


data = pd.read_csv(current_directory + "\\kartaslovsent.csv", sep=';')

text_column = 'term'

# Применение сентимент-анализа к каждой строке в столбце 'term'
data['sentiment_score'] = data[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)


# Шаг 5: Сентимент анализ
def sentiment_analysis(text):
    for index, row in data.iterrows():
        term = row['term']
        sentiment_score = row['value']

        if term in text:
            return sentiment_score

    return None


# #Обработка голосового ввода
# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Говорите...")
#         recognizer.adjust_for_ambient_noise(source, duration=1)
#         audio = recognizer.listen(source)
#
#     try:
#         text = recognizer.recognize_google(audio, language="ru-RU")
#         print("Вы сказали:", text)
#         return text
#     except sr.UnknownValueError:
#         print("Речь не распознана")
#         return None
#     except sr.RequestError as e:
#         print(f"Ошибка при запросе к Google API: {e}")
#         return None


#Озвучка текста
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang="ru")
    tts.save(filename)
    os.system(f"start {filename}")


# Шаг 6: Классификация по темам
theme_terms_books = {
    'Фэнтези': ['магия', 'эльфы', 'драконы', 'волшебство', 'колдовство'],
    'Научная фантастика': ['космос', 'инопланетяне', 'роботы', 'технологии', 'будущее'],
    'Детектив': ['расследование', 'преступление', 'детективный жанр', 'загадка', 'тайна'],
    'Роман': ['любовь', 'отношения', 'чувства', 'страсть', 'романтика'],
    'Исторический': ['история', 'эпоха', 'время', 'война', 'культура'],
    'Ужасы': ['ужасы', 'монстры', 'зомби', 'проклятия', 'паранормальное'],
    'Приключения': ['путешествие', 'опасность', 'приключенческий жанр', 'перипетии', 'поиск'],
    'Наука и образование': ['наука', 'учебные материалы', 'образование', 'учебники', 'знания'],
    'Классика': ['классическая литература', 'великие произведения', 'литературное наследие', 'классика'],
    'Психология': ['психология', 'личность', 'психотерапия', 'самопознание', 'эмоции'],
    'Биографии и мемуары': ['биография', 'автобиография', 'жизнь личности', 'воспоминания', 'портрет'],
    'Философия': ['философия', 'мышление', 'философская мысль', 'идеи', 'мудрость'],
    'Поэзия': ['поэзия', 'стихи', 'лирика', 'поэтическое творчество', 'ритм'],
    'Юмор': ['юмор', 'смех', 'комедия', 'шутки', 'веселье'],
    'Драма': ['драма', 'трагедия', 'эмоциональная литература', 'герои'],
    'Политика': ['политика', 'общество', 'государство', 'политическая литература', 'власть'],
    'Экономика': ['экономика', 'бизнес', 'финансы', 'экономическая литература', 'предпринимательство'],
    'Фантастическая проза': ['фантастика', 'волшебные существа', 'фэнтезийные миры', 'приключения', 'волшебные предметы'],
    'Триллер': ['триллер', 'напряжение', 'неожиданный поворот', 'загадка', 'волнение'],
    'Современная проза': ['современная литература', 'современные авторы', 'современные темы', 'современные проблемы', 'реализм'],
    'Детская литература': ['дети', 'детская книга', 'воспитание', 'воображение', 'образование'],
    'Готика': ['готика', 'мистика', 'темные силы', 'призраки', 'загадочное'],
    'Искусство и дизайн': ['искусство', 'дизайн', 'художественная литература', 'творчество', 'творческий процесс'],
    'Кулинария': ['кулинария', 'рецепты', 'готовка', 'поваренная книга', 'кулинарные шедевры'],
    'Спорт': ['спорт', 'фитнес', 'здоровье', 'физическая активность', 'спортивные достижения'],
    'Технологии': ['технологии', 'инновации', 'техническая литература', 'современные технологии', 'технологический прогресс'],
    'Путеводители': ['путешествия', 'гид', 'туризм', 'путеводитель', 'отпуск'],
    'Семейные отношения': ['семья', 'отношения', 'любовь', 'совместная жизнь', 'брак'],
    'Наука о здоровье': ['здоровье', 'медицина', 'забота о здоровье', 'здоров']
    }

X_train_theme = []
y_train_theme = []

for genre, terms in theme_terms_books.items():
    X_train_theme.append(' '.join(terms))
    y_train_theme.append(genre)

# Создание и обучение модели RandomForestClassifier для классификации тем
topic_model = make_pipeline(CountVectorizer(), RandomForestClassifier())
topic_model.fit(X_train_theme, y_train_theme)


# def find_best_match(question, dataset):
#     best_match_score = 0
#     best_match_answer = None
#
#     # Итерируемся по датасету с шагом 3, так как вопрос, ответ и пустая строка идут последовательно
#     for i in range(0, len(dataset), 3):
#         current_question = dataset[i].strip()
#         current_answer = dataset[i + 1].strip()
#
#         # Вычисление схожести вопроса пользователя с текущим вопросом из датасета
#         similarity_score = fuzz.ratio(question, current_question)
#
#         # Обновление наилучшего соответствия, если найдено более близкое
#         if similarity_score > best_match_score:
#             best_match_score = similarity_score
#             best_match_answer = current_answer.lstrip('- ').strip()
#
#     return best_match_answer


def get_failure_phrase():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(responses)


async def voice_to_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Получаем объект голосового сообщения из сообщения пользователя
    voice = update.message.voice

    if voice:
        # Получаем уникальный идентификатор файла голосового сообщения
        file_id = voice.file_id

        mp3_file_path = 'voice.ogg'
        wav_file_path = 'voice.wav'

        # Скачиваем файл голосового сообщения
        voice_file = await context.bot.get_file(file_id)
        await voice_file.download_to_drive(mp3_file_path)

        process = subprocess.run(['ffmpeg', '-y', '-i', mp3_file_path, wav_file_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if process.returncode != 0:
            print(f'Произошла ошибка при конверсии. Код завершения: {process.returncode}')

        # Вывод ошибок, если они есть
        if process.stderr:
            print(f'Ошибка: {process.stderr.decode("utf-8")}')

        # Используем SpeechRecognition для распознавания текста из файла
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)

        # Пытаемся распознать текст из аудио
        try:
            text = recognizer.recognize_google(audio_data, language="ru-RU")
            await update.message.reply_text(f"Распознанный текст: {text}")
            # Вызываем bot для обработки распознанного текста
            answer = bot(text)
            await update.message.reply_text(answer)
        except sr.UnknownValueError:
            await update.message.reply_text("Не удалось распознать голосовое сообщение.")
        except sr.RequestError as e:
            await update.message.reply_text(f"Ошибка при запросе к Google API: {e}")


messageCount = 0


def bot(replica):
    global messageCount
    cleaned_input = clean_input(replica)
    print("Очищенный текст:", cleaned_input)

    # Шаг 2: Лемматизация
    lemmatized_input = lemmatize_text(cleaned_input)
    print("Лемматизированный текст:", lemmatized_input)

    # Шаг 3: Классификация намерений
    predicted_intent = classify_intent(cleaned_input)
    print(f"Обнаружено намерение: {predicted_intent}")

    # Шаг 4: Извлечение сущностей
    entities = extract_entities(lemmatized_input)
    print(f"Извлеченные сущности: {entities}")

    # Шаг 5: Сентимент анализ
    sentiment_score = sentiment_analysis(lemmatized_input)
    print(f"Оценка сентимента: {sentiment_score}")

    predicted_genre = topic_model.predict([lemmatized_input])[0]
    print(f"Тема книги: {predicted_genre}")

    # # # Спустя 3 сообщения будет реклама
    # # if 3 < messageCount < 10:
    # #     # Определенные ответы на разные намерения
    # #     if predicted_intent == "приветствие":
    # #         return "Ну привет человек"
    # #     elif predicted_intent == "рекомендации":
    # #         return "Очень советую книгу \"Война и мир\""
    # #     elif predicted_intent == "война_и_мир":
    # #         return "Есть такая книга \"Война и мир\", вам может понравиться"
    # #     elif predicted_intent == "показать_книги":
    # #         return "У нас не очень много книг, штуки 2-3, одна из них \"Война и мир\""
    # #     elif predicted_intent == "стоимость":
    # #         return f"Она стоит порядка {random.randint(200, 3000)} рублей"
    # #     else:
    # #         messageCount += 1
    # #         return f"Я тут о книгах больше"
    #
    # messageCount += 1

    aiml_response = kernel.respond(lemmatized_input)

    if aiml_response:
        print(f"Ответ на AIML: {aiml_response}")
        return aiml_response

    if predicted_intent:
        answer = get_answer_by_intent(predicted_intent)
        if answer:
            return answer

    best_match_answer = predict_answer(cleaned_input)

    if best_match_answer is not None:
        return best_match_answer
    else:
        # Если в датасете не найден ответ, используем заглушку
        return get_failure_phrase()


# Шаг 7: Обработка ввода в Telegram
async def run_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    replica = update.message.text
    answer = bot(replica)
    await update.message.reply_text(answer)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Привет!')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Помогите!')


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update}, caused erorr {context.error}')


def predict_answer(question):
    answer = loaded_model.predict([question])[0]
    return answer


if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    app.add_handler(MessageHandler(filters.TEXT, run_bot))
    app.add_handler(MessageHandler(filters.VOICE, voice_to_text))

    app.add_error_handler(error)

    print('Polling...')
    app.run_polling(poll_interval=3)