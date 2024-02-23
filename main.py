import re
import pandas as pd
import nltk
import subprocess
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from autocorrect import Speller
from natasha import MorphVocab, Doc, NamesExtractor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from textblob import TextBlob
from natasha import (Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser,
                     NewsNERTagger, DatesExtractor, MoneyExtractor, AddrExtractor)
from fuzzywuzzy import fuzz

import speech_recognition as sr
from gtts import gTTS
import os
import random
import main
from typing import Final

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN: Final = 'СВОЙ ТОКЕН ИЗ БОТА https://t.me/BotFather'
language='ru_RU'

BOT_CONFIG = {
    'intents': {
        # Ваши намерения и ответы
    },
    'failure_phrases': [
        'Непонятно. Перефразируйте, пожалуйста.',
        'Я еще только учусь. Спросите что-нибудь другое',
        'Слишком сложный вопрос для меня.',
    ]
}

X_train = ["Привет, как дела?", "Что делаешь сегодня?", "Как погода в Москве?", "Книга"]
y_train = ["приветствие", "вопрос", "вопрос", "книга"]

vectorizer_ml = TfidfVectorizer(analyzer='char', ngram_range=(4, 4))
X_train_tfidf = vectorizer_ml.fit_transform(X_train)

# Создаем классификатор на основе нейронной сети
clf_ml = LinearSVC(dual=False)

# Обучаем нейронную сеть
clf_ml.fit(X_train_tfidf, y_train)


def classify_intent(replica):
    # Используем нейронку для предсказания (прогноза) намерения
    intent = clf_ml.predict(vectorizer_ml.transform([replica]))[0]
    return intent


nltk.download('stopwords')

patterns = "[A-Za-z!#$%&'()*+,./:;<=>?@[\\]^_`{|}~\"\\-]+"
stopwords_ru = set(stopwords.words("russian"))
morph = MorphAnalyzer()

spell = Speller('ru')


# Шаг 1: Обработка ввода
def clean_input(user_input):
    # Удаление лишних пробелов
    cleaned_input = re.sub(patterns, ' ', user_input.strip())
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

    for span in doc.spans:
        span.normalize(morph_vocab)

    lemmatized_tokens = [token.lemma if token.lemma is not None else "" for token in doc.tokens]
    filtered_tokens = [word for word in lemmatized_tokens if word not in stopwords_ru]
    lemmatized_text = " ".join(filtered_tokens)
    return lemmatized_text


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


data = pd.read_csv("C:/Users/Muslim/PycharmProjects/Laba3/kartaslovsent.csv", sep=';')

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


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Говорите...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        print("Вы сказали:", text)
        return text
    except sr.UnknownValueError:
        print("Речь не распознана")
        return None
    except sr.RequestError as e:
        print(f"Ошибка при запросе к Google API: {e}")
        return None


def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang="ru")
    tts.save(filename)
    os.system(f"start {filename}")


# Шаг 6: Классификация по темам
topic_classification_dictionary = {
    "роман": "литература",
    "фантастика": "литература",
    "учебник": "учебная_литература",
    "детектив": "литература",
    "психология": "психология",
    "учебнику": "учебная_литература",
}

# Создание и обучение модели RandomForestClassifier для классификации тем
topic_model = make_pipeline(CountVectorizer(), RandomForestClassifier())
topic_model.fit(list(topic_classification_dictionary.keys()), list(topic_classification_dictionary.values()))


def find_best_match(question, dataset):
    best_match_score = 0
    best_match_answer = None

    # Итерируемся по датасету с шагом 3, так как вопрос, ответ и пустая строка идут последовательно
    for i in range(0, len(dataset), 3):
        current_question = dataset[i].strip()
        current_answer = dataset[i + 1].strip()

        # Вычисление схожести вопроса пользователя с текущим вопросом из датасета
        similarity_score = fuzz.ratio(question, current_question)

        # Обновление наилучшего соответствия, если найдено более близкое
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_answer = current_answer.lstrip('- ').strip()

    return best_match_answer


def get_failure_phrase():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)


# Шаг 3: Классификация намерений
intent_dictionary = {
    "привет": "приветствие",
    "": "пустота",
    "показать книга": "показать_книги",
    "книга": "война_и_мир",
    "дать совет": "рекомендации",
    "цена": "стоимость",
    "сколько": "стоимость",
    "не понял": "не_понял",
    "ответ": "ответ",
    "реклама": "реклама",
    "человек": "человек",
    "бот": "бот",
    "как дела": "спросить_как_дела",
    "что делаешь": "спросить_что_делаешь",
    "погода": "спросить_о_погоде",
    "посоветуй фильм": "посоветовать_фильм",
    "как зовут": "спросить_как_зовут",
    "возраст": "спросить_возраст",
    "где живешь": "спросить_где_живешь",
    "сколько лет": "спросить_сколько_лет",
    "как тебя зовут": "спросить_как_тебя_зовут",
    "пока": "прощание",
    "до свидания": "прощание",
    "попрощаться": "прощание",
    "что нового": "спросить_что_нового",
    "что ты умеешь": "спросить_что_ты_умеешь",
    "смешная история": "рассказать_смешную_историю",
    "что почитать": "рекомендации",
    "как провести время": "посоветовать_как_провести_время",
    "поддержка": "запросить_поддержку",
    "помощь": "запросить_помощь",
    "твое хобби": "спросить_твое_хобби",
    "где работаешь": "спросить_где_работаешь",
    "что ты знаешь": "спросить_что_ты_знаешь",
    "сколько времени": "спросить_сколько_времени",
    "что происходит": "спросить_что_происходит",
    "что ты думаешь": "спросить_что_ты_думаешь",
    "что случилось": "спросить_что_случилось",
    "что у тебя нового": "спросить_что_у_тебя_нового",
    "посоветуй куда поехать": "посоветовать_куда_поехать",
    "позитивная мысль": "поделиться_позитивной_мыслью",
    "что посоветуешь почитать": "посоветовать_по_читать",
    "что посмотреть": "посоветовать_по_смотреть",
    "как поживаешь": "спросить_как_поживаешь",
    "расскажи анекдот": "рассказать_анекдот",
    "как провести выходные": "посоветовать_как_провести_выходные",
    "чем занимаешься": "спросить_чем_занимаешься",
}

# Создание и обучение модели RandomForestClassifier
model = make_pipeline(CountVectorizer(), RandomForestClassifier())
model.fit(list(intent_dictionary.keys()), list(intent_dictionary.values()))


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

        process = subprocess.run(['ffmpeg', '-i', mp3_file_path, wav_file_path])

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


def bot(replica):
    cleaned_input = clean_input(replica)
    print("Очищенный текст:", cleaned_input)

    # Шаг 2: Лемматизация
    lemmatized_input = lemmatize_text(cleaned_input)
    print("Лемматизированный текст:", lemmatized_input)

    # Шаг 3: Классификация намерений
    predicted_intent = model.predict([lemmatized_input])[0]
    print(f"Обнаружено намерение: {predicted_intent}")

    # Шаг 4: Извлечение сущностей
    entities = extract_entities(lemmatized_input)
    print(f"Извлеченные сущности: {entities}")

    # Шаг 5: Сентимент анализ
    sentiment_score = sentiment_analysis(lemmatized_input)
    print(f"Оценка сентимента: {sentiment_score}")

    # Используем машинное обучение для анализа намерений
    ml_intent = classify_intent(replica)
    print(f"Обнаружено намерение с помощью машинного обучения: {ml_intent} \n")

    # Спустя 3 сообщения будет реклама
    if 3 < main.messageCount < 10:
        # Определенные ответы на разные намерения
        if predicted_intent == "приветствие":
            return "Ну привет человек"
        elif predicted_intent == "рекомендации":
            return "Очень советую книгу \"Война и мир\""
        elif predicted_intent == "война_и_мир":
            return "Есть такая книга \"Война и мир\", вам может понравиться"
        elif predicted_intent == "показать_книги":
            return "У нас не очень много книг, штуки 2-3, одна из них \"Война и мир\""
        elif predicted_intent == "стоимость":
            return f"Она стоит порядка {random.randint(200, 3000)} рублей"
        else:
            main.messageCount += 1
            return f"Я тут о книгах больше"

    main.messageCount += 1

    # Генерация ответа на основе датасета
    with open('cleaned_dataset.txt', 'r', encoding='utf-8') as file:
        your_dataset = file.readlines()

    best_match_answer = find_best_match(lemmatized_input, your_dataset)

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

messageCount = 0

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    app.add_handler(MessageHandler(filters.TEXT, run_bot))
    app.add_handler(MessageHandler(filters.VOICE, voice_to_text))

    app.add_error_handler(error)

    print('Polling...')
    app.run_polling(poll_interval=3)