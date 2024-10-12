# -*- coding: utf-8 -*-
"""LLM_BOT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iTkRUYHUXelvHr4S6Utx6B8S7xt0xYFz

# Введение

Этот проект представляет собой реализацию системы для обработки запросов с использованием больших языковых моделей (LLM), таких как Saiga Mistral 7b, с интеграцией популярных библиотек для векторных индексов и эмбеддингов, включая Hugging Face, Llama Index и Langchain.
Код позволяет эффективно управлять большими текстовыми данными, структурировать и анализировать их, а также выполнять точные и релевантные ответы на запросы пользователей на основе предобученных моделей и векторных представлений.

Почему Saiga Mistral 7b?

Saiga Mistral 7b — это одна из наиболее продвинутых и сбалансированных моделей для выполнения языковых задач, таких как генерация текста, ответы на вопросы и обработка больших объемов данных.

Вот несколько причин, почему Saiga Mistral 7b была выбрана для этого проекта:

Компактный размер и мощность: Модель содержит 7 миллиардов параметров, что делает её мощной, но одновременно достаточно компактной, чтобы эффективно работать на современных аппаратных платформах с ограниченными ресурсами (в том числе в режиме квантования).

Поддержка квантования: В проекте используется квантование для работы с моделью в 4-битном режиме. Это позволяет существенно снизить нагрузку на GPU и ускорить вычисления без значительных потерь в качестве ответов. Saiga Mistral 7b поддерживает методы, такие как bnb_4bit и nf4, для этого сценария.

Качество генерации текста: Модель демонстрирует отличные результаты в задачах генерации связного и осмысленного текста, что важно для систем обработки естественного языка, требующих высокой степени контекстуального понимания.

Гибкость дообучения: Модель совместима с методами дообучения, такими как PEFT и LoRA, что делает её легко адаптируемой к специфическим задачам и данным.

Русскоязычная поддержка: Saiga Mistral 7b показывает отличные результаты на русскоязычных данных, что критично для проектов, ориентированных на русскоязычную аудиторию.

# Установка библиотек
"""

!pip install git+https://github.com/huggingface/transformers
!pip install llama_index pyvis Ipython langchain pypdf langchain_community
!pip install llama-index-llms-huggingface
!pip install llama-index-embeddings-huggingface
!pip install llama-index-embeddings-langchain
!pip install langchain-huggingface
!pip install sentencepiece accelerate
!pip install -U bitsandbytes
!pip install peft
!pip install openai

"""# Импорт библиотек"""

import warnings
warnings.filterwarnings("ignore")

from llama_index.core import SimpleDirectoryReader, Document, GPTVectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from llama_index.embeddings.langchain import LangchainEmbedding

"""# Аутентификация HF и OpenAI"""

from huggingface_hub import login
import getpass
HF_TOKEN = getpass.getpass("Вставьте ваш токен: ")

# Выполняем аутентификацию
login(HF_TOKEN, add_to_git_credential=True)

import getpass # для работы с паролями
import os      # для работы с окружением и файловой системой

# Запрос ввода ключа от OpenAI
os.environ["OPENAI_API_KEY"] = getpass.getpass("Введите OpenAI API Key:")

"""Так как языковая модель saiga_mistral_7b_lora обучена для ведения диалогов, то для нее определены специальные теги.

Сообщения к модели строиться по шаблону:

< s >{role}\n{content}< /s >,

где content - это текст сообщения к модели, role - одна из возможных ролей:

system - системная роль, определяет преднастройки модели
user - вопросы от пользователей
"""

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'bot':
            prompt += f"<s>bot\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<s>system\n"):
        prompt = "<s>system\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<s>bot\n"
    return prompt

def completion_to_prompt(completion):
    return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"

"""# Загрузка модели

Реализована поддержка дообучения модели через PEFT (Parameter-Efficient Fine-Tuning) с использованием метода LoRA, что позволяет адаптировать модель под специфические задачи с минимальными вычислительными затратами
"""

from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate

# Определяем параметры квантования, иначе модель не выполниться в колабе
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Задаем имя модели
MODEL_NAME = "IlyaGusev/saiga_mistral_7b"

# Создание конфига, соответствующего методу PEFT (в нашем случае LoRA)
config = PeftConfig.from_pretrained(MODEL_NAME)

# Загружаем базовую модель, ее имя берем из конфига для LoRA
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,          # идентификатор модели
    quantization_config=quantization_config, # параметры квантования
    torch_dtype=torch.float16,               # тип данных
    device_map="auto"                        # автоматический выбор типа устройства
)

# Загружаем LoRA модель
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)

# Переводим модель в режим инференса
# Можно не переводить, но явное всегда лучше неявного
model.eval()

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

"""Вывод конфигурации для модели"""

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

llm = HuggingFaceLLM(
    model=model,             # модель
    model_name=MODEL_NAME,   # идентификатор модели
    tokenizer=tokenizer,     # токенизатор
    max_new_tokens=generation_config.max_new_tokens, # параметр необходимо использовать здесь, и не использовать в generate_kwargs, иначе ошибка двойного использования
    model_kwargs={"quantization_config": quantization_config}, # параметры квантования
    generate_kwargs = {   # параметры для инференса
      "bos_token_id": generation_config.bos_token_id, # токен начала последовательности
      "eos_token_id": generation_config.eos_token_id, # токен окончания последовательности
      "pad_token_id": generation_config.pad_token_id, # токен пакетной обработки (указывает, что последовательность ещё не завершена)
      "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
      "repetition_penalty": generation_config.repetition_penalty,
      "temperature": generation_config.temperature,
      "do_sample": True,
      "top_k": 50,
      "top_p": 0.95
    },
    messages_to_prompt=messages_to_prompt,     # функция для преобразования сообщений к внутреннему формату
    completion_to_prompt=completion_to_prompt, # функции для генерации текста
    device_map="auto",                         # автоматически определять устройство
)

"""# Работа с базой данных

Векторизация данных производится с использованием многозадачных эмбеддингов sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, что обеспечивает поддержку нескольких языков, включая русский
"""

from langchain_huggingface  import HuggingFaceEmbeddings

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

# Настройка ServiceContext (глобальная настройка параметров LLM)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

"""Предобработка текстов и структурирования данных, включая возможность извлечения табличной информации из текстовых документов"""

import re
import pandas as pd

def extract_structured_data(text):

    tables = []
    table_pattern = re.compile(r'(\d+(?:\.\d+)?(?:\s+|,|\t)\d+(?:\.\d+)?(?:\s+|,|\t)\d+(?:\.\d+)?(?:\s+|,|\t)\d+)')

    for match in table_pattern.finditer(text):
        table_data = match.group(0)
        # Разделение строки на столбцы по пробелам или запятым
        columns = re.split(r'\s+|,|\t', table_data)
        tables.append(columns)

    if tables:
        # Преобразуем в DataFrame для структурированного хранения
        df = pd.DataFrame(tables)
        return df
    else:
        return None

def preprocess_document(doc):
    if len(doc.text) < 100:  # Фильтрация по длине документа
        return None

    # Извлекаем структурированные данные из текста
    structured_data = extract_structured_data(doc.text)

    # Если данные представлены в виде DataFrame, преобразуем их в список
    if isinstance(structured_data, pd.DataFrame):
        structured_data = structured_data.to_dict(orient="list")  # Преобразуем в сериализуемый формат

    # Возвращаем объект Document с текстом и метаданными
    return Document(text=doc.text, metadata={"structured_data": structured_data})

"""Для примера работы загружается книга "Архитектура компьютера"
"""

# Загрузка и предобработка документов
documents = [preprocess_document(doc) for doc in SimpleDirectoryReader('/kaggle/input/tanebaum-ostin').load_data() if doc]

# Фильтрация None значений после предобработки
documents = [doc for doc in documents if doc is not None]

"""Система использует GPTVectorStoreIndex для создания векторных представлений документов, что позволяет эффективно искать и извлекать информацию на основе сходства текстов"""

index = GPTVectorStoreIndex.from_documents(
	documents
)

"""Фильтрация запроса по длинне символов"""

def classify_query(query, min_length=10, max_length=100):
    """
    Функция классифицирует запрос на короткий, длинный или пустой.

    Аргументы:
    - query: строка запроса.
    - min_length: минимальная длина для "корректного" запроса.
    - max_length: максимальная длина для "корректного" запроса.

    Возвращает:
    - строка с результатом классификации.
    """
    # Удаляем лишние пробелы
    query = query.strip()

    # Проверка на пустой запрос
    if not query:
        return "Запрос пуст."

    # Проверка на короткий запрос
    if len(query) < min_length:
        return f"Запрос слишком короткий. Длина запроса: {len(query)} символов."

    # Проверка на длинный запрос
    if len(query) > max_length:
        return f"Запрос слишком длинный. Длина запроса: {len(query)} символов."

    return "Запрос корректный."

"""# Проверка работы модели

Пример запроса
"""

query = "Серверы работают под управлением каких операционных систем? Поддерживаются ли UNIX и Windows?"
classify_query(query)

"""Пример ответа"""

query_engine = index.as_query_engine(
    similarity_top_k=10
)

message_template = f"""<s>system
Ты являешься моделью, которая отвечает только на основании предоставленных источников.
Отвечай строго на основе информации из текста.
Если нужной информации нет в источнике, ответь: 'я не знаю'. Не добавляй ничего, что не указано в тексте. Не придумывай и не добавляй лишние данные.

<s>user
Вопрос: {query}
Источник:
</s>
"""
#
response = query_engine.query(message_template)
#
print()
print('Ответ:')
print(response.response)