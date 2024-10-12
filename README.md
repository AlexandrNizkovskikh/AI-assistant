# Saiga Mistral 7b NLP Project

## Технологический стек

![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-FFD700?logo=huggingface&logoColor=black&style=for-the-badge)
![LLM](https://img.shields.io/badge/-LLMs-ff8800?style=for-the-badge)
![LLaMA Index](https://img.shields.io/badge/-LLaMA%20Index-4b0082?style=for-the-badge)
![Langchain](https://img.shields.io/badge/-Langchain-00FF00?style=for-the-badge)
![Transformers](https://img.shields.io/badge/-Transformers-FF4500?logo=transformers&logoColor=white&style=for-the-badge)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)

## Описание
Проект представляет систему для работы с большими языковыми моделями (LLM) с поддержкой квантования и векторного поиска. Используется модель **Saiga Mistral 7b** для генерации текста, векторизация документов, а также для точного поиска по текстовым данным.

## Основные возможности
- **Модель Saiga Mistral 7b** с поддержкой 4-битного квантования для снижения затрат ресурсов.
- **LoRA (Low-Rank Adaptation)** для дообучения модели.
- **Векторный поиск** через **GPTVectorStoreIndex** и эмбеддинги **sentence-transformers**.
- Интеграция с API Hugging Face и OpenAI.
- Поддержка работы с русскоязычными данными.

## Установка
1. Установите зависимости:
    ```bash
    pip install git+https://github.com/huggingface/transformers
    pip install llama_index pyvis langchain pypdf
    pip install sentencepiece accelerate bitsandbytes peft openai
    ```
2. Аутентифицируйтесь с помощью токенов Hugging Face и OpenAI:
    ```python
    from huggingface_hub import login
    HF_TOKEN = getpass.getpass("Введите Hugging Face токен:")
    login(HF_TOKEN)
    ```

## Использование
1. Загрузка и квантование модели:
    ```python
    model = AutoModelForCausalLM.from_pretrained("IlyaGusev/saiga_mistral_7b", quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_mistral_7b")
    ```
2. Векторизация и создание индекса:
    ```python
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    index = GPTVectorStoreIndex.from_documents(documents)
    ```

3. Поиск по запросу:
    ```python
    query = "Какую ОС поддерживают серверы?"
    response = index.as_query_engine().query(query)
    print(response.response)
    ```
[Открыть в Google Colab](https://colab.research.google.com/drive/1iTkRUYHUXelvHr4S6Utx6B8S7xt0xYFz?usp=sharing)
