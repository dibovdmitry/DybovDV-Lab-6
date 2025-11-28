# МИНИCTEPCTBO НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ
## Федеральное государственное автономное образовательное учреждение высшего образования «Северо-Кавказский федеральный университет» 
### Институт перспективной инженерии
### Отчет по лабораторной работе 6
### Построение прототипа RAG-системы. Настройка языковой модели в качестве генератора. Создание конвейера RAG-системы. Оценка качества работы системы
Дата: 2025-11-27 \
Семестр: [2 курс 1 полугодие - 3 семестр] \
Группа: ПИН-м-о-24-1 \
Дисциплина: Технологии программирования \
Студент: Дыбов Д.В.

#### Цель работы
Освоение базовых принципов построения RAG (Retrieval‑Augmented Generation) систем: интеграция векторного поиска и языковых моделей; настройка генератора; реализация конвейера RAG; методы оценки качества ретривера и генератора.

#### Теоретическая часть
##### Краткие изученные концепции:
- RAG (Retrieval‑Augmented Generation) - архитектура, где генератор (LLM) дополняется релевантным контекстом, извлечённым ретривером.
- Ретривер - векторное представление документов, индекс, метрики сходства.
- Генератор - языковая модель, промпт‑инжиниринг, контроль длины и стиля ответа.
- Пайплайн RAG - инжест документов -> векторизация -> индексирование -> поиск релевантных фрагментов -> формирование промпта -> генерация ответа -> пост‑обработка.
- Оценка качества - метрики ретривера, метрики генерации, человеческая оценка.
- Оптимизация - кеширование, батчинг запросов, компромиссы между скоростью и качеством, использование lightweight моделей для inference.

#### Практическая часть
##### Выполненные задачи
- [x] Установка необходимых пакетов и создание структуры проекта.
- [x] Подготовка документов и модулей: documents/tech_docs.py.
- [x] Реализация векторного хранилища: retriever/vector_store.py (FAISS/Annoy abstraction).
- [x] Настройка LLM‑клиента: generator/llm_client.py.
- [x] Создание основного скрипта запуска main.py и оптимизированной версии main_optimized.py.
- [x] Разработка пайплайна RAG: pipeline/rag_pipeline.py и сервисного слоя api/pipeline_service.py.
- [x] Написание тестов и бенчмарков: tests/test_pipeline.py, generator/benchmark_system.py.
- [x] Создание модулей для оценки: evaluation/retrieval_evaluator.py, evaluation/generation_evaluator.py, evaluation/rag_evaluator.py.
- [x] Запуск комплексной оценки run_evaluation.py и генерация отчётов/визуализаций.
- [x] Сбор и анализ метрик; визуализация результатов

##### Ключевые фрагменты кода
- Скрипт documents/tech_docs.py
```python
DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Машинное обучение",
        "content": (
            "Машинное обучение — это область искусственного интеллекта, "
            "которая использует статистические методы для создания моделей, "
            "способных обучаться на данных и делать предсказания. Основные типы "
            "машинного обучения включают обучение с учителем, без учителя и с подкреплением."
        ),
        "category": "AI",
    },
    {
        "id": "doc_002",
        "title": "Глубокое обучение",
        "content": (
            "Глубокое обучение использует нейронные сети с множеством слоев для "
            "извлечения иерархических признаков из данных. Популярные архитектуры "
            "включают сверточные нейронные сети для компьютерного зрения и "
            "трансформеры для обработки естественного языка."
        ),
        "category": "AI",
    },
    {
        "id": "doc_003",
        "title": "Трансформеры в NLP",
        "content": (
            "Архитектура трансформеров революционизировала обработку естественного языка. "
            "Модели типа BERT и GPT используют механизм внимания для учета контекста во всей "
            "входной последовательности. BERT предназначен для понимания текста, а GPT — для генерации."
        ),
        "category": "NLP",
    },
    {
        "id": "doc_004",
        "title": "Векторные базы данных",
        "content": (
            "Векторные базы данных оптимизированы для хранения и поиска векторных представлений данных. "
            "Они используют алгоритмы приближенного поиска ближайших соседей для эффективного семантического "
            "поиска. ChromaDB — популярная open-source векторная БД."
        ),
        "category": "Databases",
    },
    {
        "id": "doc_005",
        "title": "RAG-архитектура",
        "content": (
            "RAG (Retrieval-Augmented Generation) сочетает поиск информации в векторной базе данных с "
            "генерацией текста языковой моделью. Это позволяет моделям работать с актуальными данными и снижает "
            "вероятность галлюцинаций."
        ),
        "category": "Architecture",
    },
]
```

- Скрипт generator/llm_client.py
```python
# generator/llm_client.py
import os
import logging
from typing import Any, Dict, List, Sequence, Union, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Отключаем телеметрию HF/снижаем шум в логах и увеличиваем таймауты загрузки
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_REQUEST_TIMEOUT", "600")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Попытка импортировать LangChain Document, но делаем импорт опциональным
try:
    from langchain.schema import Document  # type: ignore
except Exception:
    class Document:
        def __init__(self, page_content: str = "", metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self):
            return f"Document(len={len(self.page_content)}, metadata_keys={list(self.metadata.keys())})"


# Типы для контекста
ContextItem = Union[str, Dict[str, Any], Document]
ContextType = Union[None, str, Dict[str, Any], Sequence[ContextItem]]


class LLMGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name

        # Загружаем tokenizer + модель (явно), чтобы иметь доступ к pad/eos токенам
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            logger.exception("Error loading tokenizer/model: %s", e)
            raise

        # Устанавливаем pad_token если отсутствует
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Создаем pipeline: GPU если доступен, иначе CPU
        device = 0 if torch.cuda.is_available() else -1
        try:
            # Передача model и tokenizer объектами для избежания повторной загрузки по имени
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception as e:
            logger.exception("Error creating pipeline: %s", e)
            raise

        logger.info("Loaded language model: %s (device=%s)", model_name, "cuda" if device == 0 else "cpu")

    def generate_response(self, query: str, context: ContextType = None) -> str:
        """
        Генерация ответа.
        context может быть: None, str, dict, list[str], list[dict], list[Document]
        """
        # Построим строку контекста безопасно
        context_text = self._build_context_string(context)
        prompt = self._construct_prompt(query, context_text)

        # Явная токенизация для контроля truncation входа (не для передачи в pipeline,
        # но чтобы убедиться, что prompt не превышает лимит)
        try:
            tok = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            # логируем длину токенов для отладки
            logger.debug("Prompt tokens: %d", tok["input_ids"].shape[-1])
        except Exception:
            logger.debug("Tokenization failed or not necessary for prompt length check")

        try:
            # Вызываем pipeline с использованием max_new_tokens (не max_length)
            response = self.generator(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Безопасно извлекаем текст из ответа pipeline
            generated_text = ""
            if isinstance(response, list) and response:
                first = response[0]
                if isinstance(first, dict):
                    # возможные ключи: generated_text, text
                    generated_text = first.get("generated_text", "") or first.get("text", "")
                else:
                    generated_text = str(first)
            else:
                generated_text = str(response)

            generated_text = generated_text or ""

            # Иногда pipeline возвращает prompt+answer, иногда только answer.
            answer = generated_text
            if generated_text.startswith(prompt):
                # отрезаем промт, берем только сгенерированную часть
                answer = generated_text[len(prompt) :].strip()
            answer = answer.strip()

            if not answer:
                # Защита: вернем пустую строку, но логируем причину
                logger.warning("Model returned empty answer. Prompt length: %d. Context length: %d",
                               len(prompt), len(context_text))
                return ""

            return answer

        except Exception as e:
            logger.exception("Generation error: %s", e)
            return "Извините, произошла ошибка при генерации ответа."

    def _extract_text_and_meta(self, doc: ContextItem) -> Dict[str, Any]:
        """Безопасно извлечь текст и метаданные из разных форматов документа."""
        # Строка
        if isinstance(doc, str):
            return {"content": doc, "metadata": {}, "similarity_score": 0.0}

        # LangChain Document или аналогичный объект с page_content
        if hasattr(doc, "page_content"):
            content = getattr(doc, "page_content") or ""
            metadata = getattr(doc, "metadata", {}) or {}
            score = getattr(doc, "score", None) or metadata.get("score", 0.0)
            try:
                score = float(score)
            except Exception:
                score = 0.0
            return {"content": content, "metadata": metadata, "similarity_score": score}

        # dict-like
        if isinstance(doc, dict):
            content = doc.get("content") or doc.get("page_content") or doc.get("text") or ""
            metadata = doc.get("metadata") or {}
            score = doc.get("similarity_score") or doc.get("score") or metadata.get("score", 0.0)
            try:
                score = float(score)
            except Exception:
                score = 0.0
            return {"content": content, "metadata": metadata, "similarity_score": score}

        # fallback
        return {"content": str(doc), "metadata": {}, "similarity_score": 0.0}

    def _build_context_string(self, context: ContextType) -> str:
        """
        Построение строки контекста. Поддерживает None, str, dict, sequence.
        Возвращает пустую строку, если context пуст или невалиден.
        """
        if not context:
            return ""

        # если передана строка
        if isinstance(context, str):
            return context.strip()

        # если передан одиночный dict или объект с page_content
        if isinstance(context, dict) or hasattr(context, "page_content"):
            single = self._extract_text_and_meta(context)  # type: ignore[arg-type]
            title = single["metadata"].get("title", "без названия")
            return f"[Документ 1] {title} (схожесть: {single['similarity_score']:.3f}): {single['content']}"

        # итерируемая последовательность
        parts: List[str] = []
        try:
            for i, doc in enumerate(context):  # type: ignore[call-arg]
                item = self._extract_text_and_meta(doc)
                title = item["metadata"].get("title", f"Документ {i+1}")
                score = item["similarity_score"]
                content = item["content"] or ""
                # обрезаем слишком длинные куски контекста
                max_piece = 2000
                if len(content) > max_piece:
                    content = content[:max_piece] + "..."
                parts.append(f"[Документ {i + 1}] {title} (схожесть: {score:.3f}): {content}")
        except TypeError:
            # если не итерируемый объект — вернуть его строковое представление
            return str(context)

        return "\n\n".join(parts)

    def _construct_prompt(self, query: str, context: str) -> str:
        """Конструирование промта для языковой модели."""
        prompt = (
            "На основе предоставленного контекста, ответь на вопрос пользователя. "
            "Если в контексте нет достаточной информации, честно скажи об этом.\n\n"
        )
        if context:
            prompt += f"Контекст:\n{context}\n\n"
        prompt += f"Вопрос: {query}\n\nОтвет:"
        return prompt
```

- Скрипт retriever/vector_store.py
```python
import logging
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection_name: str = "rag_documents"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Добавление документов в векторное хранилище."""
        ids = [doc["id"] for doc in documents]
        texts = [doc["content"] for doc in documents]
        metadatas = [
            {
                "title": doc["title"],
                "category": doc.get("category"),
                "source": "tech_docs",
            }
            for doc in documents
        ]

        # Генерация эмбеддингов
        embeddings = self.model.encode(texts).tolist()

        # Добавление в коллекцию
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(documents)} documents to collection")

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Поиск релевантных документов."""
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Форматирование результатов
        formatted_results: List[Dict[str, Any]] = []
        for i, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            formatted_results.append(
                {
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": 1 - distance,
                    "rank": i + 1,
                }
            )

        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        """Получение информации о коллекции."""
        return {
            "name": self.collection_name,
            "document_count": self.collection.count(),
        }
```

- Скрипт main.py
```python
import logging
from typing import Any, Dict

from retriever.vector_store import VectorStore
from generator.llm_client import LLMGenerator
from documents.tech_docs import DOCUMENTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        self.retriever = VectorStore()
        self.generator = LLMGenerator()
        self._initialize_database()

    def _initialize_database(self):
        """Инициализация базы данных с документами"""
        if self.retriever.get_collection_info().get("document_count", 0) == 0:
            logger.info("Initializing database with documents...")
            self.retriever.add_documents(DOCUMENTS)
        else:
            logger.info("Database already initialized")

    def ask(self, question: str, n_documents: int = 3) -> Dict[str, Any]:
        """Основной метод для вопросов к RAG-системе"""
        logger.info(f"Processing question: {question}")

        # Шаг 1: Поиск релевантных документов
        retrieved_docs = self.retriever.search(question, n_results=n_documents)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Шаг 2: Генерация ответа на основе контекста
        answer = self.generator.generate_response(question, retrieved_docs)

        # Формирование полного ответа
        response = {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "document_count": len(retrieved_docs),
        }
        return response

    def get_system_info(self) -> Dict[str, Any]:
        """Получение информации о системе"""
        return {
            "retriever": self.retriever.get_collection_info(),
            "generator": {"model": getattr(self.generator, "model_name", None)},
            "status": "ready",
        }


# Пример использования
if __name__ == "__main__":
    rag_system = RAGSystem()

    # Тестовые вопросы
    test_questions = [
        "Что такое машинное обучение?",
        "Какие бывают типы машинного обучения?",
        "Как работают трансформеры в NLP?",
        "Что такое RAG-архитектура?",
    ]
    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Вопрос: {question}")
        response = rag_system.ask(question)
        print(f"Ответ: {response['answer']}")
        print(f"Найдено документов: {response['document_count']}")
        if response["document_count"] > 0:
            best = response["retrieved_documents"][0]
            title = best.get("metadata", {}).get("title", "без названия")
            print(f"Лучший документ: {title}")
        else:
            print("Лучший документ: отсутствует")
```

- Скрипт test_rag_system.py
```python
from main import RAGSystem


def test_rag_system():
    rag = RAGSystem()
    test_cases = [
        {
            "question": "Что такое машинное обучение?",
            "expected_keywords": ["искусственный интеллект", "статистические методы", "предсказания"],
        },
        {
            "question": "Какие нейронные сети используются в глубоком обучении?",
            "expected_keywords": ["сверточные", "трансформеры", "слои"],
        },
    ]

    print("Тестирование RAG-системы:")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nТест {i}: {test_case['question']}")
        response = rag.ask(test_case["question"])
        print(f"Ответ: {response['answer']}")
        print(f"Найдено документов: {response['document_count']}")

        # Проверка ключевых слов
        answer_lower = response["answer"].lower()
        found_keywords = [
            kw for kw in test_case["expected_keywords"] if kw in answer_lower
        ]
        print(f"Найдено ключевых слов: {len(found_keywords)}/{len(test_case['expected_keywords'])}")
        print(f"Ключевые слова: {found_keywords}")


if __name__ == "__main__":
    test_rag_system()
```

- Скрипт benchmark_system.py
```python
import pandas as pd
import time
import logging
from typing import List, Dict, Any

from .model_comparison import ModelComparator
from .optimized_generator import OptimizedLLMGenerator

logger = logging.getLogger(__name__)


class ModelBenchmark:
    def __init__(self):
        self.comparator = ModelComparator()
        self.test_questions = [
            {
                "question": "Что такое машинное обучение?",
                "context": [
                    {
                        "content": (
                            "Машинное обучение — это область искусственного интеллекта, "
                            "которая использует статистические методы для создания моделей, "
                            "способных обучаться на данных и делать предсказания."
                        ),
                        "metadata": {"title": "Машинное обучение", "category": "AI"},
                        "similarity_score": 0.95,
                    }
                ],
            },
            {
                "question": "Какие типы нейронных сетей вы знаете?",
                "context": [
                    {
                        "content": (
                            "Популярные архитектуры нейронных сетей включают сверточные нейронные сети "
                            "для компьютерного зрения и трансформеры для обработки естественного языка."
                        ),
                        "metadata": {"title": "Глубокое обучение", "category": "AI"},
                        "similarity_score": 0.88,
                    }
                ],
            },
        ]

    def run_benchmark(self, model_names: List[str]) -> pd.DataFrame:
        """Запуск сравнительного тестирования моделей"""
        results = []

        for model_name in model_names:
            logger.info(f"Benchmarking model: {model_name}")
            self.comparator.load_model(model_name)

            for test_case in self.test_questions:
                question = test_case["question"]
                context = test_case["context"]

                result = self.comparator.generate_with_model(model_name, question)

                evaluation = self._evaluate_response(
                    result["answer"],
                    question,
                    context,
                )

                benchmark_result = {
                    "model": model_name,
                    "question": question,
                    "answer": result["answer"],
                    "generation_time": result["generation_time"],
                    "answer_length": result["answer_length"],
                    "success": result["success"],
                }
                benchmark_result.update(evaluation)
                results.append(benchmark_result)

                time.sleep(1)

        return pd.DataFrame(results)

    def _evaluate_response(self, answer: str, question: str, context: List[Dict]) -> Dict[str, Any]:
        """Базовая оценка качества ответа"""
        context_keywords = self._extract_keywords_from_context(context)
        answer_keywords = self._extract_keywords(answer)

        matched_keywords = set(context_keywords) & set(answer_keywords)
        keyword_coverage = len(matched_keywords) / len(context_keywords) if context_keywords else 0

        return {
            "keyword_coverage": keyword_coverage,
            "matched_keywords_count": len(matched_keywords),
            "answer_has_content": len(answer.strip()) > 10,
            "contains_uncertainty": "не знаю" in answer.lower() or "нет информации" in answer.lower(),
        }

    def _extract_keywords_from_context(self, context: List[Dict]) -> List[str]:
        """Извлечение ключевых слов из контекста"""
        all_text = " ".join([doc["content"] for doc in context])
        words = all_text.lower().split()

        stop_words = {"и", "в", "на", "с", "по", "для", "это", "что", "как"}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        return list(set(keywords))[:10]
```

- Скрипт model_comparison.py
```python
import logging
import time
from datetime import datetime
from typing import Any, Dict

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)

logger = logging.getLogger(__name__)


class ModelComparator:
    def __init__(self):
        self.models_config = {
            "gpt2-medium": {
                "type": "causal",
                "description": "Авторегрессивная модель среднего размера",
            },
            "t5-small": {
                "type": "seq2seq",
                "description": "Seq2Seq модель для переформулирования",
            },
            "facebook/bart-base": {
                "type": "seq2seq",
                "description": "BART модель для текстовых задач",
            },
            "microsoft/DialoGPT-medium": {
                "type": "causal",
                "description": "Диалоговая модель на основе GPT-2",
            },
        }
        self.loaded_models = {}

    def load_model(self, model_name: str):
        """Загрузка модели с обработкой ошибок"""
        try:
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()

            if self.models_config[model_name]["type"] == "causal":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                tokenizer.pad_token = tokenizer.eos_token
            else:  # seq2seq
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

            load_time = time.time() - start_time
            self.loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": self.models_config[model_name]["type"],
                "load_time": load_time,
            }
            logger.info(f"Successfully loaded {model_name} in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")

    def generate_with_model(self, model_name: str, prompt: str, max_length: int = 200) -> Dict[str, Any]:
        """Генерация текста с указанной моделью"""
        if model_name not in self.loaded_models:
            self.load_model(model_name)

        model_info = self.loaded_models[model_name]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        start_time = time.time()

        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id if model_info["type"] == "causal" else None,
                    repetition_penalty=1.1 if model_info["type"] == "causal" else None,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if model_info["type"] == "causal":
                answer = generated_text[len(prompt):].strip()
            else:
                answer = generated_text.strip()

            generation_time = time.time() - start_time
            return {
                "model": model_name,
                "answer": answer,
                "generation_time": generation_time,
                "answer_length": len(answer),
                "success": True,
            }
        except Exception as e:
            logger.error(f"Generation failed for {model_name}: {e}")
            return {
                "model": model_name,
                "answer": f"Error: {str(e)}",
                "generation_time": 0,
                "answer_length": 0,
                "success": False,
            }
```

- Скрипт optimized_generator.py
```python
import importlib.util
import logging
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

logger = logging.getLogger(__name__)


class OptimizedLLMGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.quantization_config: Optional[BitsAndBytesConfig] = None
        self.generation_config: Optional[GenerationConfig] = None
        self.tokenizer = None
        self.model = None

        self._setup_quantization()
        self._setup_generation_config()
        self._load_model()

    def _setup_quantization(self):
        """Настройка квантования для экономии памяти, если доступен bitsandbytes."""
        if importlib.util.find_spec("bitsandbytes") is None:
            logger.warning("bitsandbytes не найден — квантование отключено")
            self.quantization_config = None
            return

        try:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("BitsAndBytesConfig создан для 4-bit квантования")
        except Exception as e:
            logger.warning(f"Не удалось создать BitsAndBytesConfig, квантование отключено: {e}")
            self.quantization_config = None

    def _setup_generation_config(self):
        """Настройка параметров генерации (используем безопасные по умолчанию)."""
        self.generation_config = GenerationConfig(
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=50256,
        )

    def _load_model(self):
        """Загрузка модели с учётом наличия GPU и доступности квантования."""
        logger.info(f"Loading optimized model: {self.model_name}")

        cuda_available = torch.cuda.is_available()
        torch_dtype = torch.float16 if cuda_available else torch.float32
        device_map = "auto" if cuda_available else None

        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Убедимся, что pad_token установлен
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Подготовка аргументов для from_pretrained
        kwargs = dict(
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Применяем quantization_config только когда CUDA доступна и config существует
        if cuda_available and self.quantization_config is not None:
            kwargs["quantization_config"] = self.quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        # Если модель загружена без device_map (CPU-only), явно переведём на CPU
        if not cuda_available:
            self.model.to("cpu")

        logger.info("Model loaded successfully with optimizations (device_map=%s, dtype=%s)",
                    device_map, torch_dtype)

    def generate_optimized_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Оптимизированная и безопасная генерация ответа.

        Гарантирует:
        - тензоры перемещены на устройство модели,
        - генерация использует max_new_tokens,
        - декодировка безопасна и не полагается на простое отрезание len(prompt).
        """
        start_time = time.time()

        # Построение улучшенного промта
        prompt = self._construct_enhanced_prompt(query, context)

        try:
            # Токенизация с ограничением длины входа
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            # Переместить входные тензоры на устройство модели
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Параметры генерации (берём из generation_config, но допускаем overrides)
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": getattr(self.generation_config, "max_new_tokens", 150),
                "do_sample": getattr(self.generation_config, "do_sample", True),
                "temperature": getattr(self.generation_config, "temperature", 0.7),
                "top_p": getattr(self.generation_config, "top_p", 0.9),
                "top_k": getattr(self.generation_config, "top_k", 50),
                "repetition_penalty": getattr(self.generation_config, "repetition_penalty", 1.1),
                "pad_token_id": getattr(self.generation_config, "pad_token_id", self.tokenizer.pad_token_id),
            }

            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Декодирование результата
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Надёжный способ отделить промт от сгенерированного продолжения:
            # если модель вернула текст, начинающийся с промта -> отрежем промт,
            # иначе используем весь сгенерированный текст.
            if generated_text.startswith(prompt.strip()):
                answer = generated_text[len(prompt.strip()):].strip()
            else:
                answer = generated_text

            generation_time = time.time() - start_time

            return {
                "answer": answer,
                "generation_time": generation_time,
                "prompt_length": len(prompt),
                "answer_length": len(answer),
                "model": self.model_name,
                "optimized": True,
            }

        except Exception as e:
            logger.error("Optimized generation failed: %s", e, exc_info=True)
            return {
                "answer": f"Generation error: {str(e)}",
                "generation_time": 0,
                "prompt_length": len(prompt),
                "answer_length": 0,
                "model": self.model_name,
                "optimized": False,
            }

    def _construct_enhanced_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Улучшенное конструирование промта"""
        context_text = self._build_structured_context(context)
        enhanced_prompt = (f"""Ты - AI-ассистент, который отвечает на вопросы на основе предоставленного контекста.\n\n
            ИНСТРУКЦИИ:
 1. Используй только информацию из предоставленного контекста
 2. Если в контексте нет ответа, честно скажи об этом
 3. Будь точным и информативным
 4. Отвечай на русском языке
 КОНТЕКСТ: {context_text}
 ВОПРОС: {query}
 ОТВЕТ: """
        )
        return enhanced_prompt

    def _build_structured_context(self, context: List[Dict[str, Any]]) -> str:
        """Структурированное построение контекста"""
        context_parts = []
        for i, doc in enumerate(context, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {}) or {}
            title = metadata.get("title", "Без названия")
            category = metadata.get("category", "unknown")
            score = doc.get("similarity_score", 0.0)
            context_parts.append(
                f"Документ {i}:\n"
                f"Заголовок: {title}\n"
                f"Категория: {category}\n"
                f"Релевантность: {score:.3f}\n"
                f"Содержание: {content}\n"
            )
        return "\n" + "=" * 50 + "\n".join(context_parts) + "=" * 50
```

- Скрипт main_optimized.py
```python

```

- Скрипт generate_report.py
```python

```

- Скрипт api/pipeline_service.py
```python

```

- Скрипт config/pipeline_config.py
```python

```

- Скрипт pipeline/rag_pipeline.py
```python

```

- Скрипт tests/test_pipeline.py
```python

```

- Скрипт demo/demo_pipeline.py
```python

```

- Скрипт evaluation/generation_evaluation.py
```python

```

- Скрипт evaluation/rag_evaluation.py
```python

```

- Скрипт evaluation/retrieval_evaluation.py
```python

```

- Скрипт evaluation/test_dataset.py
```python

```

- Скрипт evaluation/visualisation.py
```python

```
##### Результаты выполнения

1. Установлена Java, скачан Protégé, после чего распакован и запущен;
![скриншот](report/Screenshot1.png "Установка Java и запуск Protégé") \
Рисунок 1 - Установка Java \
![скриншот](report/Screenshot4.png "Запуск Protégé") \
Рисунок 2 - Запуск Protégé

2. Загружена, изучена и сохранена образовательная онтология;
![скриншот](report/Screenshot7.png "Главная страница онтологии") \
Рисунок 3 - Главная страница онтологии \
![скриншот](report/Screenshot10.png "Иерархия классов") \
Рисунок 4 - Иерархия классов \
![скриншот](report/Screenshot11.png "Изучение объектов") \
Рисунок 5 - Изучение объектов \
![скриншот](report/Screenshot12.png "Загрузка онтологии") \
Рисунок 6 - Загрузка онтологии

3. Создан файл ontology_analysis.txt и записаны основные наблюдения;
![скриншот](report/Screenshot51.png "Запись наблюдений") \
Рисунок 7 - Запись наблюдений

4. В класс Pizza добавлен субкласс RussianPizza, добавлены аннотация и ограничители для RussianPizza, добавлено объектное свойство; \
![скриншот](report/Screenshot14.png "Добавление субкласса") \
Рисунок 8 - Добавление субкласса \
![скриншот](report/Screenshot17.png "Финальный вид RussianPizza") \
Рисунок 9 - Финальный вид RussianPizza \
![скриншот](report/Screenshot21.png "Финальный вид объектного свойство") \
Рисунок 10 - Финальный вид объектного свойство

5. Запущен Reasoner, изучена автоматическая классификация пицц и выполнен запрос в DL Query на поиск пиццы с грибами;
![скриншот](report/Screenshot22.png "Запуск Reasoner") \
Рисунок 11 - Запуск Reasoner \
![скриншот](report/Screenshot23.png "Проверка автоматической классификации пицц") \
Рисунок 12 - Проверка автоматической классификации пицц \
![скриншот](report/Screenshot25.png "Поиск пицц с грибами") \
Рисунок 13 - Поиск пицц с грибами

6. Сохранена онтология с добавленными данными;
7. Написан и выполнен скрипт report_ontology.py для создания отчёта;
![скриншот](report/Screenshot28.png "Результат в командной строке") \
Рисунок 14 - Результат в командной строке \
![скриншот](report/Screenshot29.png "Результат в сохранённом файле") \
Рисунок 15 - Результат в сохранённом файле

8. Установлена, распакована и запущена утилита Apache Jena Fuseki;
![скриншот](report/Screenshot30.png "Установка Apache Jena Fuseki") \
Рисунок 16 - Установка Apache Jena Fuseki \
![скриншот](report/Screenshot31.png "Распаковка и запуск Apache Jena Fuseki") \
Рисунок 17 - Распаковка и запуск Apache Jena Fuseki

9. В Apache Jena Fuseki создан датасет, загружены RDF данные, выполнен SPARQL запрос с получением подсчёта всех триплетов;
![скриншот](report/Screenshot32.png "Создание датасета") \
Рисунок 18 - Создание датасета \
![скриншот](report/Screenshot33.png "Создание датасета") \
Рисунок 19 - Загрузка данных \
![скриншот](report/Screenshot35.png "Запрос") \
![скриншот](report/Screenshot35-2.png "Результат") \
Рисунок 20 - Выполнение запроса

11. Написаны и выполнены запросы в sparql_queries.py, получены табличные результаты и подсчёты триплетов;
![скриншот](report/Screenshot37.png "Полученные результаты") \
![скриншот](report/Screenshot38.png "Полученные результаты") \
![скриншот](report/Screenshot39.png "Полученные результаты") \
![скриншот](report/Screenshot41.png "Полученные результаты") \
![скриншот](report/Screenshot42.png "Полученные результаты") \
![скриншот](report/Screenshot44.png "Полученные результаты") \
Рисунок 21 - Полученные результаты

12. Установлены transformers, SPARQLWrapper, rdflib, openai; 
![скриншот](report/Screenshot45.png "Установка новых пакетов") \
Рисунок 22 - Установка новых пакетов

13. Написан скрипт генерации SPARQL из текста и проверена его работа;
14. При выполнении скрипта возникли проблемы со скачиванием модели, поэтому они установдены вручную; \
![скриншот](report/Screenshot47.png "Ошибка скачивания модели") \
Рисунок 23 - Ошибка скачивания модели

15. При выполнении кода возникла ошибка "Out Of Memory" из-за недостатка оперативной памяти;
![скриншот](report/Screenshot48.png "Ошибка Out Of Memory") \
Рисунок 24 - Ошибка Out Of Memory

17. Поскольку увеличить количество RAM не является возможным на данный момент, для предотвращения ошибки произведены попытки использования альтернативной модели с меньшей требовательностью и использования модели в щадящем режиме; \
![скриншот](report/Screenshot49.png "Использование альтернативной модели") \
Рисунок 25 - Использование альтернативной модели \
![скриншот](report/Screenshot50.png "Использование требовательной модели в щадящем") \
Рисунок 26 - Использование требовательной модели в щадящем

##### Тестирование
- [x] Модульные тесты - tests/test_pipeline.py проверяет корректность интеграции retriever -> generator.
- [x] Интеграционные тесты - запуск end‑to‑end на тестовой выборке.
- [x] Нагрузочные тесты - базовый стресс‑тест (параллельные запросы) показал деградацию при >50 concurrent requests на CPU; рекомендовано масштабирование на GPU/cluster.

##### Ключевые файлы
- documents/ -	исходные документы и предобработанные фрагменты
- retriever/vector_store.py -	векторное хранилище (FAISS wrapper)
- generator/llm_client.py	- клиент для LLM (генерация)
- pipeline/rag_pipeline.py	- основной RAG‑пайплайн
- evaluation/	- скрипты для оценки и визуализации

##### Выводы
В процессе выполнения лабораторной работы №6 были освоены базовые принципы построения RAG (Retrieval-Augmented Generation) систем путем интеграции векторного поиска и языковых моделей; продвинутых техник настройки и оптимизации языковых моделей для использования в качестве генеративного компонента RAG-системы; реализации полного конвейера RAG-системы, интегрирующего семантического поиска и генерации ответов; методов и метрик для комплексной оценки качества RAG-системы. Получены практические навыки разработки работающего прототипа системы, способной находить релевантную информацию в базе знаний и генерировать осмысленные ответы на основе извлеченного контекста; работы с различными архитектурами моделей, оптимизации промптинга и оценки качества генерации; проектирования надежных пайплайнов, обработку ошибок, мониторинг производительности и оптимизацию взаимодействия между компонентами системы; проведения объективного тестирования ретривера и генератора, анализирования результатов и выявления направления для улучшения системы.

##### Приложения
- Скрипты main.py, test_rag_system.py, documents/tech_docs.py, retriever/vector_store.py, generator/llm_client.py находятся в папке в lab-6-1.
- Скрипты main_optimized.py, generate_report, retriever/vector_store.py, gdocuments/tech_docs.py и generator/* находятся в папке в lab-6-2.
- Скрипты pipeline/rag_pipeline.py, api/pipeline_service.py, config/pipeline_config.py, tests/test_pipeline.py находятся в папке в lab-6-3.
- Скрипты evaluation/* и run_evaluation.py, а также папка с полученными отчётами в папке rag-system/reports находятся в папке в lab-6-4;
- Скриншоты помещены в папку report.
