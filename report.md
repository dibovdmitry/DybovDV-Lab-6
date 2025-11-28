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

```

- Скрипт test_rag_system.py
```python

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
