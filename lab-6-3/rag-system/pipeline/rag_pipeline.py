import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from retriever.vector_store import VectorStore
from generator.optimized_generator import OptimizedLLMGenerator
from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.retriever = VectorStore(self.config.retriever.collection_name)
        self.generator = OptimizedLLMGenerator(self.config.generator.model_name)
        self.cache: Dict[str, Dict[str, Any]] = {}  # Простой in-memory кэш для демонстрации
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
        }

    async def process_question(self, question: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Основной метод обработки вопроса через RAG-конвейер"""
        start_time = time.time()
        request_id = self._generate_request_id(question, user_context)
        logger.info(f"Processing request {request_id}: {question}")
        self.metrics["total_requests"] += 1

        try:
            # Шаг 1: Проверка кэша
            cached_result = self._get_cached_result(request_id)
            if cached_result:
                logger.info(f"Cache hit for request {request_id}")
                self.metrics["cache_hits"] += 1
                cached_result["cached"] = True
                return cached_result

            # Шаг 2: Семантический поиск
            retrieval_start = time.time()
            retrieved_docs = await self._retrieve_documents(question)
            retrieval_time = time.time() - retrieval_start

            # Шаг 3: Фильтрация и ранжирование
            filtered_docs = self._filter_documents(retrieved_docs)

            # Шаг 4: Генерация ответа
            generation_start = time.time()
            answer = await self._generate_answer(question, filtered_docs, user_context)
            generation_time = time.time() - generation_start

            # Шаг 5: Постобработка
            final_answer = self._postprocess_answer(answer, filtered_docs)

            # Формирование результата
            processing_time = time.time() - start_time
            result = self._build_response(
                question=question,
                answer=final_answer,
                documents=filtered_docs,
                processing_time=processing_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                request_id=request_id,
            )

            # Кэширование результата
            self._cache_result(request_id, result)
            self.metrics["successful_requests"] += 1
            self._update_metrics(processing_time)
            logger.info(f"Request {request_id} completed in {processing_time:.2f}s")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_result = self._build_error_response(question, str(e), processing_time, request_id)
            logger.error(f"Request {request_id} failed: {e}")
            return error_result

    async def _retrieve_documents(self, question: str) -> List[Dict[str, Any]]:
        """Асинхронный поиск документов с таймаутом"""
        try:
            loop = asyncio.get_event_loop()
            documents = await asyncio.wait_for(
                loop.run_in_executor(None, self.retriever.search, question, self.config.retriever.top_k),
                timeout=self.config.generator.timeout_seconds,
            )
            return documents
        except asyncio.TimeoutError:
            logger.warning("Document retrieval timeout")
            return []
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    async def _generate_answer(self, question: str, documents: List[Dict[str, Any]], user_context: Dict[str, Any]) -> str:
        """Асинхронная генерация ответа с обработкой ошибок"""
        try:
            if not documents:
                return "К сожалению, я не нашел достаточно информации для ответа на этот вопрос."

            loop = asyncio.get_event_loop()
            generation_result = await asyncio.wait_for(
                loop.run_in_executor(None, self.generator.generate_optimized_response, question, documents),
                timeout=self.config.generator.timeout_seconds,
            )
            return generation_result.get("answer", "")
        except asyncio.TimeoutError:
            logger.warning("Answer generation timeout")
            return "Извините, генерация ответа заняла слишком много времени. Попробуйте переформулировать вопрос."
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Произошла ошибка при генерации ответа: {str(e)}"

    def _filter_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Фильтрация и ранжирование документов"""
        # Фильтрация по порогу схожести
        filtered = [
            doc
            for doc in documents
            if doc.get("similarity_score", 0) >= self.config.retriever.similarity_threshold
        ]

        # Сортировка по релевантности
        filtered.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

        # Ограничение длины контекста
        total_length = 0
        final_documents: List[Dict[str, Any]] = []
        for doc in filtered:
            doc_length = len(doc.get("content", ""))
            if total_length + doc_length <= self.config.retriever.max_context_length:
                final_documents.append(doc)
                total_length += doc_length
            else:
                break

        return final_documents

    def _postprocess_answer(self, answer: str, documents: List[Dict[str, Any]]) -> str:
        """Постобработка сгенерированного ответа"""
        # Удаление лишних пробелов и переносов
        answer = " ".join(answer.split())

        # Проверка минимальной длины ответа
        if len(answer.strip()) < 10:
            return "Извините, не удалось сгенерировать содержательный ответ на основе найденной информации."
        return answer

    def _build_response(self, **kwargs) -> Dict[str, Any]:
        """Формирование структурированного ответа"""
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }

    def _build_error_response(self, question: str, error: str, processing_time: float, request_id: str) -> Dict[str, Any]:
        """Формирование ответа об ошибке"""
        return {
            "success": False,
            "question": question,
            "answer": f"Произошла ошибка: {error}",
            "processing_time": processing_time,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "documents": [],
        }

    def _generate_request_id(self, question: str, user_context: Dict[str, Any]) -> str:
        """Генерация уникального ID запроса"""
        content = question + json.dumps(user_context or {}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:10]

    def _get_cached_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Получение результата из кэша"""
        if not self.config.enable_caching:
            return None
        cached = self.cache.get(request_id)
        if cached and time.time() - cached["timestamp"] < self.config.cache_ttl:
            return cached["result"]
        return None

    def _cache_result(self, request_id: str, result: Dict[str, Any]):
        """Сохранение результата в кэш"""
        if self.config.enable_caching:
            self.cache[request_id] = {"result": result, "timestamp": time.time()}

        # Очистка устаревших записей (простая реализация)
        if len(self.cache) > 1000:  # Максимум 1000 записей в кэше
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

    def _update_metrics(self, processing_time: float):
        """Обновление метрик производительности"""
        if self.metrics["successful_requests"] <= 1:
            # первое значение
            self.metrics["average_processing_time"] = processing_time
            return
        total_time = self.metrics["average_processing_time"] * (self.metrics["successful_requests"] - 1)
        self.metrics["average_processing_time"] = (total_time + processing_time) / self.metrics["successful_requests"]

    def get_metrics(self) -> Dict[str, Any]:
        """Получение текущих метрик системы"""
        return self.metrics.copy()
