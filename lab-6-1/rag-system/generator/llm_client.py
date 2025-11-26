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
