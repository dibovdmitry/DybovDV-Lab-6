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
