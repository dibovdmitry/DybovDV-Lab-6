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
