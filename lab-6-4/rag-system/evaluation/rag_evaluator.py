import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List
import inspect
import asyncio

import pandas as pd

from .test_dataset import TestCase, EvaluationDataset
from .retrieval_evaluator import RetrievalEvaluator
from .generation_evaluator import GenerationEvaluator
from pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.pipeline = rag_pipeline
        self.retrieval_evaluator = RetrievalEvaluator(rag_pipeline.retriever)
        self.generation_evaluator = GenerationEvaluator()
        self.dataset = EvaluationDataset()

    async def run_comprehensive_evaluation(self, test_cases: List[TestCase] = None) -> Dict[str, Any]:
        """Запуск комплексной оценки RAG-системы (async)"""
        if test_cases is None:
            test_cases = self.dataset.test_cases
        logger.info(f"Starting comprehensive evaluation with {len(test_cases)} test cases")

        # Этап 1: Оценка ретривера (синхронно)
        retrieval_results = self.retrieval_evaluator.evaluate_dataset(test_cases)

        # Этап 2: Получение ответов от полной системы
        generated_answers: List[str] = []
        for test_case in test_cases:
            try:
                # Поддержка как асинхронного, так и синхронного process_question
                if inspect.iscoroutinefunction(self.pipeline.process_question):
                    result = await self.pipeline.process_question(test_case.question)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, self.pipeline.process_question, test_case.question)
                generated_answers.append(result.get("answer", ""))
            except Exception as e:
                logger.error(f"Error processing question: {test_case.question}, error: {e}")
                generated_answers.append("")  # Пустой ответ в случае ошибки

        # Этап 3: Оценка генерации (синхронно)
        generation_results = self.generation_evaluator.evaluate_answers(test_cases, generated_answers)

        # Этап 4: Агрегация результатов
        final_report = self._compile_final_report(retrieval_results, generation_results, test_cases)
        return final_report

    async def evaluate_by_category(self) -> Dict[str, Any]:
        """Асинхронная оценка по категориям вопросов"""
        categories = set(case.category for case in self.dataset.test_cases)
        category_results: Dict[str, Any] = {}
        for category in categories:
            category_cases = self.dataset.get_cases_by_category(category)
            if category_cases:
                results = await self.run_comprehensive_evaluation(category_cases)
                category_results[category] = results.get("aggregated_metrics", {})
        return category_results

    async def evaluate_by_difficulty(self) -> Dict[str, Any]:
        """Асинхронная оценка по сложности вопросов"""
        difficulties = ["easy", "medium", "hard"]
        difficulty_results: Dict[str, Any] = {}
        for difficulty in difficulties:
            difficulty_cases = self.dataset.get_cases_by_difficulty(difficulty)
            if difficulty_cases:
                results = await self.run_comprehensive_evaluation(difficulty_cases)
                difficulty_results[difficulty] = results.get("aggregated_metrics", {})
        return difficulty_results

    def _compile_final_report(
        self,
        retrieval_results: Dict[str, Any],
        generation_results: Dict[str, Any],
        test_cases: List[TestCase],
    ) -> Dict[str, Any]:
        """Компиляция финального отчета"""
        aggregated_metrics = {
            "retrieval": retrieval_results.get("aggregated_metrics", {}),
            "generation": generation_results.get("aggregated_metrics", {}),
            "overall_score": self._calculate_overall_score(
                retrieval_results.get("aggregated_metrics", {}),
                generation_results.get("aggregated_metrics", {}),
            ),
        }

        detailed_results: List[Dict[str, Any]] = []
        for retrieval, generation in zip(
            retrieval_results.get("detailed_results", []), generation_results.get("detailed_results", [])
        ):
            combined = {**retrieval, **generation}
            detailed_results.append(combined)

        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_cases_count": len(test_cases),
            "aggregated_metrics": aggregated_metrics,
            "detailed_results": detailed_results,
            "dataset_statistics": self.dataset.get_statistics(),
        }
        return report

    def _calculate_overall_score(self, retrieval_metrics: Dict[str, Any], generation_metrics: Dict[str, Any]) -> float:
        """Расчет общего скора системы"""
        weights = {
            "retrieval_precision": 0.3,
            "retrieval_recall": 0.2,
            "generation_semantic_similarity": 0.3,
            "generation_rougeL": 0.2,
        }
        score = 0.0
        score += retrieval_metrics.get("mean_precision", 0.0) * weights["retrieval_precision"]
        score += retrieval_metrics.get("mean_recall", 0.0) * weights["retrieval_recall"]
        score += generation_metrics.get("mean_semantic_similarity", 0.0) * weights["generation_semantic_similarity"]
        score += generation_metrics.get("mean_rougeL", 0.0) * weights["generation_rougeL"]
        return float(score)

    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Сохранение отчета в файл (создаёт директорию, атомарная запись)"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/rag_evaluation_report_{timestamp}.json"

        # Убедиться, что директория существует
        dirpath = os.path.dirname(os.path.abspath(filename)) or "."
        os.makedirs(dirpath, exist_ok=True)

        # Запись в временный файл и переименование (атомарность)
        tmp_path = os.path.join(dirpath, f".tmp_{os.path.basename(filename)}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, filename)
            logger.info("Evaluation report saved to %s", filename)
        except Exception as e:
            logger.exception("Failed to save evaluation report to %s: %s", filename, e)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    def generate_summary(self, report: Dict[str, Any]) -> str:
        """Генерация текстовой сводки"""
        metrics = report.get("aggregated_metrics", {})
        retrieval = metrics.get("retrieval", {})
        generation = metrics.get("generation", {})
        overall = metrics.get("overall_score", 0.0)

        summary = (
            f"ОТЧЕТ ОЦЕНКИ RAG-СИСТЕМЫ\n"
            f"===========================\n"
            f"Общая информация:\n"
            f" - Время оценки: {report.get('evaluation_timestamp')}\n"
            f" - Количество тестовых случаев: {report.get('test_cases_count')}\n"
            f" - Общий score системы: {overall:.3f}\n\n"
            f"Качество поиска (Retrieval):\n"
            f" - Precision: {retrieval.get('mean_precision', 0.0):.3f} ± {retrieval.get('std_precision', 0.0):.3f}\n"
            f" - Recall: {retrieval.get('mean_recall', 0.0):.3f} ± {retrieval.get('std_recall', 0.0):.3f}\n"
            f" - F1-Score: {retrieval.get('mean_f1_score', 0.0):.3f} ± {retrieval.get('std_f1_score', 0.0):.3f}\n"
            f" - MRR: {retrieval.get('mean_mrr', 0.0):.3f} ± {retrieval.get('std_mrr', 0.0):.3f}\n\n"
            f"Качество генерации (Generation):\n"
            f" - ROUGE-L: {generation.get('mean_rougeL', 0.0):.3f} ± {generation.get('std_rougeL', 0.0):.3f}\n"
            f" - BLEU: {generation.get('mean_bleu', 0.0):.3f} ± {generation.get('std_bleu', 0.0):.3f}\n"
            f" - Semantic Similarity: {generation.get('mean_semantic_similarity', 0.0):.3f} ± {generation.get('std_semantic_similarity', 0.0):.3f}\n"
            f" - Средняя длина ответа: {generation.get('mean_answer_length', 0.0):.1f} слов\n\n"
            f"Рекомендации по улучшению:\n{self._generate_recommendations(metrics)}"
        )
        return summary

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """Генерация рекомендаций по улучшению"""
        recommendations: List[str] = []
        retrieval = metrics.get("retrieval", {})
        generation = metrics.get("generation", {})

        if retrieval.get("mean_precision", 0.0) < 0.7:
            recommendations.append("• Улучшить точность поиска: настроить эмбеддинг-модель или увеличить размер базы знаний")
        if retrieval.get("mean_recall", 0.0) < 0.6:
            recommendations.append("• Увеличить полноту поиска: рассмотреть использование гибридного поиска")
        if generation.get("mean_semantic_similarity", 0.0) < 0.7:
            recommendations.append("• Улучшить качество генерации: настроить промпты или использовать более мощную языковую модель")
        if generation.get("mean_rougeL", 0.0) < 0.4:
            recommendations.append("• Работать над соответствием эталонным ответам: добавить few-shot примеры в промпты")
        if not recommendations:
            recommendations.append("• Система показывает хорошие результаты! Рекомендуется продолжить мониторинг качества.")
        return "\n".join(recommendations)
