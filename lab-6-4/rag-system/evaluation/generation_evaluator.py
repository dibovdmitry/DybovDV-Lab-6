import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List

from .test_dataset import TestCase


class GenerationEvaluator:
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate_generation(self, generated_answer: str, ground_truth: str) -> Dict[str, float]:
        """Оценка качества сгенерированного ответа"""
        # ROUGE метрики
        rouge_results = self.rouge.compute(
            predictions=[generated_answer],
            references=[ground_truth],
        )

        # BLEU метрика
        bleu_results = self.bleu.compute(
            predictions=[generated_answer],
            references=[[ground_truth]],
        )

        # Семантическая схожесть
        semantic_similarity = self._calculate_semantic_similarity(generated_answer, ground_truth)

        # Длина ответа (простейшая метрика качества)
        answer_length = len(generated_answer.split())

        return {
            "rouge1": rouge_results.get("rouge1", 0.0),
            "rouge2": rouge_results.get("rouge2", 0.0),
            "rougeL": rouge_results.get("rougeL", 0.0),
            "bleu": bleu_results.get("bleu", 0.0),
            "semantic_similarity": semantic_similarity,
            "answer_length": answer_length,
        }

    def evaluate_answers(self, test_cases: List[TestCase], generated_answers: List[str]) -> Dict[str, Any]:
        """Оценка набора сгенерированных ответов"""
        results: List[Dict[str, Any]] = []
        for test_case, generated_answer in zip(test_cases, generated_answers):
            evaluation = self.evaluate_generation(generated_answer, test_case.ground_truth_answer)
            result = {
                "question": test_case.question,
                "generated_answer": generated_answer,
                "ground_truth": test_case.ground_truth_answer,
                "category": test_case.category,
                "difficulty": test_case.difficulty,
            }
            result.update(evaluation)
            results.append(result)

        # Агрегированные метрики
        aggregated = self._aggregate_generation_metrics(results)
        return {
            "detailed_results": results,
            "aggregated_metrics": aggregated,
        }

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Вычисление семантической схожести через эмбеддинги"""
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def _aggregate_generation_metrics(self, results: List[Dict]) -> Dict[str, float]:
        metrics = ["rouge1", "rouge2", "rougeL", "bleu", "semantic_similarity"]
        aggregated: Dict[str, float] = {}
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results]
            aggregated[f"mean_{metric}"] = float(np.mean(values)) if values else 0.0
            aggregated[f"std_{metric}"] = float(np.std(values)) if values else 0.0
            aggregated[f"min_{metric}"] = float(np.min(values)) if values else 0.0
            aggregated[f"max_{metric}"] = float(np.max(values)) if values else 0.0

        # Дополнительные метрики по длине ответов
        lengths = [r.get("answer_length", 0) for r in results]
        aggregated["mean_answer_length"] = float(np.mean(lengths)) if lengths else 0.0
        aggregated["std_answer_length"] = float(np.std(lengths)) if lengths else 0.0
        aggregated["min_answer_length"] = float(np.min(lengths)) if lengths else 0.0
        aggregated["max_answer_length"] = float(np.max(lengths)) if lengths else 0.0

        return aggregated
