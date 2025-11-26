import numpy as np
from typing import Any, Dict, List
from sklearn.metrics import precision_score, recall_score  # kept for potential extensions
from .test_dataset import TestCase


class RetrievalEvaluator:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def evaluate_retrieval(self, test_case: TestCase, top_k: int = 5) -> Dict[str, float]:
        """Оценка качества поиска для одного тестового случая"""
        # Выполнение поиска
        retrieved_docs = self.vector_store.search(test_case.question, n_results=top_k)
        retrieved_ids = [doc["metadata"].get("id", "") for doc in retrieved_docs]

        # Бинарные метки релевантности
        true_relevant = set(test_case.relevant_document_ids)
        retrieved_set = set(retrieved_ids)

        # Расчет метрик
        precision = self._calculate_precision(retrieved_set, true_relevant)
        recall = self._calculate_recall(retrieved_set, true_relevant)
        f1 = self._calculate_f1(precision, recall)
        mrr = self._calculate_mrr(retrieved_ids, true_relevant)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mrr": mrr,
            "retrieved_count": len(retrieved_set),
            "relevant_count": len(true_relevant),
            "retrieved_ids": retrieved_ids,
        }

    def evaluate_dataset(self, test_cases: List[TestCase], top_k: int = 5) -> Dict[str, Any]:
        """Оценка на всем датасете"""
        results: List[Dict[str, Any]] = []
        for test_case in test_cases:
            result = self.evaluate_retrieval(test_case, top_k)
            result["question"] = test_case.question
            result["category"] = test_case.category
            result["difficulty"] = test_case.difficulty
            results.append(result)

        # Агрегированные метрики
        aggregated = self._aggregate_metrics(results)
        aggregated["total_cases"] = len(test_cases)
        return {
            "detailed_results": results,
            "aggregated_metrics": aggregated,
        }

    def _calculate_precision(self, retrieved: set, relevant: set) -> float:
        if len(retrieved) == 0:
            return 0.0
        return len(retrieved & relevant) / len(retrieved)

    def _calculate_recall(self, retrieved: set, relevant: set) -> float:
        if len(relevant) == 0:
            return 0.0
        return len(retrieved & relevant) / len(relevant)

    def _calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_mrr(self, retrieved_ids: List[str], relevant: set) -> float:
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        metrics = ["precision", "recall", "f1_score", "mrr"]
        aggregated: Dict[str, float] = {}
        for metric in metrics:
            values = [r[metric] for r in results if r.get(metric) is not None]
            aggregated[f"mean_{metric}"] = float(np.mean(values)) if values else 0.0
            aggregated[f"std_{metric}"] = float(np.std(values)) if values else 0.0
        return aggregated
