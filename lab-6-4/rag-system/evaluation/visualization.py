import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Any, Dict
import numpy as np


class ResultsVisualizer:
    def __init__(self):
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_metrics_comparison(self, report: Dict[str, Any], save_path: str = None):
        """Визуализация сравнения метрик"""
        metrics = report["aggregated_metrics"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Сравнение метрик качества RAG-системы", fontsize=16)

        # Retrieval метрики
        retrieval_metrics = ["mean_precision", "mean_recall", "mean_f1_score", "mean_mrr"]
        retrieval_values = [metrics["retrieval"].get(m, 0.0) for m in retrieval_metrics]
        retrieval_labels = ["Precision", "Recall", "F1-Score", "MRR"]
        bars1 = ax1.bar(retrieval_labels, retrieval_values, alpha=0.7)
        ax1.set_title("Качество поиска (Retrieval)")
        ax1.set_ylim(0, 1)
        self._add_value_labels(ax1, bars1)

        # Generation метрики
        generation_metrics = ["mean_rouge1", "mean_rouge2", "mean_rougeL", "mean_bleu"]
        generation_values = [metrics["generation"].get(m, 0.0) for m in generation_metrics]
        generation_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
        bars2 = ax2.bar(generation_labels, generation_values, alpha=0.7)
        ax2.set_title("Качество генерации (Generation)")
        ax2.set_ylim(0, 1)
        self._add_value_labels(ax2, bars2)

        # Semantic similarity
        similarity_data = [metrics["generation"].get("mean_semantic_similarity", 0.0)]
        ax3.bar(["Semantic\nSimilarity"], similarity_data, alpha=0.7, color="green")
        ax3.set_title("Семантическая схожесть")
        ax3.set_ylim(0, 1)
        ax3.text(
            0,
            similarity_data[0] + 0.02,
            f"{similarity_data[0]:.3f}",
            ha="center",
            va="bottom",
        )

        # Overall score
        overall_data = [metrics.get("overall_score", 0.0)]
        ax4.bar(["Overall\nScore"], overall_data, alpha=0.7, color="orange")
        ax4.set_title("Общая оценка системы")
        ax4.set_ylim(0, 1)
        ax4.text(0, overall_data[0] + 0.02, f"{overall_data[0]:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"График сохранен в: {save_path}")
        plt.show()

    def plot_category_analysis(self, category_results: Dict[str, Any], save_path: str = None):
        """Анализ результатов по категориям"""
        categories = list(category_results.keys())

        # Подготовка данных
        precision_scores = [category_results[cat]["retrieval"].get("mean_precision", 0.0) for cat in categories]
        similarity_scores = [
            category_results[cat]["generation"].get("mean_semantic_similarity", 0.0) for cat in categories
        ]

        x = np.arange(len(categories))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, precision_scores, width, label="Precision", alpha=0.7)
        bars2 = ax.bar(x + width / 2, similarity_scores, width, label="Semantic Similarity", alpha=0.7)

        ax.set_xlabel("Категории вопросов")
        ax.set_ylabel("Score")
        ax.set_title("Сравнение качества по категориям вопросов")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 1)
        self._add_value_labels(ax, bars1)
        self._add_value_labels(ax, bars2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _add_value_labels(self, ax, bars):
        """Добавление значений на столбцы графика"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.3f}", ha="center", va="bottom")
