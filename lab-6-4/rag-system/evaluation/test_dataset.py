from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TestCase:
    question: str
    ground_truth_answer: str
    relevant_document_ids: List[str]  # ID документов, которые должны быть найдены
    category: str
    difficulty: str  # easy, medium, hard


# Эталонный датасет для оценки
EVALUATION_DATASET = [
    TestCase(
        question="Что такое машинное обучение?",
        ground_truth_answer=(
            "Машинное обучение — это область искусственного интеллекта, которая использует "
            "статистические методы для создания моделей, способных обучаться на данных и делать "
            "предсказания. Основные типы включают обучение с учителем, без учителя и с подкреплением."
        ),
        relevant_document_ids=["doc_001"],
        category="AI",
        difficulty="easy",
    ),
    TestCase(
        question="Какие архитектуры нейронных сетей используются в NLP?",
        ground_truth_answer=(
            "В обработке естественного языка используются архитектуры трансформеров, такие как BERT "
            "для понимания текста и GPT для генерации. Эти модели используют механизм внимания для учета "
            "контекста во всей входной последовательности."
        ),
        relevant_document_ids=["doc_003"],
        category="NLP",
        difficulty="medium",
    ),
    TestCase(
        question="Что такое RAG-архитектура и какие преимущества она дает?",
        ground_truth_answer=(
            "RAG (Retrieval-Augmented Generation) — это архитектура, которая сочетает поиск информации "
            "в векторной базе данных с генерацией текста языковой моделью. Это позволяет моделям работать "
            "с актуальными данными, снижает вероятность галлюцинаций и повышает точность ответов."
        ),
        relevant_document_ids=["doc_005"],
        category="Architecture",
        difficulty="hard",
    ),
    TestCase(
        question="Какие бывают типы машинного обучения?",
        ground_truth_answer=(
            "Основные типы машинного обучения: обучение с учителем (supervised learning), обучение без "
            "учителя (unsupervised learning) и обучение с подкреплением (reinforcement learning)."
        ),
        relevant_document_ids=["doc_001"],
        category="AI",
        difficulty="easy",
    ),
]


class EvaluationDataset:
    def __init__(self, test_cases: List[TestCase] = None):
        self.test_cases = test_cases or EVALUATION_DATASET

    def get_cases_by_category(self, category: str) -> List[TestCase]:
        return [case for case in self.test_cases if case.category == category]

    def get_cases_by_difficulty(self, difficulty: str) -> List[TestCase]:
        return [case for case in self.test_cases if case.difficulty == difficulty]

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total_cases": len(self.test_cases),
            "by_category": {},
            "by_difficulty": {},
        }
        for case in self.test_cases:
            stats["by_category"][case.category] = stats["by_category"].get(case.category, 0) + 1
            stats["by_difficulty"][case.difficulty] = stats["by_difficulty"].get(case.difficulty, 0) + 1
        return stats
