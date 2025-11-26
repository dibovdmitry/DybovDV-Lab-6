import asyncio
from datetime import datetime

from pipeline.rag_pipeline import RAGPipeline
from evaluation.rag_evaluator import RAGEvaluator
from evaluation.visualization import ResultsVisualizer


async def main():
    print("🚀 Запуск комплексной оценки RAG-системы")
    print("=" * 50)

    # Инициализация системы
    pipeline = RAGPipeline()
    evaluator = RAGEvaluator(pipeline)
    visualizer = ResultsVisualizer()

    # Запуск оценки
    print("📋 Выполнение оценки...")
    report = await evaluator.run_comprehensive_evaluation()

    # Анализ по категориям
    print("📊 Анализ по категориям...")
    category_results = await evaluator.evaluate_by_category()

    # Анализ по сложности
    print("🎯 Анализ по сложности...")
    difficulty_results = await evaluator.evaluate_by_difficulty()

    # Сохранение отчетов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Основной отчет
    evaluator.save_report(report, f"reports/full_evaluation_{timestamp}.json")

    # Визуализация
    visualizer.plot_metrics_comparison(report, f"reports/metrics_comparison_{timestamp}.png")
    visualizer.plot_category_analysis(category_results, f"reports/category_analysis_{timestamp}.png")

    # Вывод сводки
    summary = evaluator.generate_summary(report)
    print(summary)
    print(f"\n✅ Оценка завершена! Отчеты сохранены в папке 'reports/'")


if __name__ == "__main__":
    asyncio.run(main())
