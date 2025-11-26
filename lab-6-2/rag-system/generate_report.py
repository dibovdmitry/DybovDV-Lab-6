import pandas as pd
from generator.benchmark_system import ModelBenchmark


def create_model_comparison_report():
    """Создание отчета о сравнении моделей"""
    benchmark = ModelBenchmark()
    models_to_test = [
        "gpt2-medium",
        "t5-small",
        "facebook/bart-base",
        "microsoft/DialoGPT-medium",
    ]

    results_df = benchmark.run_benchmark(models_to_test)

    # Агрегация результатов
    summary = results_df.groupby("model").agg({
        "generation_time": "mean",
        "keyword_coverage": "mean",
        "success": "mean",
        "answer_length": "mean",
    }).round(3)

    summary = summary.rename(columns={
        "generation_time": "avg_generation_time",
        "keyword_coverage": "avg_keyword_coverage",
        "success": "success_rate",
        "answer_length": "avg_answer_length",
    })

    # Сохранение отчетов
    results_df.to_csv("model_comparison_detailed.csv", index=False)
    summary.to_csv("model_comparison_summary.csv")

    print("Детальный отчет сохранен в: model_comparison_detailed.csv")
    print("Сводный отчет сохранен в: model_comparison_summary.csv")

    return summary


if __name__ == "__main__":
    report = create_model_comparison_report()
    print("\nСводный отчет по моделям:")
    print(report)
