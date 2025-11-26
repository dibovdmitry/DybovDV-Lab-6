import asyncio
import time
from pipeline.rag_pipeline import RAGPipeline
from config.pipeline_config import PipelineConfig


async def demonstrate_pipeline():
    """Демонстрация работы полного RAG-конвейера"""
    print("🚀 Запуск демонстрации RAG-конвейера")
    print("=" * 50)

    pipeline = RAGPipeline()

    # Тестовые вопросы разной сложности
    test_cases = [
        "Что такое машинное обучение?",
        "Объясни разницу между AI и ML",
        "Какие типы нейронных сетей используются в компьютерном зрении?",
        "Что такое RAG архитектура и как она работает?",
        "Расскажи о квантовых вычислениях",  # Тема, которой нет в базе знаний
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n📝 Тест {i}: {question}")
        print("-" * 40)

        start_time = time.time()
        result = await pipeline.process_question(question)
        end_time = time.time()

        print(f"✅ Успех: {result.get('success', False)}")
        print(f"⏱ Время обработки: {result.get('processing_time', end_time - start_time):.2f}с")
        print(f"🔍 Найдено документов: {len(result.get('documents', []))}")
        print(f"🤖 Ответ: {result.get('answer', '')}")
        print(f"📊 Из кэша: {result.get('cached', False)}")

        # Показ топ-документа если есть
        if result.get("documents"):
            best_doc = result["documents"][0]
            title = best_doc.get("metadata", {}).get("title", "без названия")
            score = best_doc.get("similarity_score", 0.0)
            print(f"📄 Лучший документ: {title}")
            print(f"🎯 Схожесть: {score:.3f}")

        print("-" * 40)

    # Показ метрик системы
    metrics = pipeline.get_metrics()
    print(f"\n📈 Метрики системы:")
    print(f"   Всего запросов: {metrics.get('total_requests', 0)}")
    print(f"   Успешных: {metrics.get('successful_requests', 0)}")
    print(f"   Среднее время: {metrics.get('average_processing_time', 0.0):.2f}с")
    print(f"   Попадания в кэш: {metrics.get('cache_hits', 0)}")


if __name__ == "__main__":
    asyncio.run(demonstrate_pipeline())
