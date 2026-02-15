"""
Скрипт подготовки данных: генерация эмбеддингов из CSV файла решений ФАС.
Использует Google Gemini API для создания эмбеддингов.
Запуск: python prepare_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from embeddings import EmbeddingService
from config import Config


def load_csv() -> pd.DataFrame:
    """Загрузка CSV файла с решениями ФАС."""
    # Попробовать разные пути к CSV
    possible_paths = [
        Path(__file__).parent.parent / "fas_ad_practice_dataset.csv",
        Path(__file__).parent.parent / "Legal" / "fas_ad_practice_dataset.csv",
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError(f"CSV файл не найден. Проверьте пути: {[str(p) for p in possible_paths]}")
    
    print(f"Загрузка CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    print(f"Загружено {len(df)} записей")
    return df


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Объединение текстовых полей для создания эмбеддингов."""
    texts = []
    for _, row in df.iterrows():
        # Объединяем все важные текстовые поля для лучшего поиска
        parts = []
        
        # Основное содержание рекламы (самое важное)
        if pd.notna(row.get("ad_content_cited")):
            parts.append(f"Реклама: {row['ad_content_cited']}")
        
        # Описание рекламы
        if pd.notna(row.get("ad_description")):
            parts.append(f"Описание рекламы: {row['ad_description']}")
        
        # Суть нарушения (критически важно для поиска)
        if pd.notna(row.get("violation_summary")):
            parts.append(f"Нарушение: {row['violation_summary']}")
        
        # Аргументы ФАС (содержит юридическое обоснование - очень важно)
        if pd.notna(row.get("FAS_arguments")):
            args = str(row['FAS_arguments'])
            # Извлекаем ключевой тезис если он есть
            if "Ключевой тезис:" in args:
                thesis = args.split("Ключевой тезис:")[1].split("Юридическое")[0].strip()
                parts.append(f"Обоснование ФАС: {thesis}")
            else:
                # Берем первые 500 символов если нет структуры
                parts.append(f"Обоснование ФАС: {args[:500]}")
        
        # Нарушенные статьи закона (важно для юридических запросов)
        if pd.notna(row.get("legal_provisions")):
            legal = str(row['legal_provisions'])
            # Очищаем от лишних символов если это список
            legal = legal.replace('[', '').replace(']', '').replace("'", '')
            parts.append(f"Нарушенные статьи: {legal}")
        
        # Тематические теги (важно для категоризации)
        if pd.notna(row.get("thematic_tags")):
            tags = str(row['thematic_tags'])
            parts.append(f"Теги: {tags}")
        
        # Отрасль ответчика (контекст)
        if pd.notna(row.get("defendant_industry")):
            parts.append(f"Отрасль: {row['defendant_industry']}")
        
        # Платформа размещения (контекст)
        if pd.notna(row.get("ad_platform")):
            parts.append(f"Платформа: {row['ad_platform']}")
        
        # Тип нарушения
        if pd.notna(row.get("Violation_Type")):
            violation_type = "нарушение содержания" if str(row['Violation_Type']) == "substance" else "нарушение размещения"
            parts.append(f"Тип: {violation_type}")
        
        text = " ".join(parts) if parts else "Нет данных"
        texts.append(text)
    
    return texts


def prepare_separate_field_texts(df: pd.DataFrame) -> dict:
    """
    Подготовить отдельные тексты для каждого поля с эмбеддингами.
    
    Returns:
        dict с ключами 'FAS_arguments', 'violation_summary', 'ad_description'
        и списками текстов для каждого поля
    """
    fas_args_texts = []
    violation_texts = []
    ad_desc_texts = []
    
    for _, row in df.iterrows():
        # FAS_arguments - самое важное поле
        if pd.notna(row.get("FAS_arguments")):
            args = str(row['FAS_arguments'])
            if "Ключевой тезис:" in args:
                thesis = args.split("Ключевой тезис:")[1].split("Юридическое")[0].strip()
                fas_args_texts.append(thesis)
            else:
                fas_args_texts.append(args[:500])
        else:
            fas_args_texts.append("")
        
        # violation_summary
        if pd.notna(row.get("violation_summary")):
            violation_texts.append(str(row['violation_summary']))
        else:
            violation_texts.append("")
        
        # ad_description
        if pd.notna(row.get("ad_description")):
            ad_desc_texts.append(str(row['ad_description']))
        else:
            ad_desc_texts.append("")
    
    return {
        'FAS_arguments': fas_args_texts,
        'violation_summary': violation_texts,
        'ad_description': ad_desc_texts
    }


def generate_embeddings(texts: list[str], service: EmbeddingService) -> np.ndarray:
    """Генерация эмбеддингов для списка текстов через Google Gemini."""
    print(f"Генерация эмбеддингов для {len(texts)} текстов...")
    
    embeddings = service.embed_documents(texts)
    
    print(f"Размерность эмбеддингов: {embeddings.shape}")
    return embeddings


def prepare_cases(df: pd.DataFrame) -> list[dict]:
    """Подготовка данных кейсов для JSON."""
    cases = []
    
    # Колонки для сохранения
    columns = [
        "docId", "Violation_Type", "document_date", "FASbd_link",
        "FAS_division", "violation_found", "defendant_name",
        "defendant_industry", "ad_description", "ad_content_cited",
        "ad_platform", "violation_summary", "FAS_arguments",
        "legal_provisions", "thematic_tags"
    ]
    
    for idx, row in df.iterrows():
        case = {"index": idx}
        for col in columns:
            value = row.get(col)
            if pd.isna(value):
                case[col] = None
            else:
                case[col] = str(value)
        cases.append(case)
    
    return cases


def main():
    """Главная функция подготовки данных."""
    # Создаем директорию для данных
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Загружаем CSV
    df = load_csv()
    
    # Инициализируем сервис эмбеддингов
    print("Инициализация сервиса эмбеддингов...")
    embedding_service = EmbeddingService()
    
    # Подготавливаем тексты для основного эмбеддинга (объединенный)
    texts = prepare_texts(df)
    
    # Генерируем основные эмбеддинги
    embeddings = generate_embeddings(texts, embedding_service)
    
    # Сохраняем основные эмбеддинги
    embeddings_path = data_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Эмбеддинги сохранены: {embeddings_path}")
    
    # Также генерируем отдельные эмбеддинги для каждого поля
    # (для взвешенной релевантности)
    print("\nГенерация отдельных эмбеддингов для полей...")
    field_texts = prepare_separate_field_texts(df)
    
    for field_name, field_texts_list in field_texts.items():
        print(f"  - {field_name}...")
        field_embeddings = generate_embeddings(field_texts_list, embedding_service)
        field_path = data_dir / f"embeddings_{field_name}.npy"
        np.save(field_path, field_embeddings)
        print(f"    Сохранены: {field_path}")
    
    # Подготавливаем и сохраняем кейсы
    cases = prepare_cases(df)
    cases_path = data_dir / "cases.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Кейсы сохранены: {cases_path}")
    
    print("\nПодготовка данных завершена!")
    print(f"  - Основные эмбеддинги: {embeddings.shape[0]} векторов размерности {embeddings.shape[1]}")
    print(f"  - Кейсы: {len(cases)} записей")


if __name__ == "__main__":
    main()
