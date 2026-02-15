"""
Скрипт подготовки данных: генерация эмбеддингов из CSV файла решений ФАС.
Запуск: python prepare_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Пути к файлам
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = BASE_DIR.parent / "fas_ad_practice_dataset.csv"

# Модель для русского языка
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_csv() -> pd.DataFrame:
    """Загрузка CSV файла с решениями ФАС."""
    print(f"Загрузка CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    print(f"Загружено {len(df)} записей")
    return df


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Объединение текстовых полей для создания эмбеддингов."""
    texts = []
    for _, row in df.iterrows():
        # Объединяем ключевые текстовые поля
        parts = []
        
        if pd.notna(row.get("ad_content_cited")):
            parts.append(f"Реклама: {row['ad_content_cited']}")
        
        if pd.notna(row.get("violation_summary")):
            parts.append(f"Нарушение: {row['violation_summary']}")
        
        if pd.notna(row.get("ad_description")):
            parts.append(f"Описание: {row['ad_description']}")
        
        text = " ".join(parts) if parts else "Нет данных"
        texts.append(text)
    
    return texts


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Генерация эмбеддингов для списка текстов."""
    print(f"Загрузка модели: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Генерация эмбеддингов для {len(texts)} текстов...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Нормализация для косинусного сходства
    )
    
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
    DATA_DIR.mkdir(exist_ok=True)
    
    # Загружаем CSV
    df = load_csv()
    
    # Подготавливаем тексты
    texts = prepare_texts(df)
    
    # Генерируем эмбеддинги
    embeddings = generate_embeddings(texts)
    
    # Сохраняем эмбеддинги
    embeddings_path = DATA_DIR / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Эмбеддинги сохранены: {embeddings_path}")
    
    # Подготавливаем и сохраняем кейсы
    cases = prepare_cases(df)
    cases_path = DATA_DIR / "cases.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Кейсы сохранены: {cases_path}")
    
    print("\nПодготовка данных завершена!")
    print(f"  - Эмбеддинги: {embeddings.shape[0]} векторов размерности {embeddings.shape[1]}")
    print(f"  - Кейсы: {len(cases)} записей")


if __name__ == "__main__":
    main()
