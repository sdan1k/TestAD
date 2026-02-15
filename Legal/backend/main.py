"""
FastAPI сервер для семантического поиска по решениям ФАС.
Запуск: uvicorn main:app --reload
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Пути к файлам данных
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CASES_PATH = DATA_DIR / "cases.json"

# Модель (та же, что использовалась для создания эмбеддингов)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Глобальные переменные для хранения данных
model: Optional[SentenceTransformer] = None
embeddings: Optional[np.ndarray] = None
cases: Optional[list[dict]] = None


# Pydantic модели
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Поисковый запрос")
    top_k: int = Field(default=10, ge=1, le=50, description="Количество результатов")


class CaseResult(BaseModel):
    index: int
    score: float
    docId: Optional[str] = None
    Violation_Type: Optional[str] = None
    document_date: Optional[str] = None
    FASbd_link: Optional[str] = None
    FAS_division: Optional[str] = None
    violation_found: Optional[str] = None
    defendant_name: Optional[str] = None
    defendant_industry: Optional[str] = None
    ad_description: Optional[str] = None
    ad_content_cited: Optional[str] = None
    ad_platform: Optional[str] = None
    violation_summary: Optional[str] = None
    FAS_arguments: Optional[str] = None
    legal_provisions: Optional[str] = None
    thematic_tags: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    total_cases: int
    results: list[CaseResult]


# FastAPI приложение
app = FastAPI(
    title="FAS Semantic Search API",
    description="API для семантического поиска по решениям ФАС о нарушениях в рекламе",
    version="1.0.0"
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Загрузка модели и данных при старте сервера."""
    global model, embeddings, cases
    
    print("Загрузка модели...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Загрузка эмбеддингов...")
    if not EMBEDDINGS_PATH.exists():
        raise RuntimeError(
            f"Файл эмбеддингов не найден: {EMBEDDINGS_PATH}\n"
            "Сначала запустите: python prepare_data.py"
        )
    embeddings = np.load(EMBEDDINGS_PATH)
    
    print("Загрузка кейсов...")
    if not CASES_PATH.exists():
        raise RuntimeError(
            f"Файл кейсов не найден: {CASES_PATH}\n"
            "Сначала запустите: python prepare_data.py"
        )
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    print(f"Загружено {len(cases)} кейсов с эмбеддингами размерности {embeddings.shape[1]}")


def cosine_similarity_search(query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """
    Поиск наиболее похожих документов по косинусному сходству.
    Эмбеддинги уже нормализованы, поэтому скалярное произведение = косинусное сходство.
    """
    # Вычисляем сходство со всеми документами
    similarities = np.dot(embeddings, query_embedding)
    
    # Находим индексы топ-K результатов
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Возвращаем пары (индекс, score)
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Семантический поиск по решениям ФАС.
    
    Принимает текстовый запрос и возвращает наиболее релевантные решения.
    """
    if model is None or embeddings is None or cases is None:
        raise HTTPException(status_code=503, detail="Сервер не готов. Данные не загружены.")
    
    # Генерируем эмбеддинг запроса
    query_embedding = model.encode(
        request.query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Ищем похожие документы
    results = cosine_similarity_search(query_embedding, request.top_k)
    
    # Формируем ответ
    case_results = []
    for idx, score in results:
        case_data = cases[idx].copy()
        case_data["score"] = round(score, 4)
        case_results.append(CaseResult(**case_data))
    
    return SearchResponse(
        query=request.query,
        total_cases=len(cases),
        results=case_results
    )


@app.get("/api/health")
async def health_check():
    """Проверка состояния сервера."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "data_loaded": embeddings is not None and cases is not None,
        "total_cases": len(cases) if cases else 0
    }


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API."""
    return {
        "name": "FAS Semantic Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "search": "POST /api/search"
    }
