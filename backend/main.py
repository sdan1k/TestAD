"""
FastAPI сервер для гибридного поиска по решениям ФАС.
Запуск: uvicorn main:app --reload --port 8000
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Конфигурация
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_FAS_ARGS_PATH = DATA_DIR / "embeddings_FAS_arguments.npy"
EMBEDDINGS_VIOLATION_PATH = DATA_DIR / "embeddings_violation_summary.npy"
EMBEDDINGS_AD_DESC_PATH = DATA_DIR / "embeddings_ad_description.npy"
CASES_PATH = DATA_DIR / "cases.json"

# Модель
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Веса для полей (для RAG)
FIELD_WEIGHTS = {
    'FAS_arguments': 1.0,
    'violation_summary': 0.8,
    'ad_description': 0.6,
    'ad_content_cited': 0.7,
    'legal_provisions': 0.5
}

# Количество кандидатов для переранжирования
SEARCH_TOP_CANDIDATES = 100


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Нормализация оценки в диапазон 0-1."""
    if score < min_val:
        return 0.0
    if score > max_val:
        return 1.0
    return (score - min_val) / (max_val - min_val)


# Глобальные переменные
model: Optional[SentenceTransformer] = None
embeddings: Optional[np.ndarray] = None
embeddings_fas_args: Optional[np.ndarray] = None
embeddings_violation: Optional[np.ndarray] = None
embeddings_ad_desc: Optional[np.ndarray] = None
cases: Optional[list[dict]] = None


# Pydantic модели
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Поисковый запрос")
    top_k: int = Field(default=20, ge=1, le=50, description="Количество результатов")
    year: Optional[List[int]] = Field(default=None, description="Фильтр по году")
    region: Optional[List[str]] = Field(default=None, description="Фильтр по региону")
    industry: Optional[List[str]] = Field(default=None, description="Фильтр по отрасли")
    article: Optional[List[str]] = Field(default=None, description="Фильтр по статье закона")


class CaseResult(BaseModel):
    index: int
    score: float
    field_scores: Optional[Dict[str, float]] = None
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
    results: List[CaseResult]
    filters_applied: Optional[dict] = None
    message: Optional[str] = None


class FilterOptions(BaseModel):
    years: List[int]
    regions: List[str]
    industries: List[str]
    articles: List[str]


# FastAPI приложение
app = FastAPI(
    title="FAS Hybrid Search API",
    description="API для гибридного поиска по решениям ФАС о нарушениях в рекламе",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_data():
    """Загрузка всех данных при старте."""
    global model, embeddings, embeddings_fas_args, embeddings_violation, embeddings_ad_desc, cases
    
    print("=" * 50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)
    
    print("Загрузка модели эмбеддингов...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Модель загружена: {MODEL_NAME}")
    
    # Основные эмбеддинги
    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        print(f"  Основные эмбеддинги: {embeddings.shape}")
    else:
        print(f"  ВНИМАНИЕ: Файл {EMBEDDINGS_PATH} не найден!")
    
    # Отдельные эмбеддинги для полей
    if EMBEDDINGS_FAS_ARGS_PATH.exists():
        embeddings_fas_args = np.load(EMBEDDINGS_FAS_ARGS_PATH)
        print(f"  FAS_arguments эмбеддинги: {embeddings_fas_args.shape}")
    
    if EMBEDDINGS_VIOLATION_PATH.exists():
        embeddings_violation = np.load(EMBEDDINGS_VIOLATION_PATH)
        print(f"  violation_summary эмбеддинги: {embeddings_violation.shape}")
    
    if EMBEDDINGS_AD_DESC_PATH.exists():
        embeddings_ad_desc = np.load(EMBEDDINGS_AD_DESC_PATH)
        print(f"  ad_description эмбеддинги: {embeddings_ad_desc.shape}")
    
    # Загрузка кейсов
    if CASES_PATH.exists():
        with open(CASES_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)
        print(f"  Кейсов загружено: {len(cases)}")
    else:
        print(f"  ВНИМАНИЕ: Файл {CASES_PATH} не найден!")
    
    print("=" * 50)
    print("ЗАГРУЗКА ЗАВЕРШЕНА")
    print("=" * 50)


@app.on_event("startup")
async def startup_event():
    """Загрузка данных при старте."""
    try:
        load_data()
    except Exception as e:
        print(f"ОШИБКА ЗАГРУЗКИ: {e}")
        raise


def apply_filters(candidates: List[tuple], filters: dict) -> List[tuple]:
    """Применение фильтров к результатам поиска."""
    if not filters or not cases:
        return candidates
    
    filtered = []
    for idx, score in candidates:
        case = cases[idx]
        
        # Фильтр по году
        if filters.get('year'):
            if not case.get('document_date'):
                continue
            try:
                case_year = int(case['document_date'][:4])
                if case_year not in filters['year']:
                    continue
            except:
                continue
        
        # Фильтр по региону
        if filters.get('region'):
            if not case.get('FAS_division') or case['FAS_division'] not in filters['region']:
                continue
        
        # Фильтр по отрасли
        if filters.get('industry'):
            if not case.get('defendant_industry') or case['defendant_industry'] not in filters['industry']:
                continue
        
        # Фильтр по статье
        if filters.get('article'):
            if not case.get('legal_provisions'):
                continue
            legal = case['legal_provisions']
            found = False
            for art in filters['article']:
                if art in legal:
                    found = True
                    break
            if not found:
                continue
        
        filtered.append((idx, score))
    
    return filtered


def keyword_search(query: str, top_k: int = 200) -> List[tuple]:
    """Поиск по ключевым словам в текстовых полях."""
    if not cases:
        return []
    
    # Важные поля для поиска
    search_fields = ['FAS_arguments', 'violation_summary', 'ad_description', 'ad_content_cited', 'legal_provisions']
    
    # Токенизация запроса
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    scores = []
    for idx, case in enumerate(cases):
        score = 0
        for field in search_fields:
            text = case.get(field, '') or ''
            text_lower = text.lower()
            
            # Точное совпадение (больше вес)
            if query_lower in text_lower:
                score += 10
            
            # Частичное совпадение
            for word in query_words:
                if len(word) > 3 and word in text_lower:
                    score += 1
        
        if score > 0:
            scores.append((idx, score))
    
    # Сортировка по убыванию
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def semantic_search(query_embedding: np.ndarray, top_k: int) -> List[tuple]:
    """Семантический поиск по косинусному сходству."""
    if embeddings is None:
        return []
    
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def rerank_with_field_embeddings(candidates: List[tuple], query_embedding: np.ndarray) -> List[dict]:
    """Переранжирование кандидатов с использованием отдельных эмбеддингов полей."""
    if not cases:
        return []
    
    results = []
    max_weight_sum = sum(FIELD_WEIGHTS.values())
    
    for idx, base_score in candidates:
        case = cases[idx]
        field_scores = {}
        
        # FAS_arguments (самый важный)
        if embeddings_fas_args is not None and idx < len(embeddings_fas_args):
            fas_emb = embeddings_fas_args[idx]
            if np.any(fas_emb):
                r = cosine_similarity(query_embedding.reshape(1, -1), fas_emb.reshape(1, -1))[0][0]
                field_scores['FAS_arguments'] = normalize_score(r)
            else:
                field_scores['FAS_arguments'] = 0.0
        else:
            field_scores['FAS_arguments'] = normalize_score(base_score)
        
        # violation_summary
        if embeddings_violation is not None and idx < len(embeddings_violation):
            viol_emb = embeddings_violation[idx]
            if np.any(viol_emb):
                r = cosine_similarity(query_embedding.reshape(1, -1), viol_emb.reshape(1, -1))[0][0]
                field_scores['violation_summary'] = normalize_score(r)
            else:
                field_scores['violation_summary'] = 0.0
        else:
            field_scores['violation_summary'] = 0.0
        
        # ad_description
        if embeddings_ad_desc is not None and idx < len(embeddings_ad_desc):
            ad_emb = embeddings_ad_desc[idx]
            if np.any(ad_emb):
                r = cosine_similarity(query_embedding.reshape(1, -1), ad_emb.reshape(1, -1))[0][0]
                field_scores['ad_description'] = normalize_score(r)
            else:
                field_scores['ad_description'] = 0.0
        else:
            field_scores['ad_description'] = 0.0
        
        # Взвешенная сумма
        weighted_sum = sum(
            FIELD_WEIGHTS.get(field, 0) * score 
            for field, score in field_scores.items()
        )
        
        final_score = normalize_score(weighted_sum, min_val=0.0, max_val=max_weight_sum)
        
        results.append({
            'index': idx,
            'score': final_score,
            'base_score': base_score,
            'field_scores': field_scores,
            'case': case
        })
    
    # Сортировка по убыванию
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Гибридный поиск по решениям ФАС.
    
    Архитектура:
    1. Семантический поиск (TOP-100)
    2. Поиск по ключевым словам (TOP-50)
    3. Объединение и переранжирование
    4. Применение фильтров
    5. Возврат TOP-K результатов
    """
    global model, embeddings, cases
    
    # Проверка готовности
    if model is None or embeddings is None or cases is None:
        raise HTTPException(status_code=503, detail="Сервер не готов. Данные не загружены.")
    
    # Подготовка фильтров
    filters = {}
    if request.year:
        filters['year'] = request.year
    if request.region:
        filters['region'] = request.region
    if request.industry:
        filters['industry'] = request.industry
    if request.article:
        filters['article'] = request.article
    
    # Шаг 1: Эмбеддинг запроса
    query_embedding = model.encode(
        request.query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Шаг 2: Семантический поиск
    semantic_results = semantic_search(query_embedding, SEARCH_TOP_CANDIDATES)
    
    # Шаг 3: Поиск по ключевым словам
    keyword_results = keyword_search(request.query, top_k=50)
    
    # Шаг 4: Объединение результатов
    # Объединяем, учитывая оба типа поиска
    combined_scores = {}
    for idx, score in semantic_results:
        combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.7
    
    for idx, score in keyword_results:
        combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.3
    
    # Топ-100 объединенных
    sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:SEARCH_TOP_CANDIDATES]
    
    # Шаг 5: Применение фильтров
    if filters:
        filtered_candidates = apply_filters(sorted_candidates, filters)
    else:
        filtered_candidates = sorted_candidates
    
    # Шаг 6: Переранжирование с использованием полей
    reranked = rerank_with_field_embeddings(filtered_candidates, query_embedding)
    
    # Шаг 7: Финальный топ-K
    final_results = reranked[:request.top_k]
    
    # Формирование ответа
    case_results = []
    for result in final_results:
        case_data = result['case'].copy()
        case_data['score'] = round(result['score'], 4)
        case_data['field_scores'] = {k: round(v, 4) for k, v in result.get('field_scores', {}).items()}
        case_results.append(CaseResult(**case_data))
    
    return SearchResponse(
        query=request.query,
        total_cases=len(cases),
        results=case_results,
        filters_applied=filters if filters else None,
        message=None
    )


@app.get("/api/filters", response_model=FilterOptions)
async def get_filter_options():
    """Получить доступные значения для фильтров."""
    if not cases:
        raise HTTPException(status_code=503, detail="Сервер не готов.")
    
    years = set()
    regions = set()
    industries = set()
    articles = set()
    
    for case in cases:
        # Год
        if case.get('document_date'):
            try:
                year = int(case['document_date'][:4])
                years.add(year)
            except:
                pass
        
        # Регион
        if case.get('FAS_division'):
            regions.add(case['FAS_division'])
        
        # Отрасль
        if case.get('defendant_industry'):
            industries.add(case['defendant_industry'])
        
        # Статьи
        if case.get('legal_provisions'):
            legal = case['legal_provisions']
            found_articles = re.findall(r'ст\.\s*\d+|ч\.\s*\d+\s*ст\.\s*\d+', legal, re.IGNORECASE)
            for art in found_articles:
                articles.add(art.strip())
    
    return FilterOptions(
        years=sorted(list(years), reverse=True),
        regions=sorted(list(regions)),
        industries=sorted(list(industries)),
        articles=sorted(list(articles))
    )


@app.get("/api/health")
async def health_check():
    """Проверка состояния сервера."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "data_loaded": embeddings is not None and cases is not None,
        "total_cases": len(cases) if cases else 0,
        "embedding_dimension": embeddings.shape[1] if embeddings is not None else 0
    }


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "name": "FAS Hybrid Search API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "search": "POST /api/search",
        "filters": "GET /api/filters"
    }
