"""
Сервис эмбеддингов.
Использует локальную модель sentence-transformers для запросов (так как Gemini API недоступен).
"""

import numpy as np
from typing import List, Optional

from config import Config


class EmbeddingService:
    """
    Сервис для создания эмбеддингов через sentence-transformers.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация сервиса эмбеддингов.
        """
        from sentence_transformers import SentenceTransformer
        
        # Используем мультиязычную модель для русского языка
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.dimension = 384
        
        print(f"Загрузка модели: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print(f"Модель загружена. Размерность эмбеддингов: {self.dimension}")
    
    def embed_text(self, text: str, task_type: str = "retrieval_query") -> np.ndarray:
        """
        Создать эмбеддинг для одного текста.
        
        Args:
            text: Текст для эмбеддинга
            task_type: Тип задачи (для совместимости, не влияет на результат)
        
        Returns:
            numpy array размерностью 384
        """
        if not text or not text.strip():
            return np.zeros(self.dimension)
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            print(f"Ошибка при создании эмбеддинга: {e}")
            return np.zeros(self.dimension)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Создать эмбеддинги для списка текстов (документов).
        
        Args:
            texts: Список текстов для эмбеддингов
        
        Returns:
            numpy array формы (len(texts), 384)
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Ошибка при создании эмбеддингов: {e}")
            return np.zeros((len(texts), self.dimension))
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Создать эмбеддинг для поискового запроса.
        
        Args:
            query: Поисковый запрос
        
        Returns:
            numpy array размерностью 384
        """
        return self.embed_text(query, task_type="retrieval_query")
    
    def embed_batch(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """
        Создать эмбеддинги для батча текстов.
        
        Args:
            texts: Список текстов
            task_type: Тип задачи
        
        Returns:
            numpy array формы (len(texts), 384)
        """
        return self.embed_documents(texts)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Нормализовать эмбеддинги (для косинусного сходства).
    
    Args:
        embeddings: Массив эмбеддингов
    
    Returns:
        Нормализованные эмбеддинги
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Вычислить косинусное сходство между двумя векторами.
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
    
    Returns:
        Косинусное сходство (от -1 до 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    Нормализовать оценку в диапазон [0, 1].
    
    Args:
        score: Оценка
        min_val: Минимальное значение
        max_val: Максимальное значение
    
    Returns:
        Нормализованная оценка в [0, 1]
    """
    if score < min_val:
        return 0.0
    if score > max_val:
        return 1.0
    return (score - min_val) / (max_val - min_val)
