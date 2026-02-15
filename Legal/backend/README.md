# FAS Semantic Search API

FastAPI сервер для семантического поиска по решениям ФАС (Федеральной антимонопольной службы) о нарушениях в рекламе.

## Описание

Сервер использует технологию эмбеддингов (векторных представлений текста) для поиска семантически похожих документов. В отличие от обычного текстового поиска, система понимает смысл запроса и находит релевантные решения, даже если в них нет точного совпадения слов.

### Как это работает

1. **Эмбеддинги** — каждое решение ФАС преобразуется в вектор из 384 чисел с помощью нейросети
2. **Косинусное сходство** — при поиске запрос также преобразуется в вектор, и система находит наиболее близкие векторы из базы
3. **Ранжирование** — результаты сортируются по степени семантической близости

## Технологии

- **FastAPI** — современный асинхронный веб-фреймворк
- **Sentence Transformers** — библиотека для создания эмбеддингов
- **paraphrase-multilingual-MiniLM-L12-v2** — мультиязычная модель, поддерживающая русский язык
- **NumPy** — быстрые операции с векторами
- **Pandas** — обработка CSV данных

## Установка

### Требования

- Python 3.10+
- pip

### Шаги установки

```bash
# Клонирование репозитория
git clone https://github.com/Latarho/Legal.git
cd Legal/backend

# Создание виртуального окружения (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## Подготовка данных

Перед запуском сервера необходимо сгенерировать эмбеддинги из CSV файла:

```bash
python prepare_data.py
```

Этот скрипт:
1. Загружает `fas_ad_practice_dataset.csv` из корня проекта
2. Объединяет текстовые поля для каждого решения
3. Генерирует эмбеддинги с помощью модели (занимает 1-2 минуты)
4. Сохраняет результаты в папку `data/`:
   - `embeddings.npy` — матрица эмбеддингов (7283 × 384)
   - `cases.json` — метаданные всех решений

## Запуск сервера

### Development режим (с автоперезагрузкой)

```bash
uvicorn main:app --reload --port 8000
```

### Production режим

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Сервер будет доступен по адресу: http://localhost:8000

## API Endpoints

### `GET /`

Информация об API.

**Ответ:**
```json
{
  "name": "FAS Semantic Search API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/api/health",
  "search": "POST /api/search"
}
```

### `GET /api/health`

Проверка состояния сервера.

**Ответ:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "data_loaded": true,
  "total_cases": 7283
}
```

### `POST /api/search`

Семантический поиск по решениям ФАС.

**Запрос:**
```json
{
  "query": "скидка 90% на всё",
  "top_k": 10
}
```

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| `query` | string | Текст запроса (1-5000 символов) | — |
| `top_k` | integer | Количество результатов (1-50) | 10 |

**Ответ:**
```json
{
  "query": "скидка 90% на всё",
  "total_cases": 7283,
  "results": [
    {
      "index": 1234,
      "score": 0.8542,
      "docId": "abc-123-def",
      "Violation_Type": "substance",
      "document_date": "2024-01-15",
      "defendant_name": "ООО Рога и Копыта",
      "violation_summary": "Недостоверная реклама...",
      "FASbd_link": "https://br.fas.gov.ru/cases/...",
      ...
    }
  ]
}
```

### `GET /docs`

Интерактивная документация Swagger UI.

### `GET /redoc`

Альтернативная документация ReDoc.

## Структура проекта

```
backend/
├── main.py              # FastAPI приложение
├── prepare_data.py      # Скрипт генерации эмбеддингов
├── requirements.txt     # Python зависимости
├── README.md           # Документация
└── data/               # Сгенерированные данные
    ├── embeddings.npy  # Матрица эмбеддингов
    └── cases.json      # Данные решений
```

## Поля решений ФАС

| Поле | Описание |
|------|----------|
| `docId` | Уникальный идентификатор документа |
| `Violation_Type` | Тип нарушения: `substance` (содержание) или `placement` (размещение) |
| `document_date` | Дата решения |
| `FASbd_link` | Ссылка на решение на сайте ФАС |
| `FAS_division` | Территориальное подразделение ФАС |
| `violation_found` | Признано ли нарушение (да/нет) |
| `defendant_name` | Наименование ответчика |
| `defendant_industry` | Отрасль ответчика |
| `ad_description` | Описание рекламы |
| `ad_content_cited` | Цитата рекламного контента |
| `ad_platform` | Платформа размещения рекламы |
| `violation_summary` | Краткое описание нарушения |
| `FAS_arguments` | Аргументы и обоснование ФАС |
| `legal_provisions` | Нарушенные статьи закона |
| `thematic_tags` | Тематические теги |

## Примеры использования

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/search",
    json={"query": "реклама алкоголя", "top_k": 5}
)

for result in response.json()["results"]:
    print(f"{result['score']:.2f} - {result['defendant_name']}")
    print(f"  {result['violation_summary'][:100]}...")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "СМС без согласия", "top_k": 5}'
```

### JavaScript/Fetch

```javascript
const response = await fetch("http://localhost:8000/api/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "лучший банк года", top_k: 10 })
});

const data = await response.json();
console.log(data.results);
```

## Производительность

- Загрузка модели: ~5-10 секунд при первом запуске
- Поисковый запрос: ~50-100 мс
- Память: ~500 МБ (модель + эмбеддинги)

## Лицензия

MIT
