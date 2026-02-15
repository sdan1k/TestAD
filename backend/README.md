# FAS Hybrid Search - Инструкция по запуску

## Требования

- Python 3.11+
- macOS / Linux / Windows
- 8GB RAM минимум
- 5GB свободного места на диске

## Структура проекта

```
Legal-main/
├── backend/
│   ├── main.py              # FastAPI сервер
│   ├── prepare_data.py      # Подготовка данных
│   ├── embeddings.py        # Создание эмбеддингов
│   ├── requirements.txt     # Зависимости
│   └── data/               # Данные (эмбеддинги, кейсы)
│       ├── embeddings.npy
│       ├── embeddings_FAS_arguments.npy
│       ├── embeddings_violation_summary.npy
│       ├── embeddings_ad_description.npy
│       └── cases.json
└── frontend/               # Next.js приложение
```

---

## Шаг 1: Установка зависимостей

### Создание виртуального окружения

```bash
cd /Users/sdanya321/Downloads/Legal-main

# Создаем виртуальное окружение
python3 -m venv .venv

# Активируем
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate    # Windows
```

### Установка пакетов

```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

**Требования в requirements.txt:**
- fastapi>=0.100.0
- uvicorn[standard]>=0.22.0
- pydantic>=2.0.0
- sentence-transformers>=2.2.0
- scikit-learn>=1.3.0
- numpy>=1.24.0

---

## Шаг 2: Подготовка данных

Данные уже подготовлены в `backend/data/`:
- `cases.json` - 7283 кейса
- `embeddings.npy` - основные эмбеддинги (7283, 384)
- `embeddings_FAS_arguments.npy` - эмбеддинги аргументов ФАС
- `embeddings_violation_summary.npy` - эмбеддинги описания нарушений
- `embeddings_ad_description.npy` - эмбеддинги описания рекламы

**Если нужно пересоздать данные:**

1. Скачать CSV:
```bash
# CSV уже есть: /Users/sdanya321/Downloads/Legal-main/fas_ad_practice_dataset.csv
```

2. Создать эмбеддинги:
```bash
cd backend
python embeddings.py
```

3. Подготовить JSON:
```bash
python prepare_data.py
```

---

## Шаг 3: Запуск Backend

```bash
cd backend

# Активировать виртуальное окружение (если не активировано)
source ../.venv/bin/activate

# Запуск сервера
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Ожидаемый вывод при запуске:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
==================================================
ЗАГРУЗКА ДАННЫХ
==================================================
Загрузка модели эмбеддингов...
Loading weights: 100%|████████████████████| 199/199 [00:00<00:01]
...
Модель загружена: paraphrase-multilingual-MiniLM-L12-v2
  Основные эмбеддинги: (7283, 384)
  FAS_arguments эмбеддинги: (7283, 384)
  violation_summary эмбеддинги: (7283, 384)
  ad_description эмбеддинги: (7283, 384)
  Кейсов загружено: 7283
==================================================
ЗАГРУЗКА ЗАВЕРШЕНА
==================================================
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Шаг 4: Проверка работы API

### Проверка здоровья сервера
```bash
curl http://127.0.0.1:8000/api/health
```

**Ответ:**
```json
{
  "status":"ok",
  "model_loaded":true,
  "data_loaded":true,
  "total_cases":7283,
  "embedding_dimension":384
}
```

### Получить доступные фильтры
```bash
curl http://127.0.0.1:8000/api/filters
```

### Пример поиска
```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "реклама алкоголя", "top_k": 5}'
```

### Поиск с фильтрами
```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "реклама кредита", "top_k": 5, "year": [2023]}'
```

---

## Шаг 5: Запуск Frontend (опционально)

```bash
cd frontend
npm install
npm run dev
```

Открыть http://localhost:3000

---

## Устранение проблем

### Ошибка: "port already in use"
```bash
# Найти процесс
lsof -i :8000

# Убить процесс
kill -9 <PID>
```

### Ошибка: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Ошибка: "PydanticUserError"
```bash
# Убедитесь что используете правильную версию pydantic
pip install pydantic==2.10.5
```

### Медленная загрузка модели
Модель загружается из интернета (~500MB). При первом запуске может занять 2-5 минут.

---

## Архитектура гибридного поиска

```
Запрос пользователя
        │
        ▼
┌───────────────────┐
│ Эмбеддинг запроса │  (paraphrase-multilingual-MiniLM-L12-v2)
└───────────────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│ Семантический     │     │ Поиск по          │
│ поиск (TOP-100)   │     │ ключевым словам    │
│ (косинусное       │     │ (TOP-50)          │
│ сходство)         │     │                   │
└───────────────────┘     └───────────────────┘
        │                         │
        └───────────┬─────────────┘
                    ▼
┌───────────────────────────────────┐
│ Объединение результатов           │
│ (70% семантика + 30% ключевые)    │
└───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ Переранжирование с использованием  │
│ полей (FAS_arguments, violation,  │
│ ad_description) с весами          │
└───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ Применение фильтров (год, регион, │
│ отрасль, статья)                  │
└───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│ TOP-K результатов                 │
└───────────────────────────────────┘
```

---

## Контакты

При проблемах с запуском проверить:
1. Виртуальное окружение активировано
2. Все пакеты установлены
3. Порт 8000 свободен
4. Файлы данных на месте в `backend/data/`
