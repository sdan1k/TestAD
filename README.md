# FAS Semantic Search System

Система семантического поиска по решениям ФАС (Федеральной антимонопольной службы) о нарушениях в рекламе.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![Next.js](https://img.shields.io/badge/Next.js-16+-black.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)

## О проекте

Это приложение позволяет искать похожие решения ФАС по смыслу, а не по точному совпадению слов. Вы можете ввести текст рекламы или описание нарушения, и система найдёт наиболее релевантные прецеденты из базы.

### Пример использования

**Запрос:** "скидка 90% на всё"

**Результат:** Система найдёт решения о недостоверной рекламе со скидками, вводящей в заблуждение, некорректных сравнениях и т.д.

## Как это работает

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│    Frontend     │────▶│    Backend      │────▶│   Embeddings    │
│   (Next.js)     │     │   (FastAPI)     │     │    (NumPy)      │
│                 │◀────│                 │◀────│                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                       │
        │                       ▼
        │               ┌─────────────────┐
        │               │  Sentence       │
        └───────────────│  Transformers   │
                        │  (ML Model)     │
                        └─────────────────┘
```

1. **Подготовка данных**: CSV с 7283 решениями ФАС преобразуется в векторные представления (эмбеддинги)
2. **Поиск**: Запрос пользователя также преобразуется в вектор
3. **Ранжирование**: Система находит наиболее близкие векторы с помощью косинусного сходства
4. **Отображение**: Результаты показываются в удобном интерфейсе

## Быстрый старт

### Требования

- Python 3.10+
- Node.js 18+
- ~1 ГБ свободного места

### Установка и запуск

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/Latarho/Legal.git
cd Legal

# 2. Настройте бэкенд
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Подготовьте данные (один раз, ~2 минуты)
python prepare_data.py

# 4. Запустите бэкенд
uvicorn main:app --reload --port 8000

# 5. В новом терминале — настройте фронтенд
cd ../frontend
npm install
npm run dev
```

Откройте http://localhost:3000 в браузере.

## Структура проекта

```
Legal/
├── backend/                    # FastAPI сервер
│   ├── main.py                # API endpoints
│   ├── prepare_data.py        # Скрипт генерации эмбеддингов
│   ├── requirements.txt       # Python зависимости
│   ├── README.md             # Документация бэкенда
│   └── data/                  # Сгенерированные данные
│       ├── embeddings.npy     # Матрица эмбеддингов (7283 × 384)
│       └── cases.json         # Данные решений
│
├── frontend/                   # Next.js приложение
│   ├── src/
│   │   ├── app/              # Страницы и layouts
│   │   ├── components/       # React компоненты
│   │   └── lib/              # Утилиты и API клиент
│   ├── package.json
│   └── README.md             # Документация фронтенда
│
├── fas_ad_practice_dataset.csv # Исходные данные ФАС
└── README.md                   # Этот файл
```

## Технологии

### Backend

| Технология | Назначение |
|------------|------------|
| FastAPI | Веб-фреймворк |
| Sentence Transformers | Генерация эмбеддингов |
| paraphrase-multilingual-MiniLM-L12-v2 | ML модель для русского языка |
| NumPy | Векторные операции |
| Pandas | Обработка CSV |

### Frontend

| Технология | Назначение |
|------------|------------|
| Next.js 16 | React фреймворк |
| TypeScript | Типизация |
| Tailwind CSS 4 | Стилизация |
| shadcn/ui | UI компоненты |

## API

### Поиск

```bash
POST /api/search
Content-Type: application/json

{
  "query": "реклама алкоголя",
  "top_k": 10
}
```

### Ответ

```json
{
  "query": "реклама алкоголя",
  "total_cases": 7283,
  "results": [
    {
      "score": 0.85,
      "defendant_name": "ООО Компания",
      "violation_summary": "Нарушение требований к рекламе алкоголя...",
      "FASbd_link": "https://br.fas.gov.ru/cases/..."
    }
  ]
}
```

Полная документация API: http://localhost:8000/docs

## Данные

База содержит 7283 решения ФАС с полями:

| Поле | Описание |
|------|----------|
| `Violation_Type` | Тип: `substance` (содержание) / `placement` (размещение) |
| `defendant_name` | Ответчик |
| `defendant_industry` | Отрасль |
| `ad_content_cited` | Цитата рекламы |
| `violation_summary` | Суть нарушения |
| `legal_provisions` | Нарушенные статьи |
| `FASbd_link` | Ссылка на решение ФАС |

## Примеры запросов

- "Скидка 90% на всё" — недостоверная реклама скидок
- "Лучший банк года" — некорректные сравнения
- "Реклама алкоголя" — нарушения требований к рекламе алкоголя
- "СМС без согласия" — нарушения при рассылке рекламы
- "Недостоверная реклама лекарств" — нарушения в рекламе медицинских услуг

## Производительность

- Генерация эмбеддингов: ~2 минуты (один раз)
- Загрузка сервера: ~10 секунд
- Время поиска: ~50-100 мс
- Потребление памяти: ~500 МБ

## Разработка

### Запуск тестов

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### Линтинг

```bash
# Frontend
cd frontend
npm run lint
```

## Лицензия

MIT

## Авторы

- [Latarho](https://github.com/Latarho)

## Благодарности

- [ФАС России](https://fas.gov.ru/) за открытые данные
- [Sentence Transformers](https://www.sbert.net/) за отличную библиотеку
- [shadcn/ui](https://ui.shadcn.com/) за красивые компоненты
