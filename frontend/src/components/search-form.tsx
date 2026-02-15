"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface SearchFormProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
}

export function SearchForm({ onSearch, isLoading }: SearchFormProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && query.trim().split(/\s+/).length >= 3) {
      onSearch(query.trim());
    }
  };

  // Примеры запросов для быстрого поиска
  const examples = [
    "Реклама алкоголя в интернете",
    "SMS-рассылка без согласия",
    "Некорректное сравнение товаров",
    "Недостоверная реклама лекарств",
    "Скидка 90% на всё",
  ];

  const exampleDescriptions = [
    "нарушение требований к рекламе алкогольной продукции",
    "распространение рекламы по сетям электросвязи без согласия",
    "использование сравнения без указания критериев",
    "реклама медицинских услуг с недостоверной информацией",
    "реклама с заведомо ложной информацией о скидках",
  ];

  return (
    <div className="w-full space-y-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <Input
          type="text"
          placeholder="Опишите ситуацию или вставьте текст рекламы. Например: «товар №1 в России» или «использование образа врача»"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 text-base h-12"
          disabled={isLoading}
        />
        <Button 
          type="submit" 
          disabled={isLoading || !query.trim() || query.trim().split(/\s+/).length < 3}
          className="h-12 px-6"
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Поиск
            </span>
          ) : (
            "Найти"
          )}
        </Button>
      </form>

      {/* Подсказка о минимальном количестве слов */}
      {query.trim() && query.trim().split(/\s+/).length < 3 && (
        <p className="text-sm text-amber-600">
          Минимум 3 слова для поиска. Добавьте больше деталей.
        </p>
      )}

      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-sm text-muted-foreground mr-2">Примеры:</span>
        {examples.map((example, idx) => (
          <button
            key={example}
            type="button"
            onClick={() => {
              setQuery(example);
              onSearch(example);
            }}
            disabled={isLoading}
            className="text-sm text-primary hover:underline disabled:opacity-50 text-left"
            title={exampleDescriptions[idx]}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
