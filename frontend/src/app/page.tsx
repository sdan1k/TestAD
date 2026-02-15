"use client";

import { useState, useEffect } from "react";
import { SearchForm } from "@/components/search-form";
import { CaseCard } from "@/components/case-card";
import { FilterPanel } from "@/components/filter-panel";
import { searchCases, checkHealth, getFilterOptions, CaseResult, FilterOptions } from "@/lib/api";

export default function Home() {
  const [results, setResults] = useState<CaseResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [totalCases, setTotalCases] = useState<number>(0);
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");
  const [showFilters, setShowFilters] = useState(false);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [selectedFilters, setSelectedFilters] = useState({
    year: [] as number[],
    region: [] as string[],
    industry: [] as string[],
    article: [] as string[],
  });
  const [message, setMessage] = useState<string | null>(null);

  // Проверка состояния сервера при загрузке
  useEffect(() => {
    const checkServer = async () => {
      try {
        const health = await checkHealth();
        if (health.status === "ok" && health.data_loaded) {
          setServerStatus("online");
          setTotalCases(health.total_cases);
          
          // Загрузить опции фильтров
          try {
            const options = await getFilterOptions();
            setFilterOptions(options);
          } catch (e) {
            console.error("Failed to load filter options:", e);
          }
        } else {
          setServerStatus("offline");
        }
      } catch {
        setServerStatus("offline");
      }
    };
    checkServer();
  }, []);

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setLastQuery(query);
    setMessage(null);

    try {
      const filters = {
        year: selectedFilters.year.length > 0 ? selectedFilters.year : undefined,
        region: selectedFilters.region.length > 0 ? selectedFilters.region : undefined,
        industry: selectedFilters.industry.length > 0 ? selectedFilters.industry : undefined,
        article: selectedFilters.article.length > 0 ? selectedFilters.article : undefined,
      };
      
      // Запрашиваем 20 результатов (по умолчанию с бэкенда)
      const response = await searchCases(query, 20, filters);
      setResults(response.results);
      setTotalCases(response.total_cases);
      
      if (response.message) {
        setMessage(response.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Произошла ошибка при поиске");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilterChange = (filterType: string, values: any[]) => {
    setSelectedFilters((prev) => ({
      ...prev,
      [filterType]: values,
    }));
  };

  const handleClearFilters = () => {
    setSelectedFilters({
      year: [],
      region: [],
      industry: [],
      article: [],
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="header">
        <div>
          <h1 className="text-2xl font-bold">Поиск практики ФАС по нарушениям в рекламе</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Умный поиск по {totalCases.toLocaleString("ru-RU")}+ решений органов ФАС
          </p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`btn-secondary flex items-center gap-2 ${showFilters ? 'bg-primary/10' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
            </svg>
            Фильтры
          </button>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                serverStatus === "online"
                  ? "bg-green-500"
                  : serverStatus === "offline"
                  ? "bg-red-500"
                  : "bg-yellow-500 animate-pulse"
              }`}
            />
            <span className="text-sm text-muted-foreground">
              {serverStatus === "online"
                ? `${totalCases.toLocaleString("ru-RU")} дел`
                : serverStatus === "offline"
                ? "Сервер недоступен"
                : "Проверка..."}
            </span>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar with Filters */}
        {showFilters && filterOptions && (
          <aside className="w-[320px] border-r bg-card p-6 h-[calc(100vh-64px)] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Фильтры</h2>
              <button
                onClick={handleClearFilters}
                className="text-sm text-muted-foreground hover:text-foreground"
              >
                Очистить все
              </button>
            </div>
            
            <FilterPanel
              options={filterOptions}
              selected={selectedFilters}
              onChange={handleFilterChange}
            />
          </aside>
        )}

        {/* Main content */}
        <main className={`flex-1 container mx-auto px-4 py-8 ${showFilters ? '' : ''}`}>
          {/* Search form */}
          <div className="max-w-3xl mx-auto mb-8">
            <SearchForm onSearch={handleSearch} isLoading={isLoading} />
          </div>

          {/* Error message */}
          {error && (
            <div className="max-w-3xl mx-auto mb-8">
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
                <p className="text-destructive text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Message */}
          {message && (
            <div className="max-w-3xl mx-auto mb-8">
              <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
                <p className="text-primary text-sm">{message}</p>
              </div>
            </div>
          )}

          {/* Results */}
          {results.length > 0 && (
            <div className="max-w-3xl mx-auto">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-medium">
                  Результаты поиска
                  {lastQuery && (
                    <span className="text-muted-foreground font-normal ml-2">
                      по запросу «{lastQuery}»
                    </span>
                  )}
                </h2>
                <span className="text-sm text-muted-foreground">
                  Найдено: {results.length} из {totalCases.toLocaleString("ru-RU")}
                </span>
              </div>

              <div className="space-y-4">
                {results.map((caseData, index) => (
                  <CaseCard key={caseData.docId || index} caseData={caseData} rank={index + 1} />
                ))}
              </div>
            </div>
          )}

          {/* Loading state */}
          {isLoading && (
            <div className="max-w-3xl mx-auto">
              <div className="empty-state">
                <div className="spinner" style={{ width: 48, height: 48 }} />
                <p className="text-lg font-medium mt-4">Поиск по базе решений...</p>
                <p className="text-muted-foreground mt-2">
                  Анализируем контекст и юридические формулировки
                </p>
              </div>
            </div>
          )}

          {/* Empty state - no query yet */}
          {!isLoading && results.length === 0 && !error && (
            <div className="max-w-3xl mx-auto text-center py-12">
              <div className="text-6xl mb-4">🔍</div>
              <h2 className="text-xl font-medium mb-2">
                Введите запрос для поиска
              </h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                Система найдет похожие решения ФАС по смыслу вашего запроса. 
                Опишите ситуацию или вставьте текст рекламы.
              </p>
            </div>
          )}

          {/* Empty results */}
          {!isLoading && results.length === 0 && error === null && lastQuery && !message && (
            <div className="max-w-3xl mx-auto">
              <div className="empty-state">
                <div className="text-6xl mb-4">📋</div>
                <h2 className="text-xl font-medium mb-2">
                  Мы не нашли точных совпадений
                </h2>
                <p className="text-muted-foreground max-w-md mx-auto">
                  Похоже, таких формулировок в нашей базе из {totalCases.toLocaleString("ru-RU")} дел еще не встречалось.
                  <br /><br />
                  Попробуйте:
                  <br />- Переформулировать запрос более общими словами
                  <br />- Сбросить фильтры
                  <br />- Описать суть, а не конкретный слоган
                </p>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-4">
          <p className="text-center text-sm text-muted-foreground">
            Семантический поиск на основе эмбеддингов Google Gemini • Данные из открытых решений ФАС
          </p>
        </div>
      </footer>
    </div>
  );
}
