"use client";

import { useState, useEffect } from "react";
import { SearchForm } from "@/components/search-form";
import { CaseCard } from "@/components/case-card";
import { searchCases, checkHealth, CaseResult } from "@/lib/api";

export default function Home() {
  const [results, setResults] = useState<CaseResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [totalCases, setTotalCases] = useState<number>(0);
  const [serverStatus, setServerStatus] = useState<"checking" | "online" | "offline">("checking");

  // Проверка состояния сервера при загрузке
  useEffect(() => {
    const checkServer = async () => {
      try {
        const health = await checkHealth();
        if (health.status === "ok" && health.data_loaded) {
          setServerStatus("online");
          setTotalCases(health.total_cases);
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

    try {
      const response = await searchCases(query, 10);
      setResults(response.results);
      setTotalCases(response.total_cases);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Произошла ошибка при поиске");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Поиск по решениям ФАС</h1>
              <p className="text-muted-foreground mt-1">
                Семантический поиск по базе нарушений в рекламе
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  serverStatus === "online"
                    ? "bg-green-500"
                    : serverStatus === "offline"
                    ? "bg-red-500"
                    : "bg-yellow-500 animate-pulse"
                }`}
              />
              <span className="text-sm text-muted-foreground">
                {serverStatus === "online"
                  ? `${totalCases.toLocaleString("ru-RU")} решений`
                  : serverStatus === "offline"
                  ? "Сервер недоступен"
                  : "Проверка..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto px-4 py-8">
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

        {/* Empty state */}
        {!isLoading && results.length === 0 && !error && (
          <div className="max-w-3xl mx-auto text-center py-12">
            <div className="text-6xl mb-4">🔍</div>
            <h2 className="text-xl font-medium mb-2">
              Введите запрос для поиска
            </h2>
            <p className="text-muted-foreground">
              Система найдет похожие решения ФАС по смыслу вашего запроса, 
              а не только по точному совпадению слов
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-4">
          <p className="text-center text-sm text-muted-foreground">
            Семантический поиск на основе эмбеддингов • Данные из открытых решений ФАС
          </p>
        </div>
      </footer>
    </div>
  );
}
