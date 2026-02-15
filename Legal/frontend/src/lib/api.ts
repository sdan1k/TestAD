/**
 * API клиент для взаимодействия с FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface CaseResult {
  index: number;
  score: number;
  docId: string | null;
  Violation_Type: string | null;
  document_date: string | null;
  FASbd_link: string | null;
  FAS_division: string | null;
  violation_found: string | null;
  defendant_name: string | null;
  defendant_industry: string | null;
  ad_description: string | null;
  ad_content_cited: string | null;
  ad_platform: string | null;
  violation_summary: string | null;
  FAS_arguments: string | null;
  legal_provisions: string | null;
  thematic_tags: string | null;
}

export interface SearchResponse {
  query: string;
  total_cases: number;
  results: CaseResult[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  data_loaded: boolean;
  total_cases: number;
}

/**
 * Семантический поиск по решениям ФАС
 */
export async function searchCases(
  query: string,
  topK: number = 10
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Ошибка сервера: ${response.status}`);
  }

  return response.json();
}

/**
 * Проверка состояния сервера
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error(`Сервер недоступен: ${response.status}`);
  }

  return response.json();
}
