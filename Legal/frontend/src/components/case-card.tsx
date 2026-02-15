"use client";

import { CaseResult } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface CaseCardProps {
  caseData: CaseResult;
  rank: number;
}

export function CaseCard({ caseData, rank }: CaseCardProps) {
  // Форматирование даты
  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "—";
    try {
      return new Date(dateStr).toLocaleDateString("ru-RU", {
        year: "numeric",
        month: "long",
        day: "numeric",
      });
    } catch {
      return dateStr;
    }
  };

  // Парсинг тегов
  const parseTags = (tagsStr: string | null): string[] => {
    if (!tagsStr) return [];
    return tagsStr.split(",").map((tag) => tag.trim()).filter(Boolean);
  };

  // Парсинг статей
  const parseArticles = (articlesStr: string | null): string[] => {
    if (!articlesStr) return [];
    try {
      // Формат: ['п. 1 ч. 2 ст. 5', 'п. 1 ч. 3 ст. 5']
      const parsed = JSON.parse(articlesStr.replace(/'/g, '"'));
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return articlesStr.split(",").map((a) => a.trim()).filter(Boolean);
    }
  };

  const tags = parseTags(caseData.thematic_tags);
  const articles = parseArticles(caseData.legal_provisions);
  const scorePercent = Math.round(caseData.score * 100);

  return (
    <Card className="w-full hover:shadow-lg transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="outline" className="text-xs">
                #{rank}
              </Badge>
              <Badge
                variant={scorePercent >= 70 ? "default" : "secondary"}
                className="text-xs"
              >
                {scorePercent}% совпадение
              </Badge>
              {caseData.Violation_Type && (
                <Badge variant="outline" className="text-xs">
                  {caseData.Violation_Type === "substance"
                    ? "Содержание"
                    : caseData.Violation_Type === "placement"
                    ? "Размещение"
                    : caseData.Violation_Type}
                </Badge>
              )}
            </div>
            <CardTitle className="text-lg leading-tight">
              {caseData.defendant_name || "Неизвестный ответчик"}
            </CardTitle>
            <CardDescription className="mt-1">
              {caseData.defendant_industry && (
                <span className="mr-3">{caseData.defendant_industry}</span>
              )}
              <span>{formatDate(caseData.document_date)}</span>
              {caseData.FAS_division && (
                <span className="ml-3 text-muted-foreground">
                  {caseData.FAS_division}
                </span>
              )}
            </CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Цитата рекламы */}
        {caseData.ad_content_cited && (
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              Цитата рекламы
            </h4>
            <blockquote className="border-l-2 border-primary pl-3 italic text-sm">
              {caseData.ad_content_cited.length > 400
                ? caseData.ad_content_cited.slice(0, 400) + "..."
                : caseData.ad_content_cited}
            </blockquote>
          </div>
        )}

        {/* Суть нарушения */}
        {caseData.violation_summary && (
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              Суть нарушения
            </h4>
            <p className="text-sm">
              {caseData.violation_summary.length > 500
                ? caseData.violation_summary.slice(0, 500) + "..."
                : caseData.violation_summary}
            </p>
          </div>
        )}

        {/* Платформа размещения */}
        {caseData.ad_platform && (
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              Платформа
            </h4>
            <p className="text-sm">{caseData.ad_platform}</p>
          </div>
        )}

        {/* Нарушенные статьи */}
        {articles.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              Нарушенные статьи
            </h4>
            <div className="flex flex-wrap gap-1">
              {articles.map((article, idx) => (
                <Badge key={idx} variant="secondary" className="text-xs">
                  {article}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Теги */}
        {tags.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-muted-foreground mb-1">
              Теги
            </h4>
            <div className="flex flex-wrap gap-1">
              {tags.slice(0, 5).map((tag, idx) => (
                <Badge key={idx} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
              {tags.length > 5 && (
                <Badge variant="outline" className="text-xs">
                  +{tags.length - 5}
                </Badge>
              )}
            </div>
          </div>
        )}

        {/* Ссылка на ФАС */}
        {caseData.FASbd_link && (
          <div className="pt-2 border-t">
            <a
              href={caseData.FASbd_link}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary hover:underline inline-flex items-center gap-1"
            >
              Открыть решение на сайте ФАС →
            </a>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
