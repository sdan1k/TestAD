"use client";

import { useState } from "react";
import { FilterOptions } from "@/lib/api";

interface FilterPanelProps {
  options: FilterOptions;
  selected: {
    year: number[];
    region: string[];
    industry: string[];
    article: string[];
  };
  onChange: (filterType: string, values: any[]) => void;
}

export function FilterPanel({ options, selected, onChange }: FilterPanelProps) {
  const [openSection, setOpenSection] = useState<string | null>("year");

  const toggleSection = (section: string) => {
    setOpenSection(openSection === section ? null : section);
  };

  const handleCheckboxChange = (section: string, value: string | number, checked: boolean) => {
    const currentValues = selected[section as keyof typeof selected];
    let newValues: any[];
    
    if (checked) {
      newValues = [...currentValues, value];
    } else {
      newValues = currentValues.filter(v => v !== value);
    }
    
    onChange(section, newValues);
  };

  return (
    <div className="space-y-4">
      {/* Год решения */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("year")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Год решения ФАС</span>
          <div className="flex items-center gap-2">
            {selected.year.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.year.length}
              </span>
            )}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`transition-transform ${openSection === "year" ? "rotate-180" : ""}`}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        {openSection === "year" && (
          <div className="p-4 border-t max-h-[200px] overflow-y-auto">
            {options.years.length > 0 ? (
              <div className="space-y-2">
                {options.years.map((year) => (
                  <label key={year} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.year.includes(year)}
                      onChange={(e) => handleCheckboxChange("year", year, e.target.checked)}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="text-sm">{year}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>

      {/* Регион */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("region")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Регион (УФАС)</span>
          <div className="flex items-center gap-2">
            {selected.region.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.region.length}
              </span>
            )}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`transition-transform ${openSection === "region" ? "rotate-180" : ""}`}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        {openSection === "region" && (
          <div className="p-4 border-t max-h-[200px] overflow-y-auto">
            {options.regions.length > 0 ? (
              <div className="space-y-2">
                {options.regions.slice(0, 50).map((region) => (
                  <label key={region} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.region.includes(region)}
                      onChange={(e) => handleCheckboxChange("region", region, e.target.checked)}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="text-sm truncate">{region}</span>
                  </label>
                ))}
                {options.regions.length > 50 && (
                  <p className="text-xs text-muted-foreground">
                    Показано 50 из {options.regions.length}
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>

      {/* Отрасль */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("industry")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Отрасль</span>
          <div className="flex items-center gap-2">
            {selected.industry.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.industry.length}
              </span>
            )}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`transition-transform ${openSection === "industry" ? "rotate-180" : ""}`}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        {openSection === "industry" && (
          <div className="p-4 border-t max-h-[200px] overflow-y-auto">
            {options.industries.length > 0 ? (
              <div className="space-y-2">
                {options.industries.slice(0, 50).map((industry) => (
                  <label key={industry} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.industry.includes(industry)}
                      onChange={(e) => handleCheckboxChange("industry", industry, e.target.checked)}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="text-sm truncate">{industry}</span>
                  </label>
                ))}
                {options.industries.length > 50 && (
                  <p className="text-xs text-muted-foreground">
                    Показано 50 из {options.industries.length}
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>

      {/* Статья закона */}
      <div className="border rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection("article")}
          className="w-full flex items-center justify-between p-4 bg-muted/50 hover:bg-muted transition-colors"
        >
          <span className="font-medium">Статья закона</span>
          <div className="flex items-center gap-2">
            {selected.article.length > 0 && (
              <span className="bg-primary text-primary-foreground text-xs px-2 py-0.5 rounded-full">
                {selected.article.length}
              </span>
            )}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`transition-transform ${openSection === "article" ? "rotate-180" : ""}`}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </button>
        {openSection === "article" && (
          <div className="p-4 border-t max-h-[200px] overflow-y-auto">
            {options.articles.length > 0 ? (
              <div className="space-y-2">
                {options.articles.slice(0, 30).map((article) => (
                  <label key={article} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.article.includes(article)}
                      onChange={(e) => handleCheckboxChange("article", article, e.target.checked)}
                      className="w-4 h-4 rounded border-border text-primary focus:ring-primary"
                    />
                    <span className="text-sm">{article}</span>
                  </label>
                ))}
                {options.articles.length > 30 && (
                  <p className="text-xs text-muted-foreground">
                    Показано 30 из {options.articles.length}
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Нет данных</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
