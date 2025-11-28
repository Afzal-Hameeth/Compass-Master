import { useState } from "react";
import { API } from "../utils/constants";
import { useResearch } from "../hooks/useResearch";

import { Building2, Search, Sparkles } from "lucide-react";
import { FiChevronRight, FiChevronDown, FiLayers } from "react-icons/fi";

const exampleQueries = [
  "What capabilities support customer onboarding?",
  "Show me processes for KYC verification",
  "Which applications handle customer data?",
  "What's the path from onboarding to data storage?"
];

export default function ResearchAgent() {
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<{[id: number]: boolean}>({});
  const [processes, setProcesses] = useState<{[id: number]: any[]}>({});
  const [loadingProcesses, setLoadingProcesses] = useState<{[id: number]: boolean}>({});
  const { research, results, isLoading, error } = useResearch();

  const handleSearch = async (_searchQuery: string) => {
    await research(_searchQuery);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      handleSearch(query);
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
  };

  const handleToggleExpand = async (id: number) => {
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
    if (!expanded[id]) {
      // Only fetch if not already loaded
      if (!processes[id]) {
        setLoadingProcesses(prev => ({ ...prev, [id]: true }));
        try {
          const res = await fetch(
            API.ENDPOINTS.CAPABILITIES.BASE_URL().replace("/api/capabilities", "/api/processes") + `?capability_id=${id}`
          );
          if (res.ok) {
            const data = await res.json();
            setProcesses(prev => ({ ...prev, [id]: data }));
          } else {
            setProcesses(prev => ({ ...prev, [id]: [] }));
          }
        } catch {
          setProcesses(prev => ({ ...prev, [id]: [] }));
        }
        setLoadingProcesses(prev => ({ ...prev, [id]: false }));
      }
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img src="/favicon.png" width ="40" height="40" />
              <div>
                <h1 className="text-xl font-semibold">Capability Compass</h1>
                <p className="text-xs text-muted-foreground">
                  AI-Driven What-If Scenario Analysis
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-6 py-6">
        <div className="space-y-6">
          <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask a what-if question about your business architecture..."
                  className="w-full min-h-[120px] resize-none text-base p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-background"
                  data-testid="input-query"
                />
                <div className="absolute top-3 right-3">
                  <Sparkles className="h-5 w-5 text-muted-foreground" />
                </div>
              </div>
              
              <div className="flex items-center justify-between gap-4">
                <div className="flex flex-wrap gap-2">
                  {exampleQueries.map((example, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => handleExampleClick(example)}
                      className="px-3 py-1 text-xs bg-secondary text-secondary-foreground border rounded hover:shadow-md transition-shadow cursor-pointer"
                      data-testid={`badge-example-${idx}`}
                    >
                      {example}
                    </button>
                  ))}
                </div>
                
                <button
                  type="submit"
                  disabled={!query.trim() || isLoading}
                  className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 text-white whitespace-nowrap"
                  data-testid="button-analyze"
                >
                  <Search className="h-4 w-4" />
                  {isLoading ? "Analyzing..." : "Analyze"}
                </button>
              </div>
          </form>

          {error && (
            <div className="border border-destructive rounded-lg p-4 bg-destructive/10">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          {isLoading ? (
            <div className="text-center py-16">Analyzing...</div>
          ) : results.length > 0 ? (
            <div className="py-8">
              <div className="flex items-center gap-3 mb-6">
                <FiLayers className="w-8 h-8 text-indigo-600" />
                <h3 className="text-2xl font-semibold text-gray-900">Relevant Capabilities ({results.length})</h3>
              </div>
              <ul className="space-y-4">
                {results.map((cap: any) => {
                  const isExpanded = expanded[cap.id];
                  return (
                    <li key={cap.id} className="bg-white border border-gray-100 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
                      <div className="p-4 flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <button
                              className="text-gray-400 p-2 rounded-md hover:bg-gray-50 flex items-center justify-center"
                              onClick={() => handleToggleExpand(cap.id)}
                              aria-expanded={isExpanded}
                              title={isExpanded ? 'Collapse' : 'Expand'}
                            >
                              {isExpanded ? <FiChevronDown size={18} /> : <FiChevronRight size={18} />}
                            </button>
                            <div>
                              <div className="text-lg font-semibold text-gray-900">{cap.name}</div>
                              <div className="mt-1">
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700">{cap.domain ?? 'Unassigned'}</span>
                              </div>
                              <div className="mt-3 text-sm text-gray-600">{cap.description}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                      {isExpanded && (
                        <div className="mt-4 px-6 pb-4">
                          <h4 className="text-sm font-medium mb-2 text-indigo-700">Processes</h4>
                          {loadingProcesses[cap.id] ? (
                            <div className="text-xs text-muted-foreground">Loading...</div>
                          ) : processes[cap.id] && processes[cap.id].length > 0 ? (
                            <ul className="space-y-2">
                              {processes[cap.id].map((proc: any) => (
                                <li key={proc.id} className="border rounded p-3 bg-gray-50">
                                  <div className="font-semibold text-gray-800">{proc.name}</div>
                                  <div className="text-xs text-gray-600">{proc.description}</div>
                                  {proc.level && (
                                    <div className="text-xs text-indigo-600">Level: {proc.level}</div>
                                  )}
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <div className="text-xs text-muted-foreground">No processes found.</div>
                          )}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : query.trim() ? (
            <div className="text-center py-16" data-testid="no-results-state">
              <Building2 className="h-16 w-16 mx-auto text-muted-foreground/50 mb-4" />
              <h3 className="text-lg font-medium mb-2">No Matching Capabilities Found</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                We couldn't find any capabilities matching your query. Try rewording your question or exploring different topics.
              </p>
            </div>
          ) : (
            <div className="text-center py-16" data-testid="empty-state">
              <Building2 className="h-16 w-16 mx-auto text-muted-foreground/50 mb-4" />
              <h3 className="text-lg font-medium mb-2">Ready to Explore</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Enter a question above to discover architecture relationships and pathways
                across your enterprise.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
