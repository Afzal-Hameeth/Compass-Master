import { useState } from "react";

import { Building2, Search, Sparkles } from "lucide-react";

const exampleQueries = [
  "What capabilities support customer onboarding?",
  "Show me processes for KYC verification",
  "Which applications handle customer data?",
  "What's the path from onboarding to data storage?"
];

export default function ResearchAgent() {
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [query, setQuery] = useState("");

  const handleSearch = async (_searchQuery: string) => {
    setErrorMessage("");
    setIsLoading(true);
    
    // TODO: Implement actual search logic here
    setIsLoading(false);
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

  return (
    <div className="min-h-screen bg-background flex">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
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

      <main className="container mx-auto px-6 py-6">
        <div className="space-y-6">
          <div className="bg-card border rounded-lg p-6">
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
                  className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 whitespace-nowrap"
                  data-testid="button-analyze"
                >
                  <Search className="h-4 w-4" />
                  {isLoading ? "Analyzing..." : "Analyze"}
                </button>
              </div>
            </form>
          </div>

          {errorMessage && (
            <div className="border border-destructive rounded-lg p-4 bg-destructive/10">
              <p className="text-sm text-destructive">{errorMessage}</p>
            </div>
          )}

          {!isLoading && (
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
    </div>
  );
}
