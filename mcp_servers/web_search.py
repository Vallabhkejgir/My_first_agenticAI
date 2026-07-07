import json
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

# Instantiate a FastMCP server named "WebSearch"
web_search_mcp = FastMCP("WebSearch")

@web_search_mcp.tool()
def web_search(query: str, max_results: int = 3) -> str:
    """Searches the internet for information on a given query. Use this to find facts, latest news, or verify documentation."""
    try:
        results_list = []
        with DDGS() as ddgs:
            # text search returns a generator of dicts with keys: 'title', 'href', 'body'
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                results_list.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "summary": r.get("body", "")
                })
                
        if not results_list:
            return f"No results found on the web for: '{query}'."
            
        formatted_results = []
        for i, res in enumerate(results_list, 1):
            formatted_results.append(
                f"[{i}] Title: {res['title']}\n"
                f"    URL: {res['url']}\n"
                f"    Summary: {res['summary']}\n"
            )
            
        return "\n".join(formatted_results)
    except Exception as e:
        # Graceful error reporting
        return f"Error conducting web search: {str(e)}. Fallback: Web search results could not be retrieved dynamically."
