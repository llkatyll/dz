
"""
Web search tool for LLM function calling.
Uses DuckDuckGo search API (no key required).
"""

import requests
from typing import List, Dict, Any

# Tool schema for OpenAI-compatible API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Поиск информации в интернете. Используй для получения актуальных данных: новости, цены, погода, спортивные результаты, последние версии ПО и т.д.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на русском или английском языке"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Получение содержимого веб-страницы по URL. Используй когда нужно прочитать полную статью или документ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL веб-страницы"
                    }
                },
                "required": ["url"]
            }
        }
    }
]


def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo Instant Answer API.

    Args:
        query: Search query string
        num_results: Number of results to return (max 10)

    Returns:
        Dictionary with 'results' list containing title, url, content
    """
    try:
        # Use DuckDuckGo HTML search (more reliable than API)
        from urllib.parse import quote

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # Try DuckDuckGo lite version
        url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {"error": f"Search failed with status {response.status_code}", "results": []}

        results = []
        html_content = response.text

        # Simple parsing of DuckDuckGo lite results
        import re
        from html.parser import HTMLParser

        class DuckDuckGoParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current_result = None
                self.in_title = False
                self.in_snippet = False
                self.title_text = ""
                self.snippet_text = ""
                self.url_text = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)

                # Look for result links
                if tag == "a" and "href" in attrs_dict:
                    href = attrs_dict.get("href", "")
                    if href.startswith("http") and "duckduckgo" not in href:
                        if self.current_result is None:
                            self.current_result = {}
                            self.url_text = href

                # Look for title (usually in <a> tags with specific classes)
                if tag == "a" and self.current_result is not None and not self.in_title:
                    self.in_title = True
                    self.title_text = ""

                # Look for snippet text
                if tag == "td" and attrs_dict.get("class") == "result-snippet":
                    self.in_snippet = True
                    self.snippet_text = ""

            def handle_endtag(self, tag):
                if tag == "a" and self.in_title:
                    self.in_title = False
                    if self.current_result is not None:
                        self.current_result["title"] = self.title_text.strip()

                if tag == "td" and self.in_snippet:
                    self.in_snippet = False
                    if self.current_result is not None:
                        self.current_result["content"] = self.snippet_text.strip()
                        if "url" in self.current_result and "title" in self.current_result:
                            self.results.append(self.current_result.copy())
                        self.current_result = None

            def handle_data(self, data):
                if self.in_title:
                    self.title_text += data
                if self.in_snippet:
                    self.snippet_text += data
                if self.current_result is not None and not self.in_title and not self.in_snippet:
                    if self.url_text:
                        self.current_result["url"] = self.url_text
                        self.url_text = ""

        parser = DuckDuckGoParser()
        parser.feed(html_content)
        results = parser.results[:num_results]

        # If no results from parser, try alternative approach
        if not results:
            # Try to extract any links and text from the page
            link_pattern = r'<a[^>]+href=["\'](https?://[^"\']+)["\'][^>]*>([^<]+)</a>'
            matches = re.findall(link_pattern, html_content)

            for url, title in matches[:num_results]:
                if "duckduckgo" not in url.lower():
                    results.append({
                        "title": title.strip(),
                        "url": url,
                        "content": "Релевантная страница по запросу"
                    })

        return {"results": results} if results else {"error": "No results found", "results": []}

    except Exception as e:
        return {"error": str(e), "results": []}


def fetch_url(url: str) -> Dict[str, Any]:
    """
    Fetch content from a URL.

    Args:
        url: The URL to fetch

    Returns:
        Dictionary with 'content' or 'error'
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {"error": f"Failed to fetch URL: status {response.status_code}"}

        # Extract text content (simple approach)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)

        # Limit content length
        max_length = 3000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return {"content": text, "url": url}

    except Exception as e:
        return {"error": str(e)}
