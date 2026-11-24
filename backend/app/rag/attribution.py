"""Quellen einer Empfehlung, deterministisch aus dem Agent-Verlauf statt aus den Freitext-`sources` des LLM.

- Wissensbasis: Titel der Dokumente aus `search_knowledge` bzw. `research_knowledge`.
- Web: URLs und Titel der Treffer von `search_bikes` und der Seiten von `read_product_page`.
"""

from typing import Any

KNOWLEDGE_TOOL = "search_knowledge"
RESEARCH_TOOL = "research_knowledge"
SEARCH_TOOL = "search_bikes"
PAGE_TOOL = "read_product_page"


def _tool_returns(history: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    return [
        (part.get("tool_name"), part.get("content"))
        for message in history
        for part in message.get("parts", [])
        if part.get("part_kind") == "tool-return"
    ]


def retrieved_documents(history: list[dict[str, Any]]) -> list[str]:
    """Titel der Dokumente aus den Ergebnissen von search_knowledge und research_knowledge, in Reihenfolge des
    ersten Auftretens."""
    titles: dict[str, None] = {}
    for tool_name, content in _tool_returns(history):
        if tool_name == KNOWLEDGE_TOOL:
            for hit in content if isinstance(content, list) else []:
                if isinstance(hit, dict) and hit.get("document"):
                    titles.setdefault(hit["document"], None)
        elif tool_name == RESEARCH_TOOL and isinstance(content, dict):
            for title in content.get("documents", []):
                titles.setdefault(title, None)
    return list(titles)


def web_sources(history: list[dict[str, Any]], used_urls: list[str] | None = None) -> list[dict[str, str]]:
    """Webseiten, die Websuche bzw. Seitenabruf geliefert haben. Mit `used_urls` nur diese (z. B. die Quell-URLs der
    Rangliste), in deren Reihenfolge."""
    titles: dict[str, str] = {}
    for tool_name, content in _tool_returns(history):
        if not isinstance(content, dict):
            continue
        if tool_name == SEARCH_TOOL:
            for hit in content.get("results", []):
                if isinstance(hit, dict) and hit.get("url"):
                    titles.setdefault(hit["url"], hit.get("title") or hit["url"])
        elif tool_name == PAGE_TOOL and content.get("url"):
            titles.setdefault(content["url"], content["url"])
    urls = used_urls if used_urls is not None else list(titles)
    return [{"url": url, "title": titles.get(url, url)} for url in dict.fromkeys(urls) if url in titles]
