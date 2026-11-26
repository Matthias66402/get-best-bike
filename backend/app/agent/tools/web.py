"""Live-Websuche nach Fahrradmodellen über Tavily (https://docs.tavily.com).

- `search_bikes`: Suche mit kurzen Auszügen je Treffer (Tavily /search).
- `read_product_page`: bereinigter Text einer Seite, z. B. Hersteller- oder Händlerseite (Tavily /extract).

Der Output-Validator des Beraters lässt nur Quell-URLs und Preise zu, die diese Tools im Verlauf geliefert haben
(app.agent.recommender.known_pages). Antworten bleiben `WEB_CACHE_HOURS` in der Tabelle web_cache, damit gleiche Anfragen kein
Kontingent kosten.
"""

import datetime as dt
import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic_ai import RunContext
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.agent.deps import AgentDeps
from app.agent.tools.common import http_error_message, tool_error
from app.agent.tools.prices import prices_in_text
from app.models import WebCacheEntry

logger = logging.getLogger(__name__)

SERVICE = "Websuche (Tavily)"
# Jede Seite bleibt im Verlauf und zählt bei jeder weiteren Anfrage gegen das Token-Limit pro Minute
SNIPPET_CHARS = 400
PAGE_CHARS = 5000
MAX_RESULTS_LIMIT = 10
# Shops und Ratgeberseiten liefern verwertbare Preise und Daten; Videos und soziale Netze selten
EXCLUDE_DOMAINS = ["youtube.com", "facebook.com", "instagram.com", "tiktok.com", "pinterest.com", "reddit.com"]


def normalize_url(url: str) -> str:
    """Für den Abgleich: ohne Fragment und abschließenden Schrägstrich, Schema und Host klein geschrieben."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def clean_page_text(raw: str) -> str:
    """Entfernt, was nur Zeichen kostet: Bilder, Link-Ziele, JSON-Reste aus Shop-Seiten (Canyon) und Leerzeilen."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)
    text = re.sub(r"\[([^\]]*)\]\((?:https?:)?[^)\s]*\)", r"\1", text)
    lines = [line.rstrip() for line in text.splitlines() if line.count('":') < 3]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


async def cached(deps: AgentDeps, key: str) -> dict[str, Any] | None:
    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=deps.settings.web_cache_hours)
    try:
        async with deps.session_factory() as session:
            entry = await session.scalar(
                select(WebCacheEntry).where(WebCacheEntry.key == key, WebCacheEntry.fetched_at >= since)
            )
            return entry.payload if entry else None
    except Exception as exc:  # Der Cache ist optional
        logger.warning("web_cache nicht lesbar: %s", exc)
        return None


async def store(deps: AgentDeps, key: str, payload: dict[str, Any]) -> None:
    try:
        async with deps.session_factory() as session:
            statement = insert(WebCacheEntry).values(key=key, payload=payload, fetched_at=dt.datetime.now(dt.UTC))
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=[WebCacheEntry.key],
                    set_={"payload": statement.excluded.payload, "fetched_at": statement.excluded.fetched_at},
                )
            )
            await session.commit()
    except Exception as exc:
        logger.warning("web_cache nicht beschreibbar: %s", exc)


async def tavily(deps: AgentDeps, endpoint: str, body: dict[str, Any], cache_key: str) -> dict[str, Any]:
    """Antwort von Tavily, aus dem Cache oder live; wirft httpx-Fehler."""
    if (payload := await cached(deps, cache_key)) is not None:
        return payload
    response = await deps.http.post(
        f"{deps.settings.tavily_api_url.rstrip('/')}/{endpoint}",
        json=body,
        headers={"Authorization": f"Bearer {deps.settings.tavily_api_key}"},
    )
    response.raise_for_status()
    payload = response.json()
    await store(deps, cache_key, payload)
    return payload


async def search_bikes(ctx: RunContext[AgentDeps], query: str, max_results: int | None = None) -> dict[str, Any]:
    """Sucht live im Web nach Fahrradmodellen, Preisen und Tests.

    Liefert je Treffer Titel, URL, einen Auszug und die darin erkannten Preise (`prices_eur`). Eine Suche je
    Fragestellung, auf Deutsch und mit Kategorie, Budget und Modelljahr, z. B.
    "Trekking E-Bike bis 3500 Euro Test 2026" oder "Cube Kathmandu Hybrid Pro 750 Preis".

    Args:
        query: Kurze Suchanfrage.
        max_results: Anzahl der Treffer (höchstens der Standard aus der Konfiguration).
    """
    deps = ctx.deps
    if not deps.settings.tavily_api_key:
        return tool_error("Websuche ist nicht eingerichtet (TAVILY_API_KEY fehlt). Ohne aktuelle Modelle beraten.")
    # Mehr als konfiguriert nicht, auch wenn das Modell es verlangt (es setzte gern 6)
    default = deps.settings.web_search_max_results
    limit = max(1, min(max_results or default, default, MAX_RESULTS_LIMIT))
    body = {
        "query": query, "max_results": limit, "search_depth": "basic", "topic": "general", "country": "germany",
        "exclude_domains": EXCLUDE_DOMAINS,
    }  # fmt: skip
    try:
        payload = await tavily(deps, "search", body, f"search:{query.strip().casefold()}|{limit}")
    except (httpx.HTTPError, ValueError) as exc:
        return tool_error(http_error_message(SERVICE, exc))
    results = []
    for hit in payload.get("results") or []:
        url, text = hit.get("url"), (hit.get("content") or "")[:SNIPPET_CHARS]
        if not url:
            continue
        results.append({"title": hit.get("title"), "url": url, "snippet": text, "prices_eur": prices_in_text(text)})
    if not results:
        return tool_error(f"Keine Treffer für „{query}“. Anders formulieren, z. B. allgemeiner oder ohne Modellnamen.")
    return {"query": query, "results": results}


async def read_product_page(ctx: RunContext[AgentDeps], url: str) -> dict[str, Any]:
    """Lädt den bereinigten Text einer Webseite, z. B. Produktseite eines Herstellers oder Händlers oder einen Test.

    Nutze es für die Kandidaten der Rangliste, um Preis, Ausstattung, Gewicht und Größentabelle zu belegen.
    Rufe es für mehrere Kandidaten in derselben Runde auf.

    Args:
        url: Vollständige URL aus den Ergebnissen von `search_bikes` (selbst gebaute URLs werden abgelehnt).
    """
    deps = ctx.deps
    if not deps.settings.tavily_api_key:
        return tool_error("Websuche ist nicht eingerichtet (TAVILY_API_KEY fehlt).")
    if urlsplit(url).scheme not in ("http", "https"):
        return tool_error(f"Ungültige URL: {url}")
    try:
        payload = await tavily(deps, "extract", {"urls": [url], "extract_depth": "basic"}, f"extract:{url.strip()}")
    except (httpx.HTTPError, ValueError) as exc:
        return tool_error(http_error_message(SERVICE, exc))
    pages = payload.get("results") or []
    if not pages or not pages[0].get("raw_content"):
        failed = (payload.get("failed_results") or [{}])[0].get("error", "kein Inhalt")
        return tool_error(f"Seite nicht lesbar ({failed}). Einen anderen Treffer derselben Suche nehmen.")
    raw = clean_page_text(pages[0]["raw_content"])
    text = raw[:PAGE_CHARS]
    return {"url": url, "text": text, "prices_eur": prices_in_text(text), "truncated": len(raw) > PAGE_CHARS}
