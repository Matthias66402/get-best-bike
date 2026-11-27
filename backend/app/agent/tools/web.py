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
from app.agent.tools.common import http_error_message, model_family, tool_error
from app.agent.tools.prices import prices_in_text
from app.models import WebCacheEntry

logger = logging.getLogger(__name__)

SERVICE = "Websuche (Tavily)"
# Jede Seite bleibt im Verlauf und zählt bei jeder weiteren Anfrage gegen das Token-Limit pro Minute
SNIPPET_CHARS = 400
PAGE_CHARS = 5000
# Gezielter Abruf (`model`): Zeichen vor und nach jeder Nennung der Modellfamilie
FOCUS_CHARS = 700
TRUNCATED_NOTE = (
    "Seite gekürzt. Steht ein Modell nicht im Text, rufe die Seite erneut mit `model` auf, dann kommen die Abschnitte "
    "zu diesem Modell (ohne Kosten, Cache)."
)
MAX_RESULTS_LIMIT = 10
# Übersichtsseiten (Kategorie, Filter, Preisvergleich): nennen viele Räder mit Namen und Preis, aber keine
# Ausstattung. Beratung 14: Alle fünf Räder waren nur mit fahrrad-xxl.de/.../radgroesse.28-zoll belegt, ohne jede
# Angabe zu Schaltung, Bremsen oder Gepäckträger
LISTING_URL = re.compile(
    r"/radgroesse\.|/marke\.|/productcategory/|[?&](pto|pfrom|price_(from|to)|range_value\w*|sort|order|filter\w*)=",
    re.IGNORECASE,
)
LISTING_TEXT = re.compile(r"sortieren nach|\bfiltern\b|filter zurücksetzen", re.IGNORECASE)
LISTING_PRICES = 6
LISTING_NOTE = (
    "Übersichtsseite mit vielen Rädern: Sie belegt weder Preis noch Ausstattung für Platz 1 bis 3. Nimm von hier "
    'die Modellnamen und suche jedes einzeln mit `search_bikes` und `shops=true` (z. B. "Cube Kathmandu Pro"), um '
    "seine Produktseite zu finden."
)
# Shops und Ratgeberseiten liefern verwertbare Preise und Daten; Videos und soziale Netze selten
# tavily.com: Firmenprofile statt Fahrräder (Beratung 9: 7 von 25 Treffern)
EXCLUDE_DOMAINS = [
    "youtube.com", "facebook.com", "instagram.com", "tiktok.com", "pinterest.com", "reddit.com", "tavily.com",
]  # fmt: skip
# Händlersuche (`shops=True`): deutsche Shops, Preisvergleiche und Hersteller mit Euro-Preisen. Beratung 9: Von 25
# Treffern der allgemeinen Suche hatten 4 einen Euro-Preis, der Rest waren Tests, US/UK-Shops und Übersichtsseiten
SHOP_DOMAINS = [
    "rosebikes.de", "bike24.de", "fahrrad-xxl.de", "bike-discount.de", "bike-components.de", "fahrrad.de",
    "lucky-bike.de", "zweirad-stadler.de", "boc24.de", "bikeshop.de", "idealo.de", "decathlon.de",
    "canyon.com", "cube.eu", "radon-bikes.de", "stevensbikes.de", "bulls.de", "ghost-bikes.com", "focus-bikes.com",
]  # fmt: skip


def normalize_url(url: str) -> str:
    """Für den Abgleich: ohne Fragment und abschließenden Schrägstrich, Schema und Host klein geschrieben."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def is_listing(url: str, text: str = "") -> bool:
    """Übersichts- statt Produktseite: an der URL (Kategorie-, Filter-, Preisvergleichspfade) oder am Text
    (Sortier-/Filterleiste und mindestens LISTING_PRICES verschiedene Preise)."""
    if LISTING_URL.search(url):
        return True
    return bool(LISTING_TEXT.search(text)) and len(set(prices_in_text(text))) >= LISTING_PRICES


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


async def search_bikes(
    ctx: RunContext[AgentDeps], query: str, max_results: int | None = None, shops: bool = False
) -> dict[str, Any]:
    """Sucht live im Web nach Fahrradmodellen, Preisen und Tests.

    Liefert je Treffer Titel, URL, einen Auszug und die darin erkannten Preise (`prices_eur`). Eine Suche je
    Fragestellung, auf Deutsch und mit Kategorie, Budget und Modelljahr, z. B.
    "Trekking E-Bike bis 3500 Euro Test 2026". Für Preise und Angebote `shops=true`, z. B.
    "Cube Kathmandu Hybrid Pro 750" oder "Gravelbike Carbon 2x bis 2500 Euro": Dann kommen nur Treffer deutscher
    Händler, Preisvergleiche und Hersteller mit Euro-Preisen.

    Args:
        query: Kurze Suchanfrage.
        max_results: Anzahl der Treffer (höchstens der Standard aus der Konfiguration).
        shops: Nur bei deutschen Händlern, Preisvergleichen und Herstellern suchen (für Preise und Produktseiten).
    """
    deps = ctx.deps
    if not deps.settings.tavily_api_key:
        return tool_error("Websuche ist nicht eingerichtet (TAVILY_API_KEY fehlt). Ohne aktuelle Modelle beraten.")
    # Beratung 10: „Canyon Grizl CF 6 Preis 2026 shops=true“, der Parameter zusätzlich im Suchtext
    query = re.sub(r"\s*\bshops\s*[=:]\s*(true|false)\b", "", query, flags=re.IGNORECASE).strip()
    # Mehr als konfiguriert nicht, auch wenn das Modell es verlangt (es setzte gern 6)
    default = deps.settings.web_search_max_results
    limit = max(1, min(max_results or default, default, MAX_RESULTS_LIMIT))
    body = {
        "query": query, "max_results": limit, "search_depth": "basic", "topic": "general", "country": "germany",
        "exclude_domains": EXCLUDE_DOMAINS,
    }  # fmt: skip
    if shops:
        # Mit `country` liefert Tavily bei include_domains keine Treffer (live geprüft: 0 statt 5 für „Cube Nuroad
        # C:62“); die Domains sind ohnehin deutsch
        body["include_domains"] = SHOP_DOMAINS
        del body["country"]
    key = f"search:{query.strip().casefold()}|{limit}" + ("|shops" if shops else "")
    try:
        payload = await tavily(deps, "search", body, key)
    except (httpx.HTTPError, ValueError) as exc:
        return tool_error(http_error_message(SERVICE, exc))
    results = []
    for hit in payload.get("results") or []:
        url, text = hit.get("url"), (hit.get("content") or "")[:SNIPPET_CHARS]
        if not url:
            continue
        result = {"title": hit.get("title"), "url": url, "snippet": text, "prices_eur": prices_in_text(text)}
        results.append(result | {"listing": True} if is_listing(url) else result)
    if not results:
        return tool_error(f"Keine Treffer für „{query}“. Anders formulieren, z. B. allgemeiner oder ohne Modellnamen.")
    return {"query": query, "results": results}


def focus_text(raw: str, model: str) -> str | None:
    """Abschnitte rund um die Nennungen der Modellfamilie, höchstens PAGE_CHARS Zeichen; None ohne Nennung.

    Beratung 8: Testberichte über viele Räder lieferten nur die ersten PAGE_CHARS Zeichen; das Modell las sie
    mehrfach neu, um an die übrigen Räder zu kommen, und verbrauchte damit das Budget."""
    if not (family := model_family(model)):
        return None
    spans: list[list[int]] = []
    for match in re.finditer(rf"(?<![0-9a-zäöüß]){re.escape(family)}(?![0-9a-zäöüß])", raw, re.IGNORECASE):
        start, end = max(0, match.start() - FOCUS_CHARS), match.end() + FOCUS_CHARS
        if spans and start <= spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    if not spans:
        return None
    return "\n…\n".join(raw[start:end] for start, end in spans)[:PAGE_CHARS]


async def read_product_page(ctx: RunContext[AgentDeps], url: str, model: str | None = None) -> dict[str, Any]:
    """Lädt den bereinigten Text einer Webseite, z. B. Produktseite eines Herstellers oder Händlers oder einen Test.

    Nutze es für die Kandidaten der Rangliste, um Preis, Ausstattung, Gewicht und Größentabelle zu belegen.
    Rufe es für mehrere Kandidaten in derselben Runde auf.

    Lange Seiten (z. B. Testberichte über viele Räder) werden gekürzt (`truncated`). Mit `model` liefert der Abruf
    stattdessen die Abschnitte zu diesem Modell, auch weiter hinten auf der Seite.

    Args:
        url: Vollständige URL aus den Ergebnissen von `search_bikes` (selbst gebaute URLs werden abgelehnt).
        model: Optional: Modellname, z. B. "Revolt Advanced 2", um auf einer langen Seite gezielt dessen Abschnitte
            mit Preis und Ausstattung zu lesen.
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
    listing = {"listing": True, "note": LISTING_NOTE} if is_listing(url, raw) else {}
    if model and len(raw) > PAGE_CHARS:
        if (focused := focus_text(raw, model)) is None:
            return tool_error(
                f"„{model}“ kommt auf der Seite {url} nicht vor. Belege das Modell mit einem anderen Treffer."
            )
        result = {"url": url, "model": model, "text": focused, "prices_eur": prices_in_text(focused)}
        return result | {"truncated": True} | listing
    text = raw[:PAGE_CHARS]
    result = {"url": url, "text": text, "prices_eur": prices_in_text(text), "truncated": len(raw) > PAGE_CHARS}
    if listing:
        return result | listing  # Die Modellnamen reichen; tiefer lesen lohnt auf einer Übersicht nicht
    return result | {"note": TRUNCATED_NOTE} if result["truncated"] else result
