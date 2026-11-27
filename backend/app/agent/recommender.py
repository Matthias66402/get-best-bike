"""Fahrradberater: PydanticAI-Agent, der aus Websuche, Wissensbasis und Rahmengrößen-Tool eine Rangliste der 1 bis 5
am besten passenden Fahrräder erstellt (oder nachfragt bzw. ohne Rangliste antwortet).

Der Output-Validator `check_plausibility` lässt nur belegte Empfehlungen durch: Quell-URLs und Preise müssen aus den
Ergebnissen von `search_bikes` bzw. `read_product_page` stammen, das Budget wird eingehalten und bei bekannten
Körpermaßen gibt es eine Rahmengröße aus `recommend_frame_size`.
"""

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RetryPromptPart,
    RunContext,
    Tool,
    ToolCallPart,
    ToolDefinition,
    ToolOutput,
    ToolReturnPart,
    UserPromptPart,
)  # fmt: skip
from pydantic_ai.capabilities import Hooks, PrepareOutputTools

from app.agent.compaction import compact_earlier_messages
from app.agent.deps import AgentDeps
from app.agent.history import researched_categories
from app.agent.models import build_model, model_settings
from app.agent.recommendation_schema import (
    MAX_RANKING,
    VAGUE_NAME,
    BikeCategory,
    BikeRecommendation,
    ClarificationQuestion,
    RankedBike,
    TextAnswer,
)
from app.agent.tools.common import model_family, tool_error, words
from app.agent.tools.frame_size import NO_MEASUREMENTS, recommend_frame_size
from app.agent.tools.prices import price_in_text, prices_in_text
from app.agent.tools.research import research_knowledge
from app.agent.tools.web import is_listing, normalize_url, read_product_page, search_bikes
from app.config import get_settings

logger = logging.getLogger(__name__)

RecommenderOutput = BikeRecommendation | ClarificationQuestion | TextAnswer

# Web-Aufrufe je Antwort; danach blendet der Code das Tool aus und der Berater muss mit dem Gefundenen antworten.
# Ohne Grenze las er bei Rose eine Backroad-Variante nach der anderen (Beratung 7: über 12 Minuten, kein Ende).
TOOL_BUDGETS = {"search_bikes": 10, "read_product_page": 8}
# Aufrufe von recommend_frame_size je Antwort: eines je Rad der Rangliste plus eines Reserve. Beratung 21: Nach
# aufgebrauchtem Web-Budget nutzte das Modell es 12-mal als Ersatz-Suche ("Suche Trek Checkpoint ALR5 Preis") und
# lief damit 15 Minuten weiter
FRAME_SIZE_CALLS = 6
# Runden mit recommend_frame_size je Antwort. Beratung 26: fünf Runden mit je einem Aufruf statt einer mit allen,
# je Runde ein Modellaufruf (rund eine Minute insgesamt); eine zweite Runde bleibt für ersetzte Räder
FRAME_SIZE_ROUNDS = 2
# So oft weist der Validator eine Rangliste mit weniger als MAX_RANKING Rädern zurück, danach gilt sie
MIN_RANKING_RETRIES = 2

# Höchstens so viel über dem Budget, und nur mit Begründung in `cons`
BUDGET_TOLERANCE = 0.10
# Kategorien ohne Größenwahl (Einheitsgröße)
NO_FRAME_SIZE = {BikeCategory.cargo, BikeCategory.folding}
# Begriffe, an denen eine Quellseite die Kategorie erkennen lässt. Nennt die Seite nur Begriffe anderer Kategorien,
# passt die angegebene Kategorie nicht (Pendler-Lauf: Giant Talon E+, ein E-MTB-Hardtail, als "trekking")
CATEGORY_WORDS: dict[BikeCategory, re.Pattern[str]] = {
    category: re.compile(pattern, re.IGNORECASE)
    for category, pattern in {
        BikeCategory.city: r"city|stadt|urban|holland|tiefeinst",
        BikeCategory.trekking: r"trekking|touren|reise|pendl|alltag",
        BikeCategory.road: r"rennrad|road\s?bike|endurance",
        BikeCategory.gravel: r"gravel",
        BikeCategory.mtb_hardtail: r"mtb|mountain|hardtail",
        BikeCategory.mtb_fully: r"mtb|mountain|fully|full[\s-]?suspension|enduro",
        BikeCategory.cargo: r"lasten|cargo|long[\s-]?john|longtail",
        BikeCategory.folding: r"faltrad|folding|klapprad",
        BikeCategory.kids: r"kinder|kids|jugend",
    }.items()
}
# Gebraucht- und B-Ware-Angebote (Pendler-Lauf: Canyon Pathlite:ON aus einer Gebraucht-Kollektion)
USED_OFFER = re.compile(r"gebraucht|used|refurbished|pre-?owned|second-?hand|b-?ware", re.IGNORECASE)
# Notizen statt Werten in den Datenfeldern, z. B. "Bosch (Einträge auf Händlerseiten)" oder "~85 Nm"
SPEC_NOTE = re.compile(
    r"\b(laut|quelle|eintr\w*|händler\w*|angabe\w*|vermutlich|unklar|unbekannt|ca\.|nicht\s+(genannt|angegeben|bekannt)"
    r"|k\.\s?a\.|n/a)|[~?]",
    re.IGNORECASE,
)
# Plätze, die einen belegten Preis brauchen (Alle-Fälle-Lauf: 4 von 15 Rädern ohne Preis); 5 = alle
PRICED_RANKS = 3
# Umfeld einer Modellnennung (Zeichen davor und danach); darin müssen Kategorie und Preis zum Modell passen, damit
# ein Testartikel über mehrere Räder nicht Kategorie oder Preis eines anderen Rads liefert
CONTEXT_CHARS = 400
# Mindestanteil der Wörter des Modellnamens, die auf der Seite stehen müssen
MODEL_WORD_SHARE = 0.5


def model_on_page(model: str, text: str) -> bool:
    """Steht das Modell auf der Seite? Die Modellfamilie muss vorkommen und mindestens die Hälfte der Wörter."""
    model_words, page_words = words(model), set(words(text))
    family = model_family(model)
    if not model_words or (family and family not in page_words):
        return False
    return sum(w in page_words for w in model_words) >= MODEL_WORD_SHARE * len(model_words)


def model_context(model: str, text: str) -> str:
    """Textumfeld aller Nennungen der Modellfamilie; ohne Nennung der ganze Text."""
    family = model_family(model)
    if not family:
        return text
    spans = [
        text[max(0, m.start() - CONTEXT_CHARS) : m.end() + CONTEXT_CHARS]
        for m in re.finditer(rf"(?<![0-9a-zäöüß]){re.escape(family)}(?![0-9a-zäöüß])", text, re.IGNORECASE)
    ]
    return "\n".join(spans) if spans else text


def category_conflict(category: BikeCategory, text: str) -> list[str]:
    """Kategorien, die der Text statt der angegebenen nennt; leer, wenn sie passt oder keine nennt."""
    own = CATEGORY_WORDS.get(category)
    if own is None or own.search(text):
        return []
    return [other.value for other, pattern in CATEGORY_WORDS.items() if other != category and pattern.search(text)]


INSTRUCTIONS = f"""\
Du bist ein unabhängiger, erfahrener Fahrradberater und antwortest auf Deutsch. Der Nutzer beschreibt, was er sucht;
Du lieferst eine Rangliste der 1 bis 5 am besten passenden, aktuell erhältlichen Fahrräder (`final_recommendation`).

Vorgehen:
1. Kriterien klären: Einsatzzweck, Budget, Kategorie, mit oder ohne Motor, Körpermaße, besondere Wünsche. Fehlt der
   Einsatzzweck oder das Budget und lässt es sich nicht sinnvoll aus der Nachricht ableiten, frage mit `ask_user`
   nach (alle offenen Punkte in einer Frage). Körpermaße sind nicht Pflicht: Ohne sie gibt es keine Rahmengröße.
   Frage nie, ob neu oder gebraucht: Die Rangliste enthält immer Neuräder.
2. Hintergrund: Vor der ersten Rangliste einer Beratung `research_knowledge` aufrufen (falls noch nicht im Verlauf)
   und die Hinweise bei der Auswahl berücksichtigen.
3. Kandidaten finden mit `search_bikes`: 2-3 Suchen in derselben Runde, z. B. nach Tests bzw. Bestenlisten der
   Kategorie im Budget und mit `shops=true` nach Angeboten im Budget (z. B. "Gravelbike Carbon 2x bis 2500 Euro").
   Nur Modelle aus den Ergebnissen, nichts aus dem Gedächtnis. Tests und Übersichtsseiten (`listing: true`, z. B.
   Kategorien von Shops oder Preisvergleichen) liefern Kandidaten, aber keine Belege: Wiederhole keine allgemeine
   Suche mit leicht anderem Text, sondern suche die gefundenen Modelle einzeln mit `shops=true`, z. B.
   "Cube Kathmandu Pro" (mehrere solcher Suchen in derselben Runde).
4. Belegen mit `read_product_page` für die aussichtsreichsten Kandidaten (alle in derselben Runde): Preis,
   Ausstattung, Gewicht und, falls vorhanden, die Größentabelle, möglichst von der Produktseite des Modells. Nur
   URLs aus den Suchergebnissen; selbst zusammengesetzte URLs lehnt das Tool ab. Fehlt eine Produktseite oder ein
   Euro-Preis, suche das Modell mit `search_bikes` und `shops=true`.
5. Rahmengröße mit `recommend_frame_size` je Rad (mit Größentabelle der Produktseite, falls vorhanden), alle in
   derselben Runde, sobald Körpermaße bekannt sind (Nachricht oder Profil). `frame_size` muss zum Rad passen:
   Wünscht der Nutzer ausdrücklich eine Größe (z. B. 24 Zoll) und nennt das Tool sie als `alternative`, trage sie
   ein und übernimm dessen `note` in `cons`; weicht sein Wunsch ganz vom Ergebnis ab, gilt der Wunsch, und der
   Hinweis auf die errechnete Größe gehört in `cons`, nicht als widersprüchliche Größe in `frame_size`.
6. Ausgeben mit `final_recommendation`, beste Wahl auf Platz 1, mit 5 Rädern.
   Plätze 1 bis {PRICED_RANKS} brauchen einen Preis von einer Angebots- oder Produktseite; Testartikel allein
   reichen dafür oft nicht, Übersichtsseiten nie. `model` genau so, wie es auf der Quellseite steht.

Regeln für die Rangliste:
- `price_source_url` ist eine URL aus `search_bikes` oder `read_product_page`, auf der das Modell vorkommt.
- `price_eur` genau so, wie er auf dieser Seite steht (siehe `prices_eur` im Tool-Ergebnis); unbekannt = null.
- Preis höchstens 10 % über dem Budget, dann mit Hinweis in `cons`.
- Kein Motor, wenn der Nutzer ein Rad ohne Motor will, und umgekehrt.
- Keine Spezifikationen erfinden: Was keine Quelle nennt, bleibt null. Werte ohne Zusätze wie "(laut Quelle)".
- `price_source_url` ist eine Produkt-, Händler- oder Testseite, keine Startseite eines Herstellers und kein
  Gebraucht- oder B-Ware-Angebot.
- `category` so, wie die Quelle das Rad einordnet (ein E-MTB ist kein Trekkingrad).
- Bei E-Bike-Wunsch: Motor bzw. Akku (Wh) je Rad angeben.
- `match_score` ehrlich: 90+ nur, wenn alle Kriterien erfüllt sind.
- Folgenachrichten ("günstiger", "lieber ohne Motor", Antwort auf eine Rückfrage) passen die bestehende Auswahl an:
  Nutze zuerst die Seiten, die Du in dieser Beratung schon gelesen hast, und suche nur für fehlende oder nicht mehr
  passende Räder neu. Die Wissensbasis ist dann schon recherchiert.
- `brand` nur der Markenname wie auf der Seite (z. B. "Van Rysel", "Rose"), `model` nur der Modellname, ohne
  Anmerkungen in Klammern; Hinweise wie "Vorjahresmodell" gehören in `why_fits` oder `cons`.
- Lässt sich ein Wunsch im Budget nicht bei allen Rädern erfüllen (z. B. 2 Kettenblätter und Carbon), frage nicht
  nach: Setze die Räder, die ihn erfüllen, nach vorn, fülle mit den nächstbesten auf und nenne den Kompromiss in
  `cons`.

Fragen ohne Bedarf an einer Rangliste (z. B. "Was ist ein Gravelbike?", "Erklär mir Platz 2 genauer") beantwortest
Du mit `answer_user`. Scheitert eine einzelne Suche oder ein Seitenabruf (z. B. Netzwerkfehler, Seite nicht
lesbar, keine Treffer), wiederhole die Suche einmal bzw. nimm einen anderen Treffer; das ist kein Ausfall der
Websuche. Fehler und Aussagen aus früheren Antworten beschreiben nicht den jetzigen Zustand: Suche neu. Nur wenn in
dieser Antwort mehrere Aufrufe hintereinander mit Netzwerkfehlern scheitern, sag das mit `answer_user` und gib
allgemeine Hinweise aus der Wissensbasis, statt Modelle zu erfinden. Rufe unabhängige Tools immer gemeinsam in einer
Runde auf.
Deine Ausgabe beendet die Antwort: Kündige nie an, dass Du noch recherchierst, sondern recherchiere und gib dann
die Rangliste aus.
Je Antwort hast Du höchstens {TOOL_BUDGETS["search_bikes"]} Websuchen und {TOOL_BUDGETS["read_product_page"]} Seitenabrufe;
plane sie so, dass am Ende eine Rangliste steht. Suche nicht jede Modellvariante einzeln, sondern nimm die passendste.
"""


def tool_returns(messages: list[ModelMessage], tool_name: str) -> list[Any]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


def known_pages(messages: list[ModelMessage]) -> dict[str, str]:
    """URL (normalisiert) -> Text, den search_bikes bzw. read_product_page dem Modell in dieser Beratung geliefert
    haben. Auch aus früheren Läufen, damit Folgenachrichten bekannte Seiten weiter nutzen können."""
    pages: dict[str, str] = {}

    def add(url: str, text: str) -> None:
        key = normalize_url(url)
        pages[key] = f"{pages.get(key, '')}\n{text}"

    for content in tool_returns(messages, "search_bikes"):
        for hit in content.get("results", []) if isinstance(content, dict) else []:
            if isinstance(hit, dict) and hit.get("url"):
                add(hit["url"], f"{hit.get('title') or ''}\n{hit.get('snippet') or ''}")
    for content in tool_returns(messages, "read_product_page"):
        if isinstance(content, dict) and content.get("url"):
            add(content["url"], content.get("text") or "")
    return pages


def used_tool(messages: list[ModelMessage], tool_name: str) -> bool:
    return bool(tool_returns(messages, tool_name))


def current_run(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Nachrichten seit der letzten Nutzernachricht, also dieser Antwort (inkl. der Vorab-Aufrufe des Graphen)."""
    starts = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]
    return messages[starts[-1] :] if starts else messages


# Seitenabrufe nur für URLs aus der Websuche (oder vom Nutzer genannt); andere lädt der Code nicht. Beratung 29: sechs
# von neun Abrufen gingen an selbst gebaute URLs (404), danach war das Budget weg und es kam keine Rangliste. Die
# ersten FREE_UNKNOWN_URLS solcher Aufrufe je Antwort zählen nicht aufs Budget, weitere schon (sonst gäbe es keine
# Grenze). Beratung 7: Bei 3 freien zählten sechs abgelehnte Abrufe, nach vier gelesenen Seiten galt das Budget als
# aufgebraucht, und Platz 1 bis 3 durften ohne Preis bleiben.
FREE_UNKNOWN_URLS = 8
# Ebenso für erneute Abrufe einer schon gelesenen Seite (Beratung 31: fünf Wiederholungen verbrauchten das Budget).
# Zählten sie gar nicht, bliebe das Tool angeboten, und das Modell könnte endlos wiederholen (Beratung 33)
FREE_REPEATED_READS = 3
URL_IN_TEXT = re.compile(r"https?://[^\s<>\"'()\[\]]+")


UMLAUTS = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "ss"})


def url_key(url: str) -> str:
    """Vergleichsschlüssel für URLs, die das Modell beim Abschreiben leicht verändert: ohne Schema und „www.“,
    Umlaute und ihre Umschreibung gleich. Beratung 31: „…-lastenrader-…“ (so im Suchtreffer) wurde zu
    „…-lastenraeder-…“, „www.nimms-rad.de“ zu „nimms-rad.de“; beide Belege galten als unbekannt."""
    parts = urlsplit(normalize_url(url))
    host = parts.netloc.removeprefix("www.")
    path = parts.path.lower().translate(UMLAUTS).replace("ae", "a").replace("oe", "o").replace("ue", "u")
    return f"{host}{path}?{parts.query}" if parts.query else f"{host}{path}"


def match_url(url: str, urls: Iterable[str]) -> str | None:
    """Die bekannte URL zu `url` (exakt oder nach `url_key` gleich), sonst None."""
    urls = list(urls)
    normalized = normalize_url(url)
    if normalized in urls:
        return normalized
    key = url_key(url)
    return next((known for known in urls if url_key(known) == key), None)


def readable_urls(messages: list[ModelMessage]) -> set[str]:
    """Normalisierte URLs, die `read_product_page` laden darf: aus `search_bikes` bzw. `read_product_page` dieser
    Beratung oder aus einer Nachricht des Nutzers."""
    urls = set(known_pages(messages))
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    urls.update(normalize_url(url.rstrip(".,;:!?")) for url in URL_IN_TEXT.findall(part.content))
    return urls


def similar_urls(url: str, readable: set[str], limit: int = 3) -> list[str]:
    """Bekannte URLs, die zur abgelehnten passen: gemeinsame Wörter im Pfad, sonst gleicher Shop bzw. Hersteller."""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    tokens = {w for w in words(parts.path) if len(w) >= 3 and not w.isdigit()}

    def score(candidate: str) -> tuple[int, int]:
        other = urlsplit(candidate)
        return len(tokens & set(words(other.path))), int(other.netloc.removeprefix("www.") == host)

    return sorted((u for u in readable if any(score(u))), key=score, reverse=True)[:limit]


def page_url(call: ToolCallPart) -> str:
    try:
        return str(call.args_as_dict().get("url") or "")
    except ValueError:  # Argumente kein gültiges JSON
        return ""


def read_key(call: ToolCallPart) -> str:
    """Was ein Seitenabruf liest: die Seite und ggf. das Modell, auf das er zielt (`model`). Derselbe Testbericht für
    ein anderes Modell ist keine Wiederholung (Beratung 8)."""
    try:
        model = str(call.args_as_dict().get("model") or "")
    except ValueError:
        model = ""
    return f"{url_key(page_url(call))}#{model_family(model) or ''}"


def budget_calls(messages: list[ModelMessage], tool_name: str) -> list[ToolCallPart]:
    """Aufrufe von `tool_name` in dieser Antwort, die aufs Budget zählen."""
    calls = [
        part
        for message in current_run(messages)
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == tool_name
    ]
    if tool_name != "read_product_page":
        return calls
    readable = readable_urls(messages)
    counted, free, repeats, read = [], FREE_UNKNOWN_URLS, FREE_REPEATED_READS, set()
    for call in calls:
        url = page_url(call)
        if match_url(url, readable) is None:
            if free:
                free -= 1
                continue
        elif read_key(call) in read:
            # Erneuter Abruf derselben Seite: läuft nicht (enforce_budgets); die ersten zählen nicht, weitere schon
            if repeats:
                repeats -= 1
                continue
        else:
            read.add(read_key(call))
        counted.append(call)
    return counted


def read_before(messages: list[ModelMessage], call: ToolCallPart) -> bool:
    """Hat diese Antwort die Seite von `call` (mit demselben `model`) schon vor `call` abgerufen?"""
    key, readable = read_key(call), readable_urls(messages)
    for message in current_run(messages):
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if not (isinstance(part, ToolCallPart) and part.tool_name == "read_product_page"):
                continue
            if part.tool_call_id == call.tool_call_id:
                return False
            if read_key(part) == key and match_url(page_url(part), readable) is not None:
                return True
    return False


def calls_in_run(messages: list[ModelMessage], tool_name: str) -> int:
    return len(budget_calls(messages, tool_name))


def web_budget_left(messages: list[ModelMessage]) -> bool:
    return any(calls_in_run(messages, name) < limit for name, limit in TOOL_BUDGETS.items())


async def within_budget(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Blendet ein Web-Tool aus, sobald es in dieser Antwort TOOL_BUDGETS-mal aufgerufen wurde."""
    return tool_def if calls_in_run(ctx.messages, tool_def.name) < TOOL_BUDGETS[tool_def.name] else None


def failed_without_measurements(messages: list[ModelMessage]) -> bool:
    """Ist `recommend_frame_size` in dieser Antwort schon an fehlenden Körpermaßen gescheitert?"""
    return any(
        isinstance(content, dict) and content.get("error") == NO_MEASUREMENTS
        for content in tool_returns(current_run(messages), "recommend_frame_size")
    )


async def frame_size_budget(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Blendet `recommend_frame_size` nach FRAME_SIZE_CALLS Aufrufen in dieser Antwort aus, und schon nach dem ersten
    Fehlschlag mangels Körpermaßen, wenn weder Profil noch Intake Maße kennen. Beratung 22: Mit aufgebrauchtem
    Web-Budget war es das einzige verbliebene Tool; das Modell rief es sechsmal ohne Maße auf, statt auszugeben."""
    deps = ctx.deps
    measured = deps.height_cm or deps.inseam_cm or deps.criteria.get("height_cm") or deps.criteria.get("inseam_cm")
    if not measured and failed_without_measurements(ctx.messages):
        return None
    if calls_in_run(ctx.messages, tool_def.name) >= FRAME_SIZE_CALLS:
        return None
    if rounds_in_run(ctx.messages, tool_def.name) >= FRAME_SIZE_ROUNDS:
        return None
    if known := determined_sizes(ctx.messages):
        # Beratung 24: nach jeder Zurückweisung derselbe Aufruf (Kinderrad, 126 cm), je eine Modellrunde
        note = (
            "Schon ermittelt (gilt weiter, nicht erneut aufrufen): " + "; ".join(known) + ". Weitere Größen nur für "
            "neue Räder, alle zusammen in dieser Runde."
        )
        return replace(tool_def, description=f"{tool_def.description}\n\n{note}")
    return tool_def


def rounds_in_run(messages: list[ModelMessage], tool_name: str) -> int:
    """Modellantworten dieser Antwort, die `tool_name` aufrufen (parallele Aufrufe zählen als eine Runde)."""
    return sum(
        isinstance(message, ModelResponse)
        and any(isinstance(part, ToolCallPart) and part.tool_name == tool_name for part in message.parts)
        for message in current_run(messages)
    )


async def enforce_budgets(
    ctx: RunContext[AgentDeps], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, handler: Any
) -> Any:
    """Setzt TOOL_BUDGETS auch innerhalb einer Runde durch. `within_budget` blendet ein Tool erst vor der nächsten
    Runde aus; parallele Aufrufe einer Runde konnten das Budget überschreiten (Beratung 26: 11 statt 8 Seitenabrufe).
    Aufrufe über dem Budget laufen nicht und bekommen einen Tool-Fehler, ebenso Seitenabrufe für URLs, die nicht aus
    der Websuche stammen (siehe FREE_UNKNOWN_URLS)."""
    if call.tool_name == "read_product_page":
        url, readable = page_url(call), readable_urls(ctx.messages)
        if match_url(url, readable) is not None and read_before(ctx.messages, call):
            # Beratung 31: Monty dreimal, Tern dreimal, FAZ zweimal gelesen, nachdem der Validator Angaben bemängelt
            # hatte; die Wiederholungen verbrauchten das Budget für weitere Räder
            return tool_error(
                f"Die Seite {url} hast Du in dieser Antwort schon so gelesen, ihr Text steht weiter oben; ein neuer "
                "Abruf liefert nichts anderes. Fehlt ein Modell, weil die Seite gekürzt war, rufe sie mit `model` "
                "auf; sonst nimm die Angaben von dort, belege das Rad mit einem anderen Treffer oder ersetze es."
            )
        if match_url(url, readable) is None:
            # Beratung 30: Das Modell baute trotz Ablehnung weiter URLs, eine davon zweimal; deshalb gleich Vorschläge
            suggestions = similar_urls(url, readable)
            hint = (
                "Passende Treffer aus der Suche: " + ", ".join(suggestions) + "."
                if suggestions
                else "Für dieses Modell gibt es noch keinen Treffer: Suche es mit `search_bikes`."
            )
            return tool_error(
                f"Die URL {url} stammt nicht aus `search_bikes` und wird nicht geladen; selbst zusammengesetzte URLs "
                f"führen meist ins Leere (404). {hint}"
            )
    limit = TOOL_BUDGETS.get(call.tool_name)
    if limit is not None:
        ids = [part.tool_call_id for part in budget_calls(ctx.messages, call.tool_name)]
        if call.tool_call_id in ids and ids.index(call.tool_call_id) >= limit:
            return tool_error(
                f"Budget von {limit} Aufrufen für `{call.tool_name}` in dieser Antwort ist aufgebraucht; nutze die "
                "bisherigen Ergebnisse."
            )
    result = await handler(args)
    if call.tool_name == "search_bikes" and repeats_earlier_search(result, ctx.messages):
        # Beratung 14: sieben fast gleiche Händlersuchen („… 'Shimano Deore' '1500 Euro' '2026'“) mit denselben
        # Übersichtsseiten verbrauchten das Suchbudget; nach einzelnen Modellen wurde nie gesucht
        result = result | {"note": REPEATED_SEARCH_NOTE}
    return result


REPEATED_HITS_SHARE = 0.8
REPEATED_SEARCH_NOTE = (
    "Fast alle Treffer kennst Du schon aus früheren Suchen dieser Antwort; weitere Umformulierungen bringen nichts "
    'Neues. Suche jetzt die gefundenen Modelle einzeln mit `shops=true` (z. B. "Cube Kathmandu Pro"), um ihre '
    "Produktseiten zu finden."
)


def repeats_earlier_search(result: Any, messages: list[ModelMessage]) -> bool:
    """Liefert eine Suche zu mindestens REPEATED_HITS_SHARE Treffer, die frühere Suchen dieser Antwort schon hatten?"""
    hits = [hit["url"] for hit in result.get("results", []) if hit.get("url")] if isinstance(result, dict) else []
    if len(hits) < 2:
        return False
    seen = {
        url_key(hit["url"])
        for content in tool_returns(current_run(messages), "search_bikes")
        if isinstance(content, dict)
        for hit in content.get("results", [])
        if isinstance(hit, dict) and hit.get("url")
    }
    return sum(url_key(url) in seen for url in hits) >= REPEATED_HITS_SHARE * len(hits)


def determined_sizes(messages: list[ModelMessage]) -> list[str]:
    """Rahmengrößen, die `recommend_frame_size` in dieser Antwort schon geliefert hat, z. B. „kids: 24 Zoll“."""
    sizes = []
    for content in tool_returns(current_run(messages), "recommend_frame_size"):
        if isinstance(content, dict) and content.get("size"):
            label = content.get("model") or content.get("category")
            sizes.append(f"{label}: {content['size']}")
    return list(dict.fromkeys(sizes))


WEB_BUDGET_USED_NOTE = (
    "Websuchen und Seitenabrufe dieser Antwort sind aufgebraucht. Gib jetzt die Rangliste mit `final_recommendation` "
    "aus den gefundenen Seiten aus; Preise, die keine Seite belegt, setzt Du auf null und nennst das in "
    "`buying_tips`. Rufe keine anderen Tools als Ersatz für eine Suche auf."
)


def question_needless(ctx: RunContext[AgentDeps]) -> bool:
    """Einsatzzweck und Budget sind laut Intake bekannt; dann lehnt `check_question_needed` jede Rückfrage ab."""
    return bool(ctx.deps.criteria.get("use_case")) and bool(ctx.deps.criteria.get("budget_eur"))


# Fehlermeldungen aus http_error_message (Tavily nicht erreichbar, Zeitüberschreitung, HTTP-Fehler)
NETWORK_ERROR = re.compile(r"nicht erreichbar|nicht rechtzeitig geantwortet|meldet HTTP")
WEB_OUTAGE_ERRORS = 2


def web_outage(messages: list[ModelMessage]) -> bool:
    """Scheiterten in dieser Antwort mehrere Web-Aufrufe mit Netzwerkfehlern? Dann soll das Modell das mit
    `answer_user` sagen können (siehe INSTRUCTIONS), statt Räder ohne Belege auszugeben."""
    run = current_run(messages)
    errors = [
        content
        for name in TOOL_BUDGETS
        for content in tool_returns(run, name)
        if isinstance(content, dict) and NETWORK_ERROR.search(str(content.get("error", "")))
    ]
    return len(errors) >= WEB_OUTAGE_ERRORS


def ranking_only(ctx: RunContext[AgentDeps]) -> bool:
    """Steht eine Rangliste an, bleibt nur die Rangliste als Ausgabe, wenn Einsatzzweck und Budget bekannt sind, oder
    sobald es gelesene Seiten gibt und die Web-Tools aufgebraucht sind bzw. schon eine Rückfrage oder Textantwort
    zurückgewiesen wurde. Ausnahmen: Ausfall der Websuche und, wenn Einsatzzweck oder Budget fehlen, der letzte
    Versuch, damit der Lauf nicht an der Wiederholungsgrenze scheitert. Stehen beide fest, rettet `best_last_attempt`
    die Rangliste; Beratung 8: Im letzten Versuch fragte das Modell per `answer_user` nach 10 % mehr Budget.

    Beratung 29: Mit leerem Web-Budget fragte das Modell erst per `ask_user`, dann per `answer_user`, ob es weiter
    suchen darf; die zweite Textantwort beendete den Lauf ohne Rangliste. Beratung 30: Nach einer Rangliste mit fünf
    Mängeln wich es auf „Soll ich recherchieren?“ aus, die zweite Rückfrage galt. Beratungen 4 bis 6: Rückfrage per
    `ask_user` bzw., als die ausgeblendet war, per `answer_user`, obwohl Einsatzzweck und Budget feststanden; jede
    Ablehnung kostete eine der Korrekturen, die danach für die Rangliste fehlten."""
    evaded = output_rejected(ctx.messages, "ask_user") or output_rejected(ctx.messages, "answer_user")
    stuck = bool(known_pages(ctx.messages)) and (evaded or not web_budget_left(ctx.messages))
    needless = question_needless(ctx)
    return (
        ctx.deps.ranking_due
        and bool(ctx.deps.settings.tavily_api_key)
        and (needless or stuck)
        and not web_outage(ctx.messages)
        and (needless or ctx.retry < ctx.max_retries)
    )


async def offer_outputs(ctx: RunContext[AgentDeps], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    if ranking_only(ctx):
        return [tool_def for tool_def in tool_defs if tool_def.name == "final_recommendation"]
    if question_needless(ctx):
        return [tool_def for tool_def in tool_defs if tool_def.name != "ask_user"]
    return tool_defs


async def research_once(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Blendet `research_knowledge` aus, sobald in dieser Beratung recherchiert wurde (auch vorab durch den Graphen).

    Beratung 8: viermal recherchiert, einmal sogar "find product pages with prices"; jede Recherche ist ein eigener
    Agent mit mehreren Modellaufrufen und kostete bei knappem Anfragelimit jeweils Minuten.
    """
    history_json = ModelMessagesTypeAdapter.dump_python(ctx.messages, mode="json")
    return None if researched_categories(history_json) else tool_def


def rejections(messages: list[ModelMessage], tool_name: str) -> int:
    """Wie oft wurde in dieser Antwort eine Ausgabe über `tool_name` zurückgewiesen?"""
    return sum(
        getattr(part, "tool_name", None) == tool_name
        and (isinstance(part, RetryPromptPart) or (isinstance(part, ToolReturnPart) and part.outcome == "retried"))
        for message in current_run(messages)
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def output_rejected(messages: list[ModelMessage], tool_name: str) -> bool:
    """Wurde in dieser Antwort schon eine Ausgabe über `tool_name` zurückgewiesen?"""
    return rejections(messages, tool_name) > 0


def fix_source_urls(output: BikeRecommendation, pages: dict[str, str]) -> None:
    """Ersetzt leicht verschriebene Beleg-URLs (siehe `url_key`) durch die bekannte URL, damit der Link stimmt."""
    for bike in output.ranking:
        if (known := match_url(bike.price_source_url, pages)) and known != normalize_url(bike.price_source_url):
            bike.price_source_url = known


def check_sources(
    output: BikeRecommendation,
    pages: dict[str, str],
    web_budget: bool = True,
    *,
    salvaging: bool = False,
    final_try: bool = False,
) -> None:
    """`web_budget=False`, wenn Websuchen und Seitenabrufe dieser Antwort aufgebraucht sind: Dann ließe sich nichts
    mehr nachsuchen. Fehlende Preise gelten, außer die Belegseite nennt beim Modell einen Euro-Preis, und für
    unbelegte Räder heißt die Rückmeldung „streichen“ statt „suchen“ (Beratung 22: Das Modell wollte das Radon per
    Suche ersetzen, fand kein Such-Tool mehr und lief in eine Schleife). Beim Retten (`salvaging`) bleibt ein
    fehlender Preis eine Lücke in den Daten und kein Grund zum Streichen. Übersichtsseiten belegen Platz 1 bis
    PRICED_RANKS nur noch im letzten Versuch (`final_try`) oder ohne Web-Budget: lieber so belegt als gar nicht."""
    if not pages:
        raise ModelRetry(
            "Die Rangliste braucht Belege aus der Websuche. Suche mit `search_bikes` nach passenden Modellen; "
            "funktioniert die Websuche nicht, antworte mit `answer_user`."
        )
    problems = []
    for bike in output.ranking:
        name = f"Platz {bike.rank} ({bike.brand} {bike.model})"
        known = match_url(bike.price_source_url, pages)
        text = pages[known] if known else None
        if text is None:
            problems.append(
                f"{name}: Die URL {bike.price_source_url} stammt nicht aus `search_bikes` bzw. `read_product_page`."
            )
            continue
        if urlsplit(bike.price_source_url).path.strip("/") == "":
            # Erster echter Lauf: Corratec mit https://www.corratec.com als Beleg, ohne jede technische Angabe
            problems.append(
                f"{name}: {bike.price_source_url} ist nur die Startseite. Belege das Modell mit einer Produkt-, "
                "Händler- oder Testseite (`search_bikes` mit Modellnamen, dann `read_product_page`)."
            )
            continue
        if USED_OFFER.search(urlsplit(bike.price_source_url).path):
            problems.append(
                f"{name}: {bike.price_source_url} ist ein Gebraucht- bzw. B-Ware-Angebot. Empfiehl Neuräder und "
                "belege sie mit einem Neuangebot."
            )
        if bike.brand.casefold() not in text.casefold():
            problems.append(f"{name}: Die Marke „{bike.brand}“ kommt auf der Seite {bike.price_source_url} nicht vor.")
        if not model_on_page(bike.model, text):
            # Alle-Fälle-Lauf: "Decathlon City Bike 20 Zoll D4 Rock", aus mehreren Rädern eines Tests zusammengesetzt
            problems.append(
                f"{name}: Das Modell „{bike.model}“ steht so nicht auf {bike.price_source_url}. Übernimm den "
                "Modellnamen genau von der Seite oder belege das Rad mit einer passenden Seite."
            )
            continue
        # Kategorie und Preis im Umfeld des Modells: Testartikel nennen mehrere Räder mit eigener Kategorie und Preis
        context = model_context(bike.model, text)
        if missing := [code for code in equipment_codes(bike.model) if code not in words(context)]:
            # Beratung 10: „Kanzo Adventure Alu – Shimano GRX400 2x10“, belegt mit der Seite der GRX600-Version
            problems.append(
                f"{name}: „{' '.join(missing)}“ aus dem Modellnamen steht nicht auf {bike.price_source_url}; die "
                "Seite zeigt wohl eine andere Ausstattung. Nimm den Modellnamen und den Preis genau dieser Seite "
                "oder belege die Ausstattung mit ihrer eigenen Seite."
            )
            continue
        if bike.rank <= PRICED_RANKS and web_budget and not final_try and is_listing(known, text):
            # Beratung 14: alle fünf Trekkingräder nur mit einer fahrrad-xxl-Kategorieseite belegt, ohne Ausstattung
            problems.append(
                f"{name}: {bike.price_source_url} ist eine Übersichtsseite mit vielen Rädern, keine Produktseite; "
                f"Ausstattung und Angebot sind dort nicht belegt. Suche das Modell einzeln mit "
                f'`search_bikes("{bike.brand} {bike.model}", shops=true)` und belege es mit seiner Produktseite '
                f"(`read_product_page`); findest Du keine, setze es hinter Platz {PRICED_RANKS} oder ersetze es."
            )
            continue
        # Der URL-Pfad ordnet das Rad oft ein, ohne dass der Seitentext die Kategorie nennt. Beratung 30: Die
        # Rose-Seite /bikes/gravel/.../backroad-al nannte „Gravel“ nicht, wohl aber den Reifen „Continental Grand Prix
        # Urban“; das Backroad galt deshalb als City-Rad
        if others := category_conflict(bike.category, f"{urlsplit(bike.price_source_url).path}\n{context}"):
            problems.append(
                f"{name}: Die Quelle beschreibt das Rad als {', '.join(others)}, nicht als {bike.category.value}. "
                "Setze die richtige Kategorie und prüfe, ob das Rad dann noch zum Wunsch passt; sonst ersetze es."
            )
        if bike.price_eur is not None and not price_in_text(bike.price_eur, context):
            problems.append(
                f"{name}: Der Preis {bike.price_eur:g} € steht auf {bike.price_source_url} nicht beim Modell. "
                "Übernimm den Preis von der Seite (`prices_eur`), lies die Produktseite mit `read_product_page` "
                "(bei einer gekürzten Seite mit `model`) oder setze null."
            )
        if bike.price_eur is None and bike.rank <= PRICED_RANKS and web_budget:
            problems.append(
                f"{name}: Für Platz 1 bis {PRICED_RANKS} braucht es einen belegten Preis. Suche das Modell mit "
                "`search_bikes` (Modellname, `shops=true`) und lies ein Angebot mit `read_product_page`; findest Du "
                "keinen Preis, ersetze das Rad oder setze es weiter nach hinten."
            )
        elif (
            bike.price_eur is None
            and bike.rank <= PRICED_RANKS
            and not salvaging
            and (found := prices_in_text(context))
        ):
            # Beratung 7: Mit aufgebrauchtem Budget setzte das Modell alle Preise auf null, auch die 2.000 € des Cube
            # Nuroad C:62, die auf der Belegseite beim Modell standen und im Versuch davor nicht bemängelt waren
            listed = ", ".join(f"{price:g} €" for price in dict.fromkeys(found))
            problems.append(
                f"{name}: Die Seite {bike.price_source_url} nennt beim Modell {listed}. Übernimm den Preis, der zu "
                "diesem Modell gehört, statt null."
            )
    if problems:
        message = "Nicht belegte Angaben in der Rangliste:\n- " + "\n- ".join(problems)
        if not web_budget:
            message += (
                "\nWebsuchen und Seitenabrufe sind aufgebraucht, suche nicht weiter: Korrigiere die Angaben anhand "
                "der gelesenen Seiten oder streiche die betroffenen Räder und rücke die übrigen auf; eine kürzere "
                "Rangliste ist in Ordnung."
            )
        raise ModelRetry(message)


def check_budget(output: BikeRecommendation) -> None:
    budget = output.requirements.budget_eur
    if not budget:
        return
    too_expensive = [
        b for b in output.ranking if b.price_eur is not None and b.price_eur > budget * (1 + BUDGET_TOLERANCE)
    ]
    if too_expensive:
        names = ", ".join(f"Platz {b.rank} ({b.price_eur:g} €)" for b in too_expensive)
        raise ModelRetry(
            f"Budget {budget:g} €: {names} liegt mehr als {BUDGET_TOLERANCE:.0%} darüber. Ersetze diese Räder durch "
            "günstigere Kandidaten aus der Websuche oder lass sie weg."
        )


# Hersteller von Schaltgruppen: kein Teil des Radnamens, sondern der Ausstattung
COMPONENT_BRANDS = {"shimano", "sram", "campagnolo", "microshift"}


def equipment_codes(model: str) -> list[str]:
    """Wörter aus Buchstaben und Ziffern, die eine Ausstattung bezeichnen, z. B. „grx400“, „2x10“, „105di2“."""
    return [w for w in words(model) if re.search(r"[a-zäöüß]", w) and re.search(r"\d", w)]


def base_model(model: str) -> tuple[str, ...]:
    """Modellname ohne Ausstattung: ohne Ausstattungscodes und Schaltgruppen-Hersteller."""
    codes = set(equipment_codes(model))
    return tuple(w for w in words(model) if w not in codes and w not in COMPONENT_BRANDS)


def check_distinct_models(output: BikeRecommendation) -> None:
    """Jedes Modell nur einmal. Beratung 10: Ridley Kanzo Adventure Alu mit GRX400 auf Platz 1 und mit GRX600 auf
    Platz 4; die Liste hatte damit nur vier verschiedene Räder."""
    seen: dict[tuple[str, tuple[str, ...]], int] = {}
    doubles = []
    for bike in output.ranking:
        key = (bike.brand.casefold(), base_model(bike.model))
        if key in seen:
            doubles.append(f"Platz {bike.rank} ist dasselbe Modell wie Platz {seen[key]} ({bike.brand} {bike.model})")
        else:
            seen[key] = bike.rank
    if doubles:
        raise ModelRetry(
            f"{'; '.join(doubles)}, nur in anderer Ausstattung. Nimm die passendere Ausstattung und ersetze den "
            "anderen Platz durch ein anderes Modell; die zweite Ausstattung kannst Du in `cons` erwähnen."
        )


def check_specific_models(output: BikeRecommendation, web_budget: bool = True) -> None:
    """Jeder Platz ist ein konkretes Modell. Beratungen 5 und 6: „Grizl (Aluminium-Varianten)“, „Nuroad /
    Nuroad-ähnliche Alu-Modelle“ und „Allgemeine Händlerangebote“, belegt mit Testberichten und Kategorieseiten.
    Bemängelt wurden sie erst als unbelegt, als das Web-Budget aufgebraucht war; dann blieb nur Streichen."""
    vague = [
        f"Platz {b.rank} („{b.brand} {b.model}“)"
        for b in output.ranking
        if VAGUE_NAME.search(b.brand) or VAGUE_NAME.search(b.model)
    ]
    if vague:
        raise ModelRetry(
            f"Sammeleinträge statt eines Modells: {', '.join(vague)}. Jeder Platz ist genau ein Modell mit Namen wie "
            "auf seiner Produktseite, z. B. „Grizl 7“ statt „Grizl (Alu-Varianten)“. "
            + (
                "Such das konkrete Modell mit `search_bikes` (`shops=true`) und belege es mit einer Produkt- oder "
                "Händlerseite, "
                "oder ersetze den Platz."
                if web_budget
                else "Nimm ein konkretes Modell von den gelesenen Seiten oder streiche den Platz."
            )
        )


def check_spec_notes(output: BikeRecommendation) -> None:
    """Datenfelder enthalten Werte, keine Anmerkungen; Unsicheres bleibt null und gehört in `cons`."""
    noted = [
        f"Platz {bike.rank} {field}: „{value}“"
        for bike in output.ranking
        for field, value in bike.key_specs.model_dump().items()
        if isinstance(value, str) and SPEC_NOTE.search(value)
    ]
    if noted:
        raise ModelRetry(
            "Datenfelder mit Anmerkungen statt Werten:\n- " + "\n- ".join(noted) + "\nTrage nur den Wert ein, z. B. "
            '"Bosch Performance Line CX, 85 Nm"; ist er unsicher, setze null und erwähne es in `cons`.'
        )


def without_spec_notes(bike: RankedBike) -> RankedBike:
    """Setzt Datenfelder mit Anmerkungen (siehe `check_spec_notes`) auf null."""
    noted = {
        field: None
        for field, value in bike.key_specs.model_dump().items()
        if isinstance(value, str) and SPEC_NOTE.search(value)
    }
    return bike.model_copy(update={"key_specs": bike.key_specs.model_copy(update=noted)}) if noted else bike


def motor_unknown(output: BikeRecommendation) -> list[RankedBike]:
    """E-Bike gewünscht, aber weder Motor noch Akku angegeben."""
    if output.requirements.electric is not True:
        return []
    return [b for b in output.ranking if not (b.key_specs.motor or b.key_specs.battery_wh)]


def check_electric(output: BikeRecommendation, *, require_specs: bool = True) -> None:
    """`require_specs=False` (beim Retten der Rangliste): Ein E-Bike ohne Angaben zu Motor und Akku bleibt drin."""
    motorized = [b for b in output.ranking if b.key_specs.motor or b.key_specs.battery_wh]
    if output.requirements.electric is False and motorized:
        names = ", ".join(f"Platz {b.rank}" for b in motorized)
        raise ModelRetry(f"Der Nutzer möchte ein Rad ohne Motor, {names} hat einen Motor. Ersetze es.")
    if require_specs and (unknown := motor_unknown(output)):
        names = ", ".join(f"Platz {b.rank}" for b in unknown)
        raise ModelRetry(
            f"Der Nutzer möchte ein E-Bike, aber für {names} fehlen Motor und Akku. Lies die Produktseite mit "
            "`read_product_page` und trage Motor bzw. Akku (Wh) ein, oder ersetze das Rad."
        )


def check_ranking_size(output: BikeRecommendation, ctx: RunContext[AgentDeps]) -> None:
    """Die Ergebnisseite zeigt MAX_RANKING Räder. Weniger nur, wenn das Web-Budget aufgebraucht ist, schon
    MIN_RANKING_RETRIES Ranglisten zurückgewiesen wurden oder der letzte Versuch läuft, damit die Antwort nicht an
    der Wiederholungsgrenze scheitert.

    Gezählt werden nur zurückgewiesene Ranglisten, nicht ctx.retry: Beratung 26 zählte eine abgelehnte Rückfrage
    mit, schon nach einer abgelehnten Rangliste galt die Regel nicht mehr, und das Modell gab ein einziges Rad mit der
    Zusammenfassung „dummy“ aus."""
    missing = MAX_RANKING - len(output.ranking)
    if (
        missing <= 0
        or rejections(ctx.messages, "final_recommendation") >= MIN_RANKING_RETRIES
        or ctx.retry >= ctx.max_retries
        or not web_budget_left(ctx.messages)
    ):
        return
    raise ModelRetry(
        f"Die Rangliste soll {MAX_RANKING} Räder enthalten, es fehlen {missing}. Ergänze passende Kandidaten aus "
        f"der Websuche; Plätze {PRICED_RANKS + 1} bis {MAX_RANKING} dürfen ohne Preis sein (price_eur = null)."
    )


PLACEHOLDER = re.compile(r"\b(dummy|placeholder|platzhalter|lorem|ipsum|todo|tbd|test|xxx)\b", re.IGNORECASE)


def check_placeholders(output: BikeRecommendation) -> None:
    """Beratung 26: Rangliste mit der Zusammenfassung „dummy“."""
    fields = {"title": output.title, "summary": output.summary}
    fields |= {f"Platz {b.rank} why_fits": b.why_fits for b in output.ranking}
    if bad := [name for name, text in fields.items() if len(text.strip()) < 10 or PLACEHOLDER.fullmatch(text.strip())]:
        raise ModelRetry(
            f"Platzhalter statt Inhalt in {', '.join(bad)}. Schreibe echte Texte: Was gesucht war, warum Platz 1 "
            "vorn liegt und warum jedes Rad passt."
        )


def check_question_needed(output: ClarificationQuestion, ctx: RunContext[AgentDeps]) -> None:
    """Rückfrage nur, wenn Einsatzzweck oder Budget fehlen (laut Intake). Wurde schon eine Rückfrage zurückgewiesen
    und stellt das Modell sie trotzdem, gilt sie. Alle-Fälle-Lauf: sieben Fragen zum E-Lastenrad, obwohl Einsatz,
    Budget und Motor in der Anfrage standen. Zählte jede frühere Zurückweisung (auch einer Rangliste), wich das
    Modell auf eine Rückfrage aus (Beratung 8: "Ist Carbon ein Muss?" nach einer abgelehnten Rangliste)."""
    criteria = ctx.deps.criteria
    if ctx.deps.ranking_due and not output.missing_fields and ctx.retry < ctx.max_retries:
        # Beratung 30: „Soll ich jetzt fünf Neuräder recherchieren?“ ohne fehlende Angabe, als Ausweg nach Mängeln
        raise ModelRetry(
            "Die Rückfrage nennt keine fehlende Angabe (`missing_fields` ist leer), der Nutzer wartet auf die "
            "Rangliste. Frage nicht, ob Du recherchieren sollst, sondern gib die Rangliste mit `final_recommendation` "
            "aus den gefundenen Seiten aus; offene Punkte gehören in `buying_tips`."
        )
    if output_rejected(ctx.messages, "ask_user") or not criteria.get("use_case") or not criteria.get("budget_eur"):
        return
    raise ModelRetry(
        f"Einsatzzweck („{criteria['use_case']}“) und Budget ({criteria['budget_eur']:g} €) sind bekannt. Stelle "
        "keine Rückfrage, sondern erstelle die Rangliste; fehlende Details (z. B. Körpermaße) erwähnst Du in "
        "`buying_tips`. Frage nur nach, wenn ohne die Antwort wirklich keine sinnvolle Auswahl möglich ist."
    )


def check_answer_needed(ctx: RunContext[AgentDeps]) -> None:
    """Steht laut Intake eine Rangliste an, ist eine Textantwort ein Ausweichen. Beratung 9: Nach der
    abgelehnten Rückfrage zu 1x/2x antwortete das Modell "Ich recherchiere jetzt ... soll ich nur in DE-Shops
    suchen?" und beendete damit nach 9 Minuten die Antwort ohne Rangliste. Beim zweiten Mal gilt sie."""
    if not ctx.deps.ranking_due or output_rejected(ctx.messages, "answer_user"):
        return
    raise ModelRetry(
        "Der Nutzer wartet auf die Rangliste, und Deine Ausgabe beendet die Antwort. Kündige nichts an und stelle "
        "keine Frage, sondern gib jetzt mit `final_recommendation` die Rangliste aus den gefundenen Seiten aus; "
        "Kompromisse nennst Du in `cons`, offene Punkte in `buying_tips`."
    )


def check_frame_sizes(output: BikeRecommendation, deps: AgentDeps, messages: list[ModelMessage]) -> None:
    measured = output.requirements.height_cm or output.requirements.inseam_cm or deps.height_cm or deps.inseam_cm
    # Das Tool kannte keine Maße (und ist dann ausgeblendet): eine Größe ließe sich nicht mehr bestimmen
    if not measured or failed_without_measurements(messages):
        return
    # Kinderräder: Die Maße im Profil sind die des Nutzers. Eine Größe ist nur Pflicht, wenn das Tool mit der
    # Körpergröße des Kindes schon eine geliefert hat; sonst ließe sie sich nicht bestimmen
    kids_sized = any(
        isinstance(content, dict) and content.get("category") == BikeCategory.kids.value and content.get("size")
        for content in tool_returns(current_run(messages), "recommend_frame_size")
    )
    needs_size = [
        b for b in output.ranking if b.category not in NO_FRAME_SIZE and (b.category != BikeCategory.kids or kids_sized)
    ]
    if needs_size and not used_tool(messages, "recommend_frame_size"):
        raise ModelRetry(
            "Die Körpermaße sind bekannt: Bestimme die Rahmengrößen mit `recommend_frame_size` (je Rad, mit "
            "Größentabelle der Produktseite, falls vorhanden) und trage sie in `frame_size` ein."
        )
    if missing := [b for b in needs_size if not b.frame_size]:
        names = ", ".join(f"Platz {b.rank}" for b in missing)
        raise ModelRetry(f"`frame_size` fehlt für {names}; nimm die Größe aus `recommend_frame_size`.")


KEEP_RESULTS_NOTE = (
    "Behebe alle genannten Punkte in einem Schritt. Ergebnisse früherer Tool-Aufrufe dieser Antwort gelten weiter, "
    "z. B. Rahmengrößen aus `recommend_frame_size`: Übernimm sie, statt das Tool erneut aufzurufen."
)


def collect_problems(*checks: Callable[[], None]) -> list[str]:
    """Führt alle Prüfungen aus und sammelt ihre Rückmeldungen. Beratung 24: Nacheinander gemeldet kam der
    Budgetfehler von Platz 3 erst beim vierten Versuch ans Licht, danach war die Wiederholungsgrenze erreicht."""
    problems = []
    for check in checks:
        try:
            check()
        except ModelRetry as exc:
            problems.append(str(exc))
    return problems


def check_researched(messages: list[ModelMessage]) -> None:
    if not researched_categories(ModelMessagesTypeAdapter.dump_python(messages, mode="json")):
        raise ModelRetry(
            "Bevor Du die Rangliste ausgibst, recherchiere mit `research_knowledge` in der Wissensbasis zur "
            "Kategorie und zum Einsatzzweck und berücksichtige die Hinweise."
        )


def bike_problems(output: BikeRecommendation, ctx: RunContext[AgentDeps], *, salvaging: bool = False) -> list[str]:
    """Prüfungen, die einzelne Räder betreffen; ein betroffenes Rad lässt sich notfalls streichen. Beim Retten
    (`salvaging`) zählen fehlende Motor-/Akkuangaben nicht: Sie sind eine Lücke in den Daten, kein falsches Rad."""
    return collect_problems(
        lambda: check_sources(
            output,
            known_pages(ctx.messages),
            web_budget=web_budget_left(ctx.messages),
            salvaging=salvaging,
            final_try=ctx.retry >= ctx.max_retries,
        ),
        lambda: check_budget(output),
        lambda: check_spec_notes(output),
        lambda: check_specific_models(output, web_budget=web_budget_left(ctx.messages)),
        lambda: check_distinct_models(output),
        lambda: check_electric(output, require_specs=not salvaging),
        lambda: check_frame_sizes(output, ctx.deps, ctx.messages),
    )


def salvage_ranking(output: BikeRecommendation, ctx: RunContext[AgentDeps]) -> BikeRecommendation | None:
    """Behält die Räder, die alle Prüfungen bestehen (geprüft auf dem Rang, den sie dann bekommen), und rückt sie auf;
    None, wenn keines übrig bleibt. Datenfelder mit Anmerkungen werden vorher geleert statt das Rad zu streichen.

    Beratung 32: Im letzten Versuch fehlten bei drei Lastenrädern aus Testartikeln Motor und Akku, beim vierten
    stand „Bosch Performance Line (Angabe in Quelle)“ als Motor; gestrichen blieb ein einziges Rad übrig."""
    kept: list[RankedBike] = []
    for bike in output.ranking:
        candidate = without_spec_notes(bike.model_copy(update={"rank": len(kept) + 1}))
        if not bike_problems(output.model_copy(update={"ranking": [candidate]}), ctx, salvaging=True):
            kept.append(candidate)
    if not kept:
        return None
    salvaged = output.model_copy(update={"ranking": kept})
    tips = list(output.buying_tips)
    if dropped := len(output.ranking) - len(kept):
        tips.append(
            f"{dropped} weitere Kandidaten ließen sich nicht ausreichend belegen und fehlen deshalb in der Liste."
        )
    if unknown := motor_unknown(salvaged):
        names = ", ".join(f"Platz {b.rank}" for b in unknown)
        tips.append(f"Für {names} ließen sich Motor und Akku nicht belegen; frag beim Händler nach.")
    return salvaged.model_copy(update={"buying_tips": tips})


def rejected_rankings(messages: list[ModelMessage]) -> list[BikeRecommendation]:
    """Alle zurückgewiesenen Ranglisten dieser Antwort, älteste zuerst; ohne solche, die das Schema verletzten."""
    run = current_run(messages)
    rejected = {
        part.tool_call_id
        for message in run
        if isinstance(message, ModelRequest)
        for part in message.parts
        if getattr(part, "tool_name", None) == "final_recommendation"
        and (isinstance(part, RetryPromptPart) or (isinstance(part, ToolReturnPart) and part.outcome == "retried"))
    }
    calls = [
        part
        for message in run
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_call_id in rejected
    ]
    rankings = []
    for call in calls:
        try:
            rankings.append(BikeRecommendation.model_validate(call.args_as_dict()))
        except ValueError:
            continue
    return rankings


def previous_ranking(messages: list[ModelMessage]) -> BikeRecommendation | None:
    """Die zuletzt zurückgewiesene Rangliste dieser Antwort."""
    rankings = rejected_rankings(messages)
    return rankings[-1] if rankings else None


def bike_key(bike: RankedBike) -> tuple[str, str]:
    return bike.brand.strip().lower(), bike.model.strip().lower()


def dropped_sound_bikes(output: BikeRecommendation, ctx: RunContext[AgentDeps]) -> list[RankedBike]:
    """Räder der zuletzt zurückgewiesenen Rangliste, die die Rettung bestanden hätten, in `output` aber fehlen."""
    if not (previous := previous_ranking(ctx.messages)) or not (salvaged := salvage_ranking(previous, ctx)):
        return []
    current = {bike_key(b) for b in output.ranking}
    return [b for b in salvaged.ranking if bike_key(b) not in current]


def check_kept_bikes(output: BikeRecommendation, ctx: RunContext[AgentDeps]) -> None:
    """Beratung 2 (nach dem DB-Reset): Die zweite Rangliste hatte fünf Räder, bemängelt waren nur zwei davon und
    Anmerkungen in Datenfeldern. Die dritte enthielt ein einziges Rad, Titel „Vorabfrage zur Budget-Flexibilität“:
    Das Modell schob die zuvor abgelehnte Rückfrage als Rangliste nach und strich dafür zwei einwandfreie Räder."""
    if len(output.ranking) >= MAX_RANKING or not (dropped := dropped_sound_bikes(output, ctx)):
        return
    names = ", ".join(f"{b.brand} {b.model}" for b in dropped)
    raise ModelRetry(
        f"Gegenüber Deiner letzten Rangliste fehlen {names}, obwohl an ihnen nichts zu beanstanden war. Nimm sie "
        "wieder auf; streiche nur Räder, die bemängelt wurden. Eine Rückfrage an den Nutzer gehört nicht in die "
        "Rangliste, offene Punkte nennst Du in `buying_tips`."
    )


def best_last_attempt(
    output: BikeRecommendation, ctx: RunContext[AgentDeps], problems: list[str], global_problems: list[str]
) -> BikeRecommendation | None:
    """Letzter Versuch: die längste aus der (geretteten) aktuellen und allen geretteten früheren Ranglisten dieser
    Antwort. Die aktuelle zählt nur ohne globale Probleme (z. B. Platzhalter); bei Gleichstand gewinnt die neueste.

    Beratung 11: Versuch 1 hatte ein belegtes Rad, Versuch 2 und 4 waren Platzhalter-Ranglisten („Placeholder“,
    example.com), Versuch 3 nur eine erfundene URL. Gerettet wurde nur bei globalen Problemen nichts und auch nur
    die unmittelbar vorige Rangliste; der Lauf endete mit „Exceeded maximum output retries“."""
    candidates = [] if global_problems else [salvage_ranking(output, ctx) if problems else output]
    candidates += [salvage_ranking(ranking, ctx) for ranking in reversed(rejected_rankings(ctx.messages))]
    found = [candidate for candidate in candidates if candidate is not None]
    return max(found, key=lambda candidate: len(candidate.ranking)) if found else None


NO_RANKING_MESSAGE = (
    "Ich konnte diesmal keine Rangliste mit belegten Rädern erstellen: Für die gefundenen Modelle ließen sich Preis "
    "und Ausstattung auf keiner Händler- oder Herstellerseite nachweisen. Schick mir die Anfrage gern noch einmal, "
    "etwas allgemeiner oder mit einem Modell, das Dich interessiert, dann suche ich neu."
)


def build_agent() -> Agent[AgentDeps, RecommenderOutput]:
    settings = get_settings()
    agent = Agent(
        build_model(settings.llm_model, settings.llm_requests_per_minute, settings.llm_tokens_per_minute),
        name="bike-recommender",  # Name des Agent-Spans im Tracing
        deps_type=AgentDeps,
        output_type=[
            ToolOutput(
                BikeRecommendation,
                name="final_recommendation",
                description="Rangliste der 5 am besten passenden Räder ausgeben",
                strict=True,
            ),
            ToolOutput(
                ClarificationQuestion, name="ask_user", description="Rückfrage an den Nutzer stellen", strict=True
            ),
            ToolOutput(
                TextAnswer,
                name="answer_user",
                description="Antwort ohne Rangliste, z. B. Erklärung oder Frage zu einem Rad",
                strict=True,
            ),
        ],
        model_settings=model_settings(settings.llm_model, settings.llm_reasoning_effort),
        instructions=INSTRUCTIONS,
        tools=[
            Tool(search_bikes, prepare=within_budget),
            Tool(read_product_page, prepare=within_budget),
            Tool(recommend_frame_size, prepare=frame_size_budget),
            Tool(research_knowledge, prepare=research_once),
        ],
        retries=3,
        capabilities=[compact_earlier_messages, Hooks(tool_execute=enforce_budgets), PrepareOutputTools(offer_outputs)],
        defer_model_check=True,
    )

    @agent.instructions
    def current_context(ctx: RunContext[AgentDeps]) -> str:
        deps = ctx.deps
        lines = [f"Heute ist {deps.today.isoformat()}. Der Nutzer heißt {deps.user_name}."]
        if deps.height_cm or deps.inseam_cm:
            lines.append(
                "Körpermaße aus dem Profil (gelten, wenn der Nutzer nichts anderes nennt): "
                f"Körpergröße {deps.height_cm or '?'} cm, Schrittlänge {deps.inseam_cm or '?'} cm."
            )
        if not deps.settings.tavily_api_key:
            lines.append("Die Websuche ist nicht eingerichtet; eine Rangliste ist daher nicht möglich.")
        if deps.intake_note:
            lines.append(deps.intake_note)
        if deps.settings.tavily_api_key and not web_budget_left(ctx.messages):
            lines.append(WEB_BUDGET_USED_NOTE)
        return "\n".join(lines)

    @agent.output_validator
    def check_plausibility(ctx: RunContext[AgentDeps], output: RecommenderOutput) -> RecommenderOutput:
        """Selbstprüfung: Unbelegte oder unpassende Ranglisten gehen mit konkretem Hinweis zurück an das Modell."""
        try:
            return validate(ctx, output)
        except ModelRetry as exc:
            # Im Log nachvollziehbar, warum ein Lauf weitere Runden braucht
            logger.warning("Validator weist %s zurück (Versuch %d): %s", type(output).__name__, ctx.retry + 1, exc)
            raise

    def validate(ctx: RunContext[AgentDeps], output: RecommenderOutput) -> RecommenderOutput:
        if isinstance(output, ClarificationQuestion):
            check_question_needed(output, ctx)
        if isinstance(output, TextAnswer):
            check_answer_needed(ctx)
        if not isinstance(output, BikeRecommendation):
            return output
        fix_source_urls(output, known_pages(ctx.messages))
        final_try = ctx.retry >= ctx.max_retries
        global_problems = collect_problems(
            lambda: check_researched(ctx.messages),
            lambda: check_ranking_size(output, ctx),
            lambda: check_placeholders(output),
            # Im letzten Versuch übernimmt best_last_attempt die gestrichenen Räder
            *([] if final_try else [lambda: check_kept_bikes(output, ctx)]),
        )
        problems = global_problems + bike_problems(output, ctx)
        # Letzter Versuch: lieber eine kürzere, belegte Rangliste oder eine ehrliche Textantwort als „Exceeded
        # maximum output retries“
        if final_try:
            if (best := best_last_attempt(output, ctx, problems, global_problems)) is not None:
                if best is not output:
                    logger.warning("Letzter Versuch: %d Räder statt %d", len(best.ranking), len(output.ranking))
                return best
            logger.warning("Letzter Versuch: keine Rangliste zu retten, Textantwort statt Abbruch")
            return TextAnswer(message=NO_RANKING_MESSAGE)
        if not problems:
            return output
        raise ModelRetry("\n\n".join([*problems, KEEP_RESULTS_NOTE]))

    return agent


recommender_agent = build_agent()
