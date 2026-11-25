"""Fahrradberater: PydanticAI-Agent, der aus Websuche, Wissensbasis und Rahmengrößen-Tool eine Rangliste der 1 bis 5
am besten passenden Fahrräder erstellt (oder nachfragt bzw. ohne Rangliste antwortet).

Der Output-Validator `check_plausibility` lässt nur belegte Empfehlungen durch: Quell-URLs und Preise müssen aus den
Ergebnissen von `search_bikes` bzw. `read_product_page` stammen, das Budget wird eingehalten und bei bekannten
Körpermaßen gibt es eine Rahmengröße aus `recommend_frame_size`.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
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

from app.agent.compaction import compact_earlier_messages
from app.agent.deps import AgentDeps
from app.agent.history import researched_categories
from app.agent.models import build_model, model_settings
from app.agent.recommendation_schema import (
    MAX_RANKING,
    BikeCategory,
    BikeRecommendation,
    ClarificationQuestion,
    RankedBike,
    TextAnswer,
)
from app.agent.tools.frame_size import NO_MEASUREMENTS, recommend_frame_size
from app.agent.tools.prices import price_in_text
from app.agent.tools.research import research_knowledge
from app.agent.tools.web import normalize_url, read_product_page, search_bikes
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


def words(text: str) -> list[str]:
    """Kleingeschriebene Wörter ohne Satz- und Sonderzeichen ("E‑Power Easy" → ["e", "power", "easy"])."""
    return re.findall(r"[0-9a-zäöüß]+", text.casefold())


def model_family(model: str) -> str | None:
    """Erstes aussagekräftiges Wort des Modellnamens, z. B. "talon" in "Talon E+ EX Gen2"."""
    return next((w for w in words(model) if len(w) >= 3 and not w.isdigit()), None)


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
   Kategorie im Budget und nach konkreten Modellen. Nur Modelle aus den Ergebnissen, nichts aus dem Gedächtnis.
4. Belegen mit `read_product_page` für die aussichtsreichsten Kandidaten (alle in derselben Runde): Preis,
   Ausstattung, Gewicht und, falls vorhanden, die Größentabelle.
5. Rahmengröße mit `recommend_frame_size` je Rad (mit Größentabelle der Produktseite, falls vorhanden), alle in
   derselben Runde, sobald Körpermaße bekannt sind (Nachricht oder Profil).
6. Ausgeben mit `final_recommendation`, beste Wahl auf Platz 1, mit 5 Rädern.
   Plätze 1 bis {PRICED_RANKS} brauchen einen Preis von einer Angebots- oder Produktseite; Testartikel allein
   reichen dafür oft nicht. `model` genau so, wie es auf der Quellseite steht.

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


def calls_in_run(messages: list[ModelMessage], tool_name: str) -> int:
    return sum(
        isinstance(part, ToolCallPart) and part.tool_name == tool_name
        for message in current_run(messages)
        for part in getattr(message, "parts", [])
    )


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
    if known := determined_sizes(ctx.messages):
        # Beratung 24: nach jeder Zurückweisung derselbe Aufruf (Kinderrad, 126 cm), je eine Modellrunde
        note = "Schon ermittelt (gilt weiter, nicht erneut aufrufen): " + "; ".join(known) + "."
        return replace(tool_def, description=f"{tool_def.description}\n\n{note}")
    return tool_def


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


async def research_once(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Blendet `research_knowledge` aus, sobald in dieser Beratung recherchiert wurde (auch vorab durch den Graphen).

    Beratung 8: viermal recherchiert, einmal sogar "find product pages with prices"; jede Recherche ist ein eigener
    Agent mit mehreren Modellaufrufen und kostete bei knappem Anfragelimit jeweils Minuten.
    """
    history_json = ModelMessagesTypeAdapter.dump_python(ctx.messages, mode="json")
    return None if researched_categories(history_json) else tool_def


def output_rejected(messages: list[ModelMessage], tool_name: str) -> bool:
    """Wurde in dieser Antwort schon eine Ausgabe über `tool_name` zurückgewiesen?"""
    return any(
        getattr(part, "tool_name", None) == tool_name
        and (isinstance(part, RetryPromptPart) or (isinstance(part, ToolReturnPart) and part.outcome == "retried"))
        for message in current_run(messages)
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def check_sources(output: BikeRecommendation, pages: dict[str, str], web_budget: bool = True) -> None:
    """`web_budget=False`, wenn Websuchen und Seitenabrufe dieser Antwort aufgebraucht sind: Dann ließe sich nichts
    mehr nachsuchen. Fehlende Preise gelten, und für unbelegte Räder heißt die Rückmeldung „streichen“ statt
    „suchen“ (Beratung 22: Das Modell wollte das Radon per Suche ersetzen, fand kein Such-Tool mehr und lief in
    eine Schleife)."""
    if not pages:
        raise ModelRetry(
            "Die Rangliste braucht Belege aus der Websuche. Suche mit `search_bikes` nach passenden Modellen; "
            "funktioniert die Websuche nicht, antworte mit `answer_user`."
        )
    problems = []
    for bike in output.ranking:
        name = f"Platz {bike.rank} ({bike.brand} {bike.model})"
        text = pages.get(normalize_url(bike.price_source_url))
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
        if others := category_conflict(bike.category, context):
            problems.append(
                f"{name}: Die Quelle beschreibt das Rad als {', '.join(others)}, nicht als {bike.category.value}. "
                "Setze die richtige Kategorie und prüfe, ob das Rad dann noch zum Wunsch passt; sonst ersetze es."
            )
        if bike.price_eur is not None and not price_in_text(bike.price_eur, context):
            problems.append(
                f"{name}: Der Preis {bike.price_eur:g} € steht auf {bike.price_source_url} nicht beim Modell. "
                "Übernimm den Preis von der Seite (`prices_eur`), lies die Produktseite mit `read_product_page` "
                "oder setze null."
            )
        if bike.price_eur is None and bike.rank <= PRICED_RANKS and web_budget:
            problems.append(
                f"{name}: Für Platz 1 bis {PRICED_RANKS} braucht es einen belegten Preis. Suche das Modell mit "
                "`search_bikes` (Modellname + Preis) und lies ein Angebot mit `read_product_page`; findest Du "
                "keinen Preis, ersetze das Rad oder setze es weiter nach hinten."
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


def check_electric(output: BikeRecommendation) -> None:
    motorized = [b for b in output.ranking if b.key_specs.motor or b.key_specs.battery_wh]
    if output.requirements.electric is False and motorized:
        names = ", ".join(f"Platz {b.rank}" for b in motorized)
        raise ModelRetry(f"Der Nutzer möchte ein Rad ohne Motor, {names} hat einen Motor. Ersetze es.")
    if output.requirements.electric is True and (unknown := [b for b in output.ranking if b not in motorized]):
        names = ", ".join(f"Platz {b.rank}" for b in unknown)
        raise ModelRetry(
            f"Der Nutzer möchte ein E-Bike, aber für {names} fehlen Motor und Akku. Lies die Produktseite mit "
            "`read_product_page` und trage Motor bzw. Akku (Wh) ein, oder ersetze das Rad."
        )


def check_ranking_size(output: BikeRecommendation, ctx: RunContext[AgentDeps]) -> None:
    """Die Ergebnisseite zeigt MAX_RANKING Räder. Weniger nur, wenn das Web-Budget aufgebraucht ist oder schon
    MIN_RANKING_RETRIES-mal zurückgewiesen wurde, damit die Antwort nicht an der Wiederholungsgrenze scheitert."""
    missing = MAX_RANKING - len(output.ranking)
    if missing <= 0 or ctx.retry >= MIN_RANKING_RETRIES or not web_budget_left(ctx.messages):
        return
    raise ModelRetry(
        f"Die Rangliste soll {MAX_RANKING} Räder enthalten, es fehlen {missing}. Ergänze passende Kandidaten aus "
        f"der Websuche; Plätze {PRICED_RANKS + 1} bis {MAX_RANKING} dürfen ohne Preis sein (price_eur = null)."
    )


def check_question_needed(output: ClarificationQuestion, ctx: RunContext[AgentDeps]) -> None:
    """Rückfrage nur, wenn Einsatzzweck oder Budget fehlen (laut Intake). Wurde schon eine Rückfrage zurückgewiesen
    und stellt das Modell sie trotzdem, gilt sie. Alle-Fälle-Lauf: sieben Fragen zum E-Lastenrad, obwohl Einsatz,
    Budget und Motor in der Anfrage standen. Zählte jede frühere Zurückweisung (auch einer Rangliste), wich das
    Modell auf eine Rückfrage aus (Beratung 8: "Ist Carbon ein Muss?" nach einer abgelehnten Rangliste)."""
    criteria = ctx.deps.criteria
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


def bike_problems(output: BikeRecommendation, ctx: RunContext[AgentDeps]) -> list[str]:
    """Prüfungen, die einzelne Räder betreffen; ein betroffenes Rad lässt sich notfalls streichen."""
    return collect_problems(
        lambda: check_sources(output, known_pages(ctx.messages), web_budget=web_budget_left(ctx.messages)),
        lambda: check_budget(output),
        lambda: check_spec_notes(output),
        lambda: check_electric(output),
        lambda: check_frame_sizes(output, ctx.deps, ctx.messages),
    )


def salvage_ranking(output: BikeRecommendation, ctx: RunContext[AgentDeps]) -> BikeRecommendation | None:
    """Behält die Räder, die alle Prüfungen bestehen (geprüft auf dem Rang, den sie dann bekommen), und rückt sie auf;
    None, wenn keines übrig bleibt."""
    kept: list[RankedBike] = []
    for bike in output.ranking:
        candidate = bike.model_copy(update={"rank": len(kept) + 1})
        if not bike_problems(output.model_copy(update={"ranking": [candidate]}), ctx):
            kept.append(candidate)
    if not kept:
        return None
    dropped = len(output.ranking) - len(kept)
    tip = f"{dropped} weitere Kandidaten ließen sich nicht ausreichend belegen und fehlen deshalb in der Liste."
    return output.model_copy(update={"ranking": kept, "buying_tips": [*output.buying_tips, tip]})


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
        capabilities=[compact_earlier_messages],
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
        global_problems = collect_problems(
            lambda: check_researched(ctx.messages), lambda: check_ranking_size(output, ctx)
        )
        problems = global_problems + bike_problems(output, ctx)
        if not problems:
            return output
        # Letzter Versuch: lieber eine kürzere, belegte Rangliste als „Exceeded maximum output retries“
        last_attempt = ctx.retry >= ctx.max_retries and not global_problems
        if last_attempt and (salvaged := salvage_ranking(output, ctx)) is not None:
            logger.warning(
                "Letzter Versuch: %d von %d Rädern gestrichen", len(output.ranking) - len(salvaged.ranking),
                len(output.ranking),
            )  # fmt: skip
            return salvaged
        raise ModelRetry("\n\n".join([*problems, KEEP_RESULTS_NOTE]))

    return agent


recommender_agent = build_agent()
