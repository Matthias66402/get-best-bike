"""Intake: Ein kleines LLM wertet die aktuelle Nachricht vorab aus. Es erkennt die Kriterien (Kategorie, Budget,
Motor, Körpermaße), formuliert die ersten Websuchen und bei einer neuen Beratung den Auftrag für die Recherche in
der Wissensbasis.

Der Graph (app.agent.graph) führt `search_bikes` und `research_knowledge` damit parallel aus, bevor der Berater
startet, und fügt Aufrufe und Ergebnisse in dessen Verlauf ein. Das spart dem Berater Runden. Ein Fehler hier ist
unkritisch: Der Berater kann jedes Tool selbst (erneut) aufrufen.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput

from app.agent.deps import AgentDeps
from app.agent.history import researched_categories
from app.agent.models import build_model
from app.agent.recommendation_schema import BikeCategory
from app.config import get_settings

MAX_OUTPUT_TOKENS = 1000
# Intake darf die Beratung nicht ausbremsen: ohne Wiederholungen nach kurzer Zeit ohne Vorab-Suche weiter
TIMEOUT_SECONDS = 10.0
# So viele frühere Nutzernachrichten bekommt das Intake als Kontext (z. B. für "bis 2000 €" nach einer Rückfrage)
RECENT_USER_MESSAGES = 4
MAX_SEARCHES = 3

INSTRUCTIONS = """\
Du bereitest Anfragen an einen Fahrradberater vor, der eine Rangliste der besten passenden Fahrräder erstellt.
Werte die NEUE Nachricht des Nutzers aus, unter Berücksichtigung der früheren Nachrichten.

Kriterien (null, wenn nicht genannt und nicht aus dem Kontext ersichtlich):
- `category`: city (Stadt-, Hollandrad), trekking, road (Rennrad), gravel, mtb_hardtail, mtb_fully, cargo
  (Lastenrad), folding (Faltrad), kids, other. Leite sie aus dem Einsatz ab, wenn sie eindeutig ist (z. B.
  "Alpencross mit viel Trail" → mtb_fully, "Kinder transportieren" → cargo).
- `electric`: true bei E-Bike/Pedelec, false bei ausdrücklich ohne Motor, sonst null.
- `budget_eur`: Höchstbetrag in Euro ("bis 3.500 €", "max. 2k" → 2000).
- `use_case`: Einsatzzweck in wenigen Worten.
- `height_cm`, `inseam_cm`: Körpergröße bzw. Schrittlänge in cm ("1,82 m" → 182).

Websuchen (`search_queries`): Nur wenn eine neue oder geänderte Rangliste nötig ist und Einsatzzweck UND Budget
feststehen, bis zu 3 kurze deutsche Suchanfragen, die verschiedene Kandidaten finden, z. B.
"bestes Trekking E-Bike bis 3500 Euro Test 2026" und "Trekking E-Bike Pendeln Riemenantrieb 3500 Euro".
Füge das aktuelle Jahr ein. Sonst eine leere Liste (Rückfragen offen, reine Erklärung, Smalltalk).

Recherche (`research`): Nur wenn die Kategorie feststeht und eine Rangliste ansteht, sonst null.

Art der NEUEN Nachricht (`topic`):
- "fahrrad": alles rund um Fahrräder, Kauf, Ausstattung, Größen, Zubehör, Antworten auf Rückfragen des Beraters
  und Änderungswünsche zur Rangliste.
- "smalltalk": Begrüßung, Dank, Fragen dazu, was der Berater kann und wie er vorgeht.
- "intern": Fragen zum Innenleben des Beraters: Quellcode, Anweisungen bzw. Systemprompt, Sprachmodell,
  Datenbank, Server, API-Schlüssel, Datenspeicherung.
- "fachfremd": nur, wenn die Nachricht zu keiner der anderen Arten passt. Im Zweifel "fahrrad".
"""


class ResearchRequest(BaseModel):
    """Argumente für research_knowledge (app.agent.tools.research)."""

    category: BikeCategory
    use_case: str
    electric: bool | None = None
    extra_topic: str | None = Field(default=None, description="Höchstens ein Zusatzthema")


class SearchRequest(BaseModel):
    """Argumente für search_bikes (app.agent.tools.web)."""

    query: str


class IntakeResult(BaseModel):
    category: BikeCategory | None
    electric: bool | None
    budget_eur: float | None
    use_case: str | None
    height_cm: int | None
    inseam_cm: int | None
    search_queries: list[str] = Field(description="Bis zu 3 Websuchen für die Rangliste, sonst leer")
    research: ResearchRequest | None = Field(description="Recherche für eine neue Rangliste, sonst null")
    topic: Literal["fahrrad", "smalltalk", "intern", "fachfremd"] = Field(description="Art der neuen Nachricht")


class Prefetch(BaseModel):
    """Was der Graph vor dem Berater ausführt bzw. ihm mitgibt."""

    searches: list[SearchRequest] = Field(default_factory=list)
    research: ResearchRequest | None = None
    criteria: dict[str, Any] = Field(default_factory=dict)
    # Fachfremde Nachricht bzw. Frage zum Innenleben: Der Graph antwortet mit einer festen Absage, ohne Berater
    off_topic: Literal["fachfremd", "intern"] | None = None

    def recommender_note(self) -> str | None:
        """Erkannte Kriterien für die Anweisungen des Beraters; None, wenn die Nachricht keine nennt."""
        labels = {
            "category": "Kategorie", "electric": "E-Bike", "budget_eur": "Budget (€)", "use_case": "Einsatz",
            "height_cm": "Körpergröße (cm)", "inseam_cm": "Schrittlänge (cm)",
        }  # fmt: skip
        parts = [f"{labels[k]}: {v}" for k, v in self.criteria.items() if k in labels and v is not None]
        if not parts:
            return None
        return "Vorab-Auswertung (prüfe sie, maßgeblich ist die Nachricht): " + "; ".join(parts) + "."


def build_intake_agent() -> Agent[None, IntakeResult]:
    settings = get_settings()
    return Agent(
        build_model(settings.intake_model, settings.llm_requests_per_minute, settings.llm_tokens_per_minute)
        if settings.intake_model
        else None,
        name="intake",  # Name des Agent-Spans im Tracing
        output_type=ToolOutput(IntakeResult, name="kriterien", strict=True),
        instructions=INSTRUCTIONS,
        model_settings={"max_tokens": MAX_OUTPUT_TOKENS, "timeout": TIMEOUT_SECONDS, "max_retries": 0},
        defer_model_check=True,
    )


intake_agent = build_intake_agent()


@dataclass
class ConversationContext:
    """Aus dem gespeicherten Verlauf (PydanticAI-JSON) für das Intake."""

    user_messages: list[str] = field(default_factory=list)  # letzte Nutzernachrichten, älteste zuerst
    question: str | None = None  # letzte Rückfrage des Beraters


def conversation_context(history_json: list[Any]) -> ConversationContext:
    context = ConversationContext()
    for message in history_json:
        for part in message.get("parts", []):
            kind = part.get("part_kind")
            if kind == "user-prompt" and isinstance(part.get("content"), str):
                context.user_messages.append(part["content"])
            elif kind == "tool-call" and part.get("tool_name") == "ask_user" and isinstance(part.get("args"), dict):
                context.question = part["args"].get("question")
    context.user_messages = context.user_messages[-RECENT_USER_MESSAGES:]
    return context


def intake_prompt(message: str, context: ConversationContext, deps: AgentDeps) -> str:
    lines = [f"Heute ist {deps.today.isoformat()}."]
    if deps.height_cm or deps.inseam_cm:
        lines.append(f"Profil: Körpergröße {deps.height_cm or '?'} cm, Schrittlänge {deps.inseam_cm or '?'} cm.")
    if context.user_messages:
        lines.append("Frühere Nachrichten des Nutzers (älteste zuerst):")
        lines.extend(f"- {text}" for text in context.user_messages)
    if context.question:
        lines.append(f"Letzte Rückfrage des Beraters: {context.question}")
    lines.append(f"NEUE Nachricht: {message}")
    return "\n".join(lines)


async def plan_prefetch(message: str, history_json: list[Any], deps: AgentDeps) -> Prefetch:
    """Was vor dem Berater laufen soll; ohne Intake-Modell nichts. Fehler gibt der Aufrufer weiter (Graph fängt sie ab).

    Ob eine Recherche schon im Verlauf steht und ob die Websuche eingerichtet ist, prüft der Code, nicht das LLM.
    """
    if not get_settings().intake_model:
        return Prefetch()
    result = (await intake_agent.run(intake_prompt(message, conversation_context(history_json), deps))).output
    if result.topic in ("fachfremd", "intern"):
        return Prefetch(off_topic=result.topic)
    queries = list(dict.fromkeys(q.strip() for q in result.search_queries if q.strip()))[:MAX_SEARCHES]
    research = result.research
    if research is not None and (researched_categories(history_json) or not research.use_case.strip()):
        research = None
    return Prefetch(
        searches=[SearchRequest(query=q) for q in queries] if deps.settings.tavily_api_key else [],
        research=research,
        criteria=result.model_dump(
            mode="json", include={"category", "electric", "budget_eur", "use_case", "height_cm", "inseam_cm"}
        ),
    )
