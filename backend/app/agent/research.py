"""Recherche-Agent: durchsucht die Wissensbasis im Auftrag des Beraters und fasst die Treffer zusammen.

Der Berater ruft ihn über das Tool `research_knowledge` auf ("Agent als Tool", app.agent.tools.research). Die
Suchen, Fundstellen und Umformulierungen bleiben im Verlauf dieses Agenten; der Berater bekommt nur die kurze
Zusammenfassung und hält so seinen Kontext klein.
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelResponse, RunContext, Tool, ToolCallPart, ToolDefinition, ToolOutput

from app.agent.deps import AgentDeps
from app.agent.models import build_model, model_settings
from app.agent.tools.knowledge import search_knowledge
from app.config import get_settings

# Danach steht nur noch die Zusammenfassung zur Wahl. Üblich sind 2-4 Suchen; ohne Grenze formulierte der Agent
# vereinzelt immer weiter um (ein Lauf: rund 50 Anfragen, 2,2 Mio. Eingabe-Tokens, 20 Minuten).
MAX_SEARCHES = 5

INSTRUCTIONS = """Du recherchierst für einen Fahrradberater in seiner Wissensbasis (Kaufratgeber) und antwortest auf Deutsch.

Stelle mit `search_knowledge` kurze Suchen zu je einem Thema, alle in derselben Runde:
1. Die Fahrradkategorie und ihr Einsatz, z. B. "Trekkingrad zum Pendeln" oder "Gravelbike Einsatzbereich".
2. Worauf man bei Ausstattung achten sollte, z. B. "E-Bike Motor und Akku" oder "Schaltung und Bremsen".
3. Falls angegeben: Rahmengröße bzw. Geometrie und das Zusatzthema (z. B. "Lastenrad mit Kindern").
Findet eine Suche nichts Passendes, darfst Du sie einmal anders formulieren. Mehr als 5 Suchen sind nicht nötig.

Fasse danach in kurzen Stichpunkten zusammen, was die Fundstellen für die Auswahl hergeben. Übernimm nur, was in den
Fundstellen steht, und erfinde nichts. Gibt die Wissensbasis zu einem Punkt nichts her, bleibt die Liste leer.
"""


class ResearchBrief(BaseModel):
    category_notes: list[str] = Field(description="Hinweise zu Kategorie und Einsatzbereich")
    component_notes: list[str] = Field(description="Worauf es bei Antrieb, Motor/Akku, Bremsen, Federung ankommt")
    fit_notes: list[str] = Field(description="Hinweise zu Rahmengröße, Geometrie und Sitzposition")
    buying_tips: list[str] = Field(description="Kauftipps, z. B. Budget, Probefahrt, Kaufzeitpunkt")


def searches_so_far(ctx: RunContext[AgentDeps]) -> int:
    return sum(
        isinstance(part, ToolCallPart) and part.tool_name == "search_knowledge"
        for message in ctx.messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


async def until_search_limit(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Blendet `search_knowledge` nach MAX_SEARCHES Suchen aus; das Modell muss dann zusammenfassen."""
    return tool_def if searches_so_far(ctx) < MAX_SEARCHES else None


def build_research_agent() -> Agent[AgentDeps, ResearchBrief]:
    settings = get_settings()
    return Agent(
        build_model(settings.research_model, settings.llm_requests_per_minute, settings.llm_tokens_per_minute),
        name="knowledge-researcher",  # Name des Agent-Spans im Tracing, unterhalb des Beraters
        deps_type=AgentDeps,
        output_type=ToolOutput(
            ResearchBrief, name="research_brief", description="Zusammenfassung der Recherche ausgeben", strict=True
        ),
        model_settings=model_settings(settings.research_model),
        instructions=INSTRUCTIONS,
        tools=[Tool(search_knowledge, prepare=until_search_limit)],
        retries=2,
        defer_model_check=True,
    )


research_agent = build_research_agent()
