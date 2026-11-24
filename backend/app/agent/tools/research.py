import json
from typing import Any

from pydantic_ai import RunContext, ToolCallPart
from pydantic_ai.exceptions import UsageLimitExceeded

from app.agent.deps import AgentDeps
from app.agent.recommendation_schema import BikeCategory
from app.agent.research import research_agent
from app.agent.tools.common import tool_error
from app.rag.attribution import retrieved_documents


async def research_knowledge(
    ctx: RunContext[AgentDeps],
    category: BikeCategory,
    use_case: str,
    electric: bool | None = None,
    extra_topic: str | None = None,
) -> dict[str, Any]:
    """Lässt einen Recherche-Agenten die Wissensbasis (Kaufratgeber) durchsuchen und die Treffer zusammenfassen.

    Liefert Hinweise zu Kategorie, Ausstattung, Rahmengröße und Kauftipps sowie die Titel der gefundenen
    Dokumente (`documents`) für `sources`.

    Args:
        category: Gesuchte Fahrradkategorie.
        use_case: Einsatzzweck, z. B. "täglich 15 km pendeln, teils Schotter".
        electric: E-Bike gewünscht (true), ausgeschlossen (false) oder offen (null).
        extra_topic: Höchstens ein Zusatzthema, z. B. "Kindersitz" oder "Riemenantrieb".
    """
    topics = [f"Kategorie: {category.value}", f"Einsatz: {use_case}"]
    if electric is not None:
        topics.append("Antrieb: E-Bike" if electric else "Antrieb: ohne Motor")
    if extra_topic:
        topics.append(f"Zusatzthema: {extra_topic}")
    try:
        # Gemeinsame Nutzung: Die Anfragen des Recherche-Agenten zählen zum Limit des Berater-Laufs
        result = await research_agent.run("\n".join(topics), deps=ctx.deps, usage=ctx.usage)
    except UsageLimitExceeded:
        raise
    except Exception as exc:  # z. B. Modell nicht erreichbar oder keine gültige Zusammenfassung
        return tool_error(f"Recherche nicht möglich: {exc}. Ohne Wissensbasis beraten.")

    queries = [
        part.args_as_dict().get("query", "")
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "search_knowledge"
    ]
    documents = retrieved_documents(json.loads(result.all_messages_json()))
    if not documents:
        return tool_error(
            f"Die Wissensbasis enthält dazu nichts Passendes (gesucht: {', '.join(queries) or '–'}). "
            "Ohne diese Informationen beraten."
        )
    return result.output.model_dump() | {"queries": queries, "documents": documents}
