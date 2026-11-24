"""Verdichtet frühere Nachrichten, bevor der Berater sie an das Modell schickt.

Jede gelesene Seite bliebe sonst bis zum Ende der Beratung im Verlauf und würde bei jeder Folgenachricht erneut
gesendet. Bei 100.000 Tokens pro Minute passen dann nur noch zwei Anfragen pro Minute (Beratung 9: 30.000 bis
35.000 Tokens je Anfrage). Gekürzt werden nur Tool-Ergebnisse aus früheren Nachrichten und nur in der Anfrage an das
Modell: Gespeicherter Verlauf und Output-Validator (known_pages) sehen weiter den vollen Text.
"""

from dataclasses import replace
from typing import Any

from pydantic_ai import ModelMessage, ModelRequest, RunContext, ToolReturnPart
from pydantic_ai.capabilities import Hooks
from pydantic_ai.models import ModelRequestContext

from app.agent.deps import AgentDeps

EARLIER_PAGE_CHARS = 1200
EARLIER_SNIPPET_CHARS = 200
EARLIER_NOTE = "Aus einer früheren Nachricht, gekürzt; Volltext bei Bedarf mit read_product_page (ohne Kosten, Cache)."


def run_start(messages: list[ModelMessage]) -> int:
    """Index der letzten Nutzernachricht; davor liegen die früheren Antworten."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelRequest) and any(part.part_kind == "user-prompt" for part in message.parts):
            return index
    return 0


def compact_content(tool_name: str, content: Any) -> Any:
    if not isinstance(content, dict):
        return content
    if tool_name == "read_product_page" and isinstance(content.get("text"), str):
        text = content["text"]
        if len(text) <= EARLIER_PAGE_CHARS:
            return content
        return {**content, "text": text[:EARLIER_PAGE_CHARS], "truncated": True, "note": EARLIER_NOTE}
    if tool_name == "search_bikes" and isinstance(content.get("results"), list):
        results = [
            {**hit, "snippet": (hit.get("snippet") or "")[:EARLIER_SNIPPET_CHARS]} if isinstance(hit, dict) else hit
            for hit in content["results"]
        ]
        return {**content, "results": results}
    return content


def compact_earlier(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Kopie des Verlaufs mit gekürzten Seiten und Suchauszügen vor der letzten Nutzernachricht."""
    start = run_start(messages)
    compacted: list[ModelMessage] = []
    for index, message in enumerate(messages):
        if index >= start or not isinstance(message, ModelRequest):
            compacted.append(message)
            continue
        parts = [
            replace(part, content=compact_content(part.tool_name, part.content))
            if isinstance(part, ToolReturnPart)
            else part
            for part in message.parts
        ]
        compacted.append(replace(message, parts=parts))
    return compacted


async def compact_request(ctx: RunContext[AgentDeps], request_context: ModelRequestContext) -> ModelRequestContext:
    # Nur die Anfrage ändern, nicht ctx.messages (das täte ProcessHistory): Der Verlauf bleibt vollständig
    request_context.messages = compact_earlier(list(request_context.messages))
    return request_context


compact_earlier_messages = Hooks(before_model_request=compact_request)
