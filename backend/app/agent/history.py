"""Auswertung des gespeicherten Berater-Verlaufs (PydanticAI-JSON), gemeinsam für Ausgabeprüfungen und Intake.

Beide müssen dasselbe entscheiden: Hält das Intake eine Recherche für erledigt, die der Berater nicht gelten lässt,
kostet das eine zusätzliche Berater-Runde.
"""

import json
from collections.abc import Iterator
from typing import Any

RESEARCH_TOOLS = {"research_knowledge", "search_knowledge"}
OUTPUT_TOOLS = {"final_recommendation", "ask_user", "answer_user"}


def tool_args(part: dict[str, Any]) -> dict[str, Any]:
    """Argumente eines Tool-Aufrufs; im Verlauf als Objekt oder (bei manchen Modellen) als JSON-Text."""
    args = part.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {}
    return args if isinstance(args, dict) else {}


def runs(history_json: list[Any]) -> Iterator[list[dict[str, Any]]]:
    """Teilt den Verlauf in Läufe: je eine Nutzernachricht mit den folgenden Nachrichten bis zur nächsten."""
    run: list[dict[str, Any]] = []
    for message in history_json:
        starts_run = message.get("kind") == "request" and any(
            part.get("part_kind") == "user-prompt" for part in message.get("parts", [])
        )
        if starts_run and run:
            yield run
            run = []
        run.append(message)
    if run:
        yield run


def tool_calls(messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for message in messages:
        if message.get("kind") == "response":
            yield from (part for part in message.get("parts", []) if part.get("part_kind") == "tool-call")


def researched_categories(history_json: list[Any]) -> set[str | None]:
    """Kategorien, für die in der Beratung recherchiert wurde; None für eine Recherche ohne Kategorie
    (search_knowledge direkt), die für jede gilt."""
    categories: set[str | None] = set()
    for call in tool_calls(history_json):
        if call.get("tool_name") in RESEARCH_TOOLS:
            category = tool_args(call).get("category")
            categories.add(category if isinstance(category, str) else None)
    return categories


def searched_web(history_json: list[Any]) -> bool:
    """Hat die Beratung schon im Web gesucht (Berater oder Vorab-Aufruf)?"""
    return any(call.get("tool_name") == "search_bikes" for call in tool_calls(history_json))


def researched_for(history_json: list[Any], category: str) -> bool:
    categories = researched_categories(history_json)
    return None in categories or category in categories
