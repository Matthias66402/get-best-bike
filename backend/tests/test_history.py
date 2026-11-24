from typing import Any

from app.agent.history import researched_categories, researched_for, tool_args


def user(text: str) -> dict[str, Any]:
    return {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": text}]}


def calls(*tools: tuple[str, Any]) -> dict[str, Any]:
    return {
        "kind": "response",
        "parts": [
            {"part_kind": "tool-call", "tool_name": name, "args": args, "tool_call_id": f"c-{name}-{i}"}
            for i, (name, args) in enumerate(tools)
        ],
    }


def test_research_counts_per_category() -> None:
    history = [
        user("Trekking-E-Bike bis 3500 €"),
        calls(("research_knowledge", '{"category": "trekking", "use_case": "Pendeln"}')),  # Argumente als JSON-Text
        calls(("final_recommendation", {})),
    ]
    assert researched_categories(history) == {"trekking"}
    assert researched_for(history, "trekking") and not researched_for(history, "gravel")


def test_direct_search_counts_for_any_category() -> None:
    history = [user("Was ist ein Gravelbike?"), calls(("search_knowledge", {"query": "Gravelbike"}))]
    assert researched_categories(history) == {None} and researched_for(history, "cargo")


def test_tool_args_tolerates_broken_json() -> None:
    assert tool_args({"args": "{kaputt"}) == {}
    assert tool_args({"args": None}) == {}
