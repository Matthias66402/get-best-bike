"""Gemeinsame Hilfen für Agent- und Chat-Tests: geskriptetes Modell statt LLM, Tavily-Antworten, SSE-Parser."""

import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic_ai import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

Step = tuple[str, dict[str, Any]] | list[tuple[str, dict[str, Any]]] | str

TAVILY_URL = "https://api.tavily.com"
CUBE_URL = "https://www.example-shop.de/cube-kathmandu-hybrid-pro-750"
KTM_URL = "https://www.example-shop.de/ktm-macina-style-xl"

SEARCH_OK = {
    "query": "Trekking E-Bike bis 3500 Euro",
    "results": [
        {
            "title": "Cube Kathmandu Hybrid Pro 750 Trekking E-Bike",
            "url": CUBE_URL,
            "content": "Cube Kathmandu Hybrid Pro 750, Bosch Performance Line CX, 750 Wh, jetzt nur 3.299,00 € statt 3.599 €.",
            "score": 0.9,
        },
        {
            "title": "KTM Macina Style XL im Test",
            "url": KTM_URL,
            "content": "Das KTM Macina Style XL mit Bosch-Motor und Riemenantrieb kostet 3.499 Euro.",
            "score": 0.8,
        },
    ],
}
EXTRACT_OK = {
    "results": [
        {
            "url": CUBE_URL,
            "raw_content": "Cube Kathmandu Hybrid Pro 750 Modell 2026. Preis: 3.299,00 €. Gewicht 26,4 kg. "
            "Größentabelle: S 160-170 cm, M 168-180 cm, L 178-190 cm.",
        }
    ],
    "failed_results": [],
}


def bike(rank: int, url: str = CUBE_URL, price: float | None = 3299.0, **changes: Any) -> dict[str, Any]:
    data = {
        "rank": rank,
        "brand": "Cube",
        "model": f"Kathmandu Hybrid Pro 750 Variante {rank}",
        "model_year": 2026,
        "category": "trekking",
        "price_eur": price,
        "price_source_url": url,
        "key_specs": {
            "weight_kg": 26.4,
            "drivetrain": "Shimano Deore 1x12",
            "motor": "Bosch Performance Line CX",
            "battery_wh": 750,
            "suspension": "Federgabel 63 mm",
            "brakes": "hydraulische Scheibenbremsen",
        },
        "frame_size": "M",
        "match_score": 88,
        "why_fits": "Kräftiger Motor und großer Akku für tägliches Pendeln.",
        "pros": ["starker Motor"],
        "cons": ["schwer"],
    }
    return data | changes


def recommendation(*bikes: dict[str, Any], budget: float | None = 3500, **requirements: Any) -> dict[str, Any]:
    return {
        "title": "Trekking-E-Bike bis 3.500 €",
        "summary": "Zum Pendeln mit Motor.",
        "requirements": {
            "use_case": "täglich 15 km pendeln",
            "category": "trekking",
            "electric": True,
            "budget_eur": budget,
            "height_cm": 175,
            "inseam_cm": 82,
            "other_wishes": [],
        }
        | requirements,
        "ranking": list(bikes) or [bike(1)],
        "buying_tips": ["Probefahrt machen"],
        "sources": ["LLM-Quelle, wird vom Backend ersetzt"],
    }


def streaming_model(respond: Callable[[list[ModelMessage]], Step]) -> FunctionModel:
    """FunctionModel für den Streaming-Pfad: `respond` liefert (tool_name, args), eine Liste davon oder Text."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        step = respond(messages)
        if isinstance(step, str):
            yield step
            return
        for index, (name, args) in enumerate(step if isinstance(step, list) else [step]):
            yield {
                index: DeltaToolCall(
                    name=name, json_args=json.dumps(args), tool_call_id=f"call_{uuid.uuid4().hex[:12]}"
                )
            }

    return FunctionModel(stream_function=stream)


def scripted_model(*steps: Step) -> FunctionModel:
    """Jeder Modellaufruf liefert den nächsten Schritt."""
    calls = iter(steps)
    return streaming_model(lambda _: next(calls))


def parse_sse(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for block in body.strip().split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]
