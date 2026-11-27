"""Berater, Output-Validator und Chat-Endpunkt mit geskriptetem Modell statt echtem LLM; Tavily per respx."""

import json
import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from pydantic import ValidationError
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RetryPromptPart,
    ToolCallPart,
    ToolDefinition,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from app.agent.compaction import EARLIER_PAGE_CHARS
from app.agent.deps import AgentDeps
from app.agent.models import REQUEST_TIMEOUT_SECONDS
from app.agent.recommendation_schema import (
    MAX_RANKING,
    BikeCategory,
    BikeRecommendation,
    TextAnswer,
    clean_name,
)
from app.agent.recommender import (
    FRAME_SIZE_CALLS,
    FREE_REPEATED_READS,
    FREE_UNKNOWN_URLS,
    INSTRUCTIONS,
    PRICED_RANKS,
    TOOL_BUDGETS,
    calls_in_run,
    category_conflict,
    check_placeholders,
    check_ranking_size,
    check_sources,
    check_spec_notes,
    check_specific_models,
    fix_source_urls,
    frame_size_budget,
    known_pages,
    match_url,
    model_on_page,
    ranking_only,
    recommender_agent,
)
from app.agent.tools.frame_size import NO_MEASUREMENTS
from app.agent.tools.web import focus_text, normalize_url
from app.config import get_settings
from app.db import SessionLocal
from tests.conftest import auth_headers, create_user
from tests.helpers import (
    CUBE_URL,
    EXTRACT_OK,
    KTM_URL,
    SEARCH_OK,
    TAVILY_URL,
    bike,
    parse_sse,
    recommendation,
    scripted_model,
)

SEARCH = ("search_bikes", {"query": "Trekking E-Bike bis 3500 Euro"})
RESEARCH = ("research_knowledge", {"category": "trekking", "use_case": "Pendeln", "electric": True})
FRAME = ("recommend_frame_size", {"category": "trekking", "height_cm": 175, "inseam_cm": 82})


@pytest.fixture(autouse=True)
def tavily(monkeypatch: pytest.MonkeyPatch) -> respx.MockRouter:
    monkeypatch.setattr(get_settings(), "tavily_api_key", "tvly-test")
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{TAVILY_URL}/search").respond(json=SEARCH_OK)
        router.post(f"{TAVILY_URL}/extract").respond(json=EXTRACT_OK)
        yield router


async def create_consultation(client: httpx.AsyncClient, headers: dict[str, str]) -> int:
    response = await client.post("/api/consultations", headers=headers, json={})
    assert response.status_code == 201
    return response.json()["id"]


async def chat(client: httpx.AsyncClient, headers: dict[str, str], cid: int, message: str) -> list[dict[str, Any]]:
    response = await client.post(f"/api/chat/{cid}", headers=headers, json={"message": message})
    assert response.status_code == 200
    return parse_sse(response.text)


def retry_prompts(messages: list[Any], tool_name: str = "final_recommendation") -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if getattr(part, "tool_name", None) == tool_name
        and (isinstance(part, RetryPromptPart) or (isinstance(part, ToolReturnPart) and part.outcome == "retried"))
    ]


async def run(*steps: Any) -> Any:
    with recommender_agent.override(model=scripted_model(*steps)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
            async with recommender_agent.run_stream_events("Trekking-E-Bike bis 3500 €", deps=deps) as events:
                async for event in events:
                    if hasattr(event, "result"):
                        return event.result
    raise AssertionError("kein Ergebnis")


# --- Schema ---


def test_ranks_must_be_consecutive_and_unique() -> None:
    BikeRecommendation.model_validate(recommendation(bike(2), bike(1)))  # Reihenfolge egal
    with pytest.raises(ValidationError, match="lückenlos"):
        BikeRecommendation.model_validate(recommendation(bike(1), bike(3)))
    with pytest.raises(ValidationError, match="nur einmal"):
        BikeRecommendation.model_validate(recommendation(bike(1), bike(2, model=bike(1)["model"])))
    with pytest.raises(ValidationError):
        BikeRecommendation.model_validate(recommendation(*(bike(i) for i in range(1, 7))))


def test_agent_has_request_timeout() -> None:
    assert recommender_agent.model_settings and recommender_agent.model_settings["timeout"] == REQUEST_TIMEOUT_SECONDS  # type: ignore[index]


# --- Output-Validator ---


async def test_valid_recommendation_passes() -> None:
    result = await run(SEARCH, RESEARCH, FRAME, ("final_recommendation", recommendation()))
    assert isinstance(result.output, BikeRecommendation)
    assert not retry_prompts(result.all_messages())


async def test_requires_research_before_ranking() -> None:
    result = await run(
        SEARCH, FRAME, ("final_recommendation", recommendation()), RESEARCH, ("final_recommendation", recommendation())
    )
    assert "research_knowledge" in retry_prompts(result.all_messages())[0]


async def test_rejects_unknown_url_and_unbelegten_price() -> None:
    wrong = recommendation(
        bike(1, url="https://erfunden.example/rad"),
        bike(2, url=KTM_URL, brand="KTM", model="Macina Style XL", price=2999),
    )
    fixed = recommendation(bike(1), bike(2, url=KTM_URL, brand="KTM", model="Macina Style XL", price=3499))
    result = await run(SEARCH, RESEARCH, FRAME, ("final_recommendation", wrong), ("final_recommendation", fixed))
    feedback = retry_prompts(result.all_messages())[0]
    assert (
        "stammt nicht aus `search_bikes`" in feedback
        and "2999 € steht auf" in feedback
        and "nicht beim Modell" in feedback
    )
    assert [b.price_eur for b in result.output.ranking] == [3299, 3499]


async def test_rejects_brand_not_on_page() -> None:
    wrong = recommendation(bike(1, brand="Riese & Müller"))
    result = await run(
        SEARCH, RESEARCH, FRAME, ("final_recommendation", wrong), ("final_recommendation", recommendation())
    )
    assert "Marke „Riese & Müller“" in retry_prompts(result.all_messages())[0]


async def test_rejects_price_far_over_budget() -> None:
    over = recommendation(bike(1), budget=2800)  # 3.299 € > 2.800 € + 10 %
    ok = recommendation(bike(1), budget=3000)  # 3.299 € ≤ 3.300 €
    result = await run(SEARCH, RESEARCH, FRAME, ("final_recommendation", over), ("final_recommendation", ok))
    assert "Budget 2800 €" in retry_prompts(result.all_messages())[0]


async def test_rejects_motor_when_unwanted() -> None:
    wrong = recommendation(electric=False)
    ok = recommendation(bike(1, key_specs={**bike(1)["key_specs"], "motor": None, "battery_wh": None}), electric=False)
    result = await run(SEARCH, RESEARCH, FRAME, ("final_recommendation", wrong), ("final_recommendation", ok))
    assert "ohne Motor" in retry_prompts(result.all_messages())[0]


async def test_rejects_ebike_without_motor_data() -> None:
    no_motor = bike(1, key_specs={**bike(1)["key_specs"], "motor": None, "battery_wh": None})
    result = await run(
        SEARCH,
        RESEARCH,
        FRAME,
        ("final_recommendation", recommendation(no_motor)),
        ("final_recommendation", recommendation()),
    )
    assert "fehlen Motor und Akku" in retry_prompts(result.all_messages())[0]


async def test_rejects_homepage_as_source(tavily: respx.MockRouter) -> None:
    home = "https://www.cube.eu/"
    # Die Startseite kommt als Treffer der Suche (Beratung 29: vanrysel.com), nur dann darf sie gelesen werden
    homepage_hit = {"title": "Cube Bikes", "url": home, "content": "Cube Kathmandu Hybrid Pro 750", "score": 0.5}
    tavily.routes[0].respond(json={**SEARCH_OK, "results": [*SEARCH_OK["results"], homepage_hit]})
    steps = (SEARCH, ("read_product_page", {"url": home}), RESEARCH, FRAME)
    result = await run(
        *steps,
        ("final_recommendation", recommendation(bike(1, url=home))),
        ("final_recommendation", recommendation(bike(1, url=CUBE_URL))),
    )
    assert "nur die Startseite" in retry_prompts(result.all_messages())[0]


def test_category_conflict() -> None:
    mtb_page = "Giant Talon E+ EX – das E-Mountainbike Hardtail für Trails"
    assert category_conflict(BikeCategory.trekking, mtb_page) == ["mtb_hardtail", "mtb_fully"]
    assert category_conflict(BikeCategory.mtb_hardtail, mtb_page) == []
    # Seite ohne Kategoriebegriffe (z. B. kurzer Auszug) und Testberichte, die beide nennen, gelten als passend
    assert category_conflict(BikeCategory.trekking, "KTM Macina Style XL für 3.499 Euro") == []
    assert category_conflict(BikeCategory.trekking, "Test: Trekking-E-Bikes und E-MTBs 2026") == []


def test_check_sources_rejects_wrong_category_and_used_offers() -> None:
    used_url = "https://www.example-shop.de/collections/gebraucht/cube-kathmandu"
    pages = {
        normalize_url(CUBE_URL): "Cube Kathmandu Hybrid Pro 750 – E-Mountainbike Hardtail, 3.299 €",
        normalize_url(used_url): "Cube Kathmandu Hybrid Pro 750 Trekking, 3.299 €",
    }
    rec = BikeRecommendation.model_validate(recommendation(bike(1), bike(2, url=used_url)))
    with pytest.raises(ModelRetry) as error:
        check_sources(rec, pages)
    message = str(error.value)
    assert "Platz 1" in message and "nicht als trekking" in message
    assert "Platz 2" in message and "Gebraucht" in message


def test_model_must_be_on_page() -> None:
    assert model_on_page("Talon E+ EX Gen2", "Giant Talon E+ EX im Test")  # 3 von 4 Wörtern, Familie vorhanden
    assert not model_on_page("City Bike 20 Zoll D4 Rock", "Btwin Rockrider ST 500 Kinder 24 Zoll")
    assert not model_on_page("Kathmandu Hybrid", "Cube Hybrid Trekking")  # Modellfamilie fehlt

    mixed = BikeRecommendation.model_validate(recommendation(bike(1, model="Kathmandu Turbo Supreme X")))
    with pytest.raises(ModelRetry, match="steht so nicht auf"):
        check_sources(mixed, {normalize_url(CUBE_URL): "Cube Kathmandu Hybrid Pro 750 Trekking, 3.299 €"})


def test_test_article_category_and_price_count_per_model() -> None:
    """Testartikel über mehrere Räder: Kategorie und Preis müssen im Umfeld des jeweiligen Modells stehen."""
    filler = " Lorem ipsum." * 60  # > CONTEXT_CHARS zwischen den beiden Rädern
    article = (
        f"E-Trekkingräder 2026 im Test.{filler} Giant Talon E+ EX: ein E-Mountainbike Hardtail für Trails, "
        f"2.749 Euro.{filler} Cube Kathmandu Hybrid Pro 750 für 3.299 €."
    )
    pages = {normalize_url(CUBE_URL): article}
    talon = bike(1, brand="Giant", model="Talon E+ EX", price=2749)
    with pytest.raises(ModelRetry, match="nicht als trekking"):
        check_sources(BikeRecommendation.model_validate(recommendation(talon)), pages)
    # Preis eines anderen Rads aus demselben Artikel zählt nicht
    wrong_price = bike(1, price=2749)
    with pytest.raises(ModelRetry, match="nicht beim Modell"):
        check_sources(BikeRecommendation.model_validate(recommendation(wrong_price)), pages)
    check_sources(BikeRecommendation.model_validate(recommendation(bike(1))), pages)


def test_top_ranks_need_a_price() -> None:
    pages = {normalize_url(CUBE_URL): "Cube Kathmandu Hybrid Pro 750 Variante Trekking, 3.299 €"}
    ranking = [bike(i, price=3299 if i != 2 else None) for i in range(1, MAX_RANKING + 1)]
    with pytest.raises(ModelRetry) as error:
        check_sources(BikeRecommendation.model_validate(recommendation(*ranking)), pages)
    assert "Platz 2" in str(error.value) and "belegten Preis" in str(error.value)
    assert f"1 bis {PRICED_RANKS}" in INSTRUCTIONS
    # Hinter PRICED_RANKS darf der Preis fehlen (bei PRICED_RANKS = MAX_RANKING gibt es solche Plätze nicht)
    ranking[1] = bike(2)
    for rank in range(PRICED_RANKS + 1, MAX_RANKING + 1):
        ranking[rank - 1] = bike(rank, price=None)
    check_sources(BikeRecommendation.model_validate(recommendation(*ranking)), pages)


def test_no_price_required_once_web_budget_is_used() -> None:
    """Beratung 21: Ohne Web-Budget ließ sich der Preis nicht mehr belegen; die Forderung hielt den Lauf 15 Minuten
    in einer Schleife."""
    pages = {
        normalize_url(CUBE_URL): "Cube Kathmandu Hybrid Pro 750 Variante Trekking, 3.299 €",
        normalize_url(KTM_URL): "KTM Macina Style XL Trekking mit Bosch-Motor und Riemenantrieb",  # ohne Preis
    }
    ktm = bike(2, url=KTM_URL, price=None, brand="KTM", model="Macina Style XL")
    ranking = [ktm if i == 2 else bike(i) for i in range(1, MAX_RANKING + 1)]
    check_sources(BikeRecommendation.model_validate(recommendation(*ranking)), pages, web_budget=False)


def test_unbelegtes_rad_streichen_statt_suchen_ohne_web_budget() -> None:
    """Beratung 22: Ohne Such-Tool wollte das Modell ein unbelegtes Rad per Suche ersetzen und lief in eine Schleife."""
    unknown = BikeRecommendation.model_validate(recommendation(bike(1, url="https://www.example-shop.de/unbekannt")))
    with pytest.raises(ModelRetry) as with_budget:
        check_sources(unknown, {normalize_url(CUBE_URL): "Cube"})
    assert "streiche" not in str(with_budget.value)
    with pytest.raises(ModelRetry) as without_budget:
        check_sources(unknown, {normalize_url(CUBE_URL): "Cube"}, web_budget=False)
    assert "suche nicht weiter" in str(without_budget.value) and "streiche" in str(without_budget.value)


async def test_frame_size_tool_hidden_after_budget() -> None:
    tool_def = ToolDefinition(name="recommend_frame_size")

    def ctx(calls: int, deps: Any = None) -> Any:
        response = ModelResponse(parts=[ToolCallPart("recommend_frame_size", {}) for _ in range(calls)])
        deps = deps or SimpleNamespace(height_cm=172, inseam_cm=None, criteria={})
        return SimpleNamespace(messages=[ModelRequest(parts=[UserPromptPart("Gravelbike")]), response], deps=deps)

    assert await frame_size_budget(ctx(FRAME_SIZE_CALLS - 1), tool_def) is tool_def
    assert await frame_size_budget(ctx(FRAME_SIZE_CALLS), tool_def) is None


async def test_frame_size_tool_hidden_after_failing_without_measurements() -> None:
    """Beratung 22: sechs Aufrufe ohne Körpermaße, jeder mit demselben Fehler."""
    tool_def = ToolDefinition(name="recommend_frame_size")
    unmeasured = SimpleNamespace(height_cm=None, inseam_cm=None, criteria={})
    prompt = ModelRequest(parts=[UserPromptPart("Gravelbike")])
    call = ModelResponse(parts=[ToolCallPart("recommend_frame_size", {}, tool_call_id="c1")])
    failed = ModelRequest(parts=[ToolReturnPart("recommend_frame_size", {"error": NO_MEASUREMENTS}, tool_call_id="c1")])
    assert await frame_size_budget(SimpleNamespace(messages=[prompt], deps=unmeasured), tool_def) is tool_def
    after_failure = [prompt, call, failed]
    assert await frame_size_budget(SimpleNamespace(messages=after_failure, deps=unmeasured), tool_def) is None
    # Mit Maßen aus dem Intake bleibt es verfügbar (z. B. wenn das Modell sie nur nicht übergeben hatte)
    measured = SimpleNamespace(height_cm=None, inseam_cm=None, criteria={"height_cm": 172})
    assert await frame_size_budget(SimpleNamespace(messages=after_failure, deps=measured), tool_def) is tool_def


async def test_ranking_needs_five_bikes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agent.recommender.MIN_RANKING_RETRIES", 2)
    five = recommendation(*(bike(i, price=3299 if i <= 3 else None) for i in range(1, 6)))
    result = await run(
        SEARCH, RESEARCH, FRAME, ("final_recommendation", recommendation()), ("final_recommendation", five)
    )
    assert "soll 5 Räder enthalten, es fehlen 4" in retry_prompts(result.all_messages())[0]
    assert len(result.output.ranking) == 5


async def test_short_ranking_accepted_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agent.recommender.MIN_RANKING_RETRIES", 2)
    final = ("final_recommendation", recommendation())
    result = await run(SEARCH, RESEARCH, FRAME, final, final, final)
    assert len(retry_prompts(result.all_messages())) == 2 and len(result.output.ranking) == 1


async def test_web_tools_hidden_after_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nach TOOL_BUDGETS Aufrufen sind die Web-Tools weg; danach gilt auch eine kürzere Rangliste."""
    monkeypatch.setattr("app.agent.recommender.MIN_RANKING_RETRIES", 2)
    monkeypatch.setattr("app.agent.recommender.TOOL_BUDGETS", {"search_bikes": 1, "read_product_page": 1})
    seen: list[set[str]] = []

    def respond(messages: list[Any], info: Any) -> Any:
        seen.append({tool.name for tool in info.function_tools})
        steps = [
            SEARCH,
            ("read_product_page", {"url": CUBE_URL}),
            RESEARCH,
            FRAME,
            ("final_recommendation", recommendation()),
        ]
        return steps[len(seen) - 1]

    def model(messages: list[Any], info: Any) -> ModelResponse:
        name, args = respond(messages, info)
        return ModelResponse(parts=[ToolCallPart(name, args)])

    with recommender_agent.override(model=FunctionModel(model)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
            result = await recommender_agent.run("Trekking-E-Bike bis 3500 €", deps=deps)
    assert "search_bikes" in seen[0] and "read_product_page" in seen[0]
    assert "search_bikes" not in seen[2] and "read_product_page" not in seen[2]
    assert len(result.output.ranking) == 1 and not retry_prompts(result.all_messages())


@pytest.mark.parametrize(
    ("criteria", "offered"),
    [
        ({"use_case": "Kinder transportieren", "budget_eur": 6000}, False),
        ({"use_case": "Kinder transportieren"}, True),
        ({"budget_eur": 6000}, True),
    ],
)
async def test_question_offered_only_when_use_case_or_budget_missing(criteria: dict[str, Any], offered: bool) -> None:
    """Beratung 4: Rückfrage trotz bekanntem Einsatzzweck und Budget, abgelehnt, eine Modellanfrage verschenkt."""
    outputs: list[set[str]] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        outputs.append({tool.name for tool in info.output_tools})
        return ModelResponse(parts=[ToolCallPart("answer_user", {"message": "Gern."})])

    with recommender_agent.override(model=FunctionModel(model)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(
                settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute", criteria=criteria
            )
            await recommender_agent.run("E-Lastenrad", deps=deps)
    assert ("ask_user" in outputs[0]) is offered and "answer_user" in outputs[0]


@pytest.mark.parametrize(
    ("brand", "model"),
    [
        ("Canyon", "Grizl (Aluminium-Varianten)"),
        ("Canyon", "Grizl CF / AL (je nach Ausführung)"),
        ("Cube", "Nuroad / Nuroad-ähnliche Alu-Modelle"),
        ("Allgemeine Händlerangebote", "verschiedene Alu-Gravel-Modelle"),
        ("Specialized", "Diverge E5 (Alu-Builds)"),
    ],
)
def test_collective_entries_are_rejected(brand: str, model: str) -> None:
    """Beratungen 5 und 6: Sammeleinträge statt konkreter Modelle, erst bemängelt, als nichts mehr nachzusuchen war."""
    rec = BikeRecommendation.model_validate(recommendation(bike(1), bike(2, brand=brand, model=model)))
    with pytest.raises(ModelRetry, match="Platz 2") as exc:
        check_specific_models(rec)
    assert "search_bikes" in str(exc.value)
    with pytest.raises(ModelRetry, match="streiche den Platz"):
        check_specific_models(rec, web_budget=False)


def test_concrete_models_pass_collective_check() -> None:
    rec = BikeRecommendation.model_validate(
        recommendation(bike(1, brand="Van Rysel", model="GRVL AF (Shimano GRX 2x12 Variante)"), bike(2))
    )
    assert rec.ranking[0].model == "GRVL AF"  # einzelne Variante: Anmerkung, wird entfernt
    check_specific_models(rec)


def outputs_ctx(messages: list[Any], **criteria: Any) -> Any:
    settings = SimpleNamespace(tavily_api_key="tvly-test")
    deps = SimpleNamespace(ranking_due=True, settings=settings, criteria=criteria)
    return SimpleNamespace(messages=messages, deps=deps, retry=0, max_retries=3)


def test_only_ranking_offered_when_use_case_and_budget_known() -> None:
    """Beratungen 5 und 6: Ohne `ask_user` fragte das Modell per `answer_user` nach; jede Ablehnung kostete eine
    Korrektur der Rangliste. Fällt die Websuche aus, bleibt `answer_user`, um das zu sagen."""
    start = [ModelRequest(parts=[UserPromptPart("Gravelbike bis 2500 €")])]
    assert ranking_only(outputs_ctx(start, use_case="Gravel", budget_eur=2500))
    assert not ranking_only(outputs_ctx(start, use_case="Gravel"))
    failed = {"error": "Websuche (Tavily) ist nicht erreichbar: ConnectError"}
    calls = [ToolCallPart("search_bikes", {"query": "x"}, tool_call_id=f"s{i}") for i in range(2)]
    outage = [
        *start,
        ModelResponse(parts=calls),
        ModelRequest(parts=[ToolReturnPart("search_bikes", failed, tool_call_id=c.tool_call_id) for c in calls]),
    ]
    assert not ranking_only(outputs_ctx(outage, use_case="Gravel", budget_eur=2500))
    last = outputs_ctx(start, use_case="Gravel", budget_eur=2500)
    last.retry = 3
    assert ranking_only(last)  # Beratung 8: auch im letzten Versuch keine Rückfrage per `answer_user`


def test_brand_and_model_are_cleaned() -> None:
    """Beratung 8: Anmerkungen in Marke und Modell führten zu einer abgelehnten Rangliste und einer Extrarunde."""
    rec = BikeRecommendation.model_validate(
        recommendation(
            bike(1, brand="Van Rysel (Decathlon)", model="GRVL AF (Shimano GRX 2x12 Variante)"),
            bike(2, brand="ROSE / Backroad", model="Backroad AL (Vorjahresmodell)"),
            bike(3, brand="Rose Bikes GmbH", model="Backroad AL GRX 400"),
        )
    )
    assert [(b.brand, b.model) for b in rec.ranking] == [
        ("Van Rysel", "GRVL AF"), ("ROSE", "Backroad AL"), ("Rose Bikes", "Backroad AL GRX 400"),
    ]  # fmt: skip
    assert clean_name("Riese & Müller", brand=True) == "Riese & Müller"


async def test_research_only_once_per_consultation() -> None:
    seen: list[set[str]] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        seen.append({tool.name for tool in info.function_tools})
        steps = [RESEARCH, SEARCH, FRAME, ("final_recommendation", recommendation())]
        name, args = steps[len(seen) - 1]
        return ModelResponse(parts=[ToolCallPart(name, args)])

    with recommender_agent.override(model=FunctionModel(model)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
            first = await recommender_agent.run("Trekking-E-Bike bis 3500 €", deps=deps)
            seen.clear()
            await recommender_agent.run(
                "Lieber günstiger", deps=deps, message_history=first.all_messages(),
                model=FunctionModel(lambda m, i: (seen.append({t.name for t in i.function_tools}),
                                                  ModelResponse(parts=[ToolCallPart("answer_user", {"message": "ok"})]))[1]),
            )  # fmt: skip
    assert "research_knowledge" not in seen[0]  # Folgenachricht: schon recherchiert


async def test_question_after_rejected_ranking_is_still_checked() -> None:
    """Beratung 8: Nach einer abgelehnten Rangliste wich das Modell auf "Ist Carbon ein Muss?" aus."""
    wrong = recommendation(bike(1, url="https://erfunden.example/rad"))
    question = ("ask_user", {"question": "Ist Carbon ein Muss?", "missing_fields": []})
    steps = (
        SEARCH,
        RESEARCH,
        FRAME,
        ("final_recommendation", wrong),
        question,
        ("final_recommendation", recommendation()),
    )
    with recommender_agent.override(model=scripted_model(*steps)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(
                settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute",
                criteria={"use_case": "Bikepacking", "budget_eur": 2500},
            )  # fmt: skip
            async with recommender_agent.run_stream_events("Gravelbike bis 2500 €", deps=deps) as events:
                result = [e async for e in events][-1].result
    assert isinstance(result.output, BikeRecommendation)


def test_spec_notes_are_rejected() -> None:
    noted = bike(
        1, key_specs={**bike(1)["key_specs"], "motor": "Bosch (Einträge auf Händlerseiten)", "brakes": "~4-Kolben"}
    )
    with pytest.raises(ModelRetry, match="Anmerkungen statt Werten") as error:
        check_spec_notes(BikeRecommendation.model_validate(recommendation(noted)))
    assert "motor" in str(error.value) and "brakes" in str(error.value)
    check_spec_notes(BikeRecommendation.model_validate(recommendation(bike(1))))  # saubere Werte
    tenways = bike(1, key_specs={**bike(1)["key_specs"], "motor": "Mittelmotor (Hersteller nicht genannt)"})
    with pytest.raises(ModelRetry, match="Anmerkungen"):
        check_spec_notes(BikeRecommendation.model_validate(recommendation(tenways)))


async def test_requires_frame_size_tool_when_measures_known() -> None:
    result = await run(
        SEARCH, RESEARCH, ("final_recommendation", recommendation()), FRAME, ("final_recommendation", recommendation())
    )
    assert "recommend_frame_size" in retry_prompts(result.all_messages())[0]


async def test_no_frame_size_needed_without_measures() -> None:
    rec = recommendation(bike(1, frame_size=None), height_cm=None, inseam_cm=None)
    result = await run(SEARCH, RESEARCH, ("final_recommendation", rec))
    assert not retry_prompts(result.all_messages())


async def test_page_from_read_product_page_counts() -> None:
    page = ("read_product_page", {"url": CUBE_URL + "/"})  # abweichende Schreibweise der URL
    result = await run(SEARCH, page, RESEARCH, FRAME, ("final_recommendation", recommendation()))
    assert not retry_prompts(result.all_messages())


# --- Chat-Endpunkt ---


async def test_chat_streams_and_stores_recommendation(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    cid = await create_consultation(client, user_headers)
    steps = (
        [SEARCH, RESEARCH], FRAME,
        ("final_recommendation", recommendation(bike(1), bike(2, url=KTM_URL, brand="KTM", model="Macina Style XL", price=3499))),
    )  # fmt: skip
    with recommender_agent.override(model=scripted_model(*steps)):
        events = await chat(client, user_headers, cid, "Trekking-E-Bike bis 3500 €, 1,75 m, Schrittlänge 82")

    types = [e["type"] for e in events]
    assert types.count("tool_call") == 3 and types[-2:] == ["recommendation", "done"]
    stored = events[-2]["recommendation"]
    assert [b["rank"] for b in stored["ranking"]] == [1, 2]
    # Quellen bestimmt das Backend aus dem Verlauf, mit Seitentitel aus der Websuche
    assert stored["web_sources"] == [
        {"url": CUBE_URL, "title": SEARCH_OK["results"][0]["title"]},
        {"url": KTM_URL, "title": SEARCH_OK["results"][1]["title"]},
    ]

    consultation = (await client.get(f"/api/consultations/{cid}", headers=user_headers)).json()
    assert consultation["status"] == "recommended" and consultation["title"] == "Trekking-E-Bike bis 3.500 €"
    assert consultation["recommendation"]["ranking"][0]["brand"] == "Cube"
    assert [m["role"] for m in consultation["messages"]] == ["user", "assistant"]
    assert len(consultation["messages"][1]["tool_calls"]) == 3

    listing = (await client.get("/api/consultations", headers=user_headers)).json()
    assert listing[0]["bikes"][0].startswith("Cube Kathmandu")


async def test_chat_question_and_follow_up_keeps_history(
    client: httpx.AsyncClient, user_headers: dict[str, str]
) -> None:
    cid = await create_consultation(client, user_headers)
    question = (
        "ask_user",
        {"question": "Wofür nutzt Du das Rad, und was darf es kosten?", "missing_fields": ["budget_eur"]},
    )
    with recommender_agent.override(model=scripted_model(question)):
        events = await chat(client, user_headers, cid, "Ich brauche ein Fahrrad")
    assert events[-2]["type"] == "question" and "kosten" in events[-2]["question"]

    with recommender_agent.override(model=scripted_model(("answer_user", {"message": "Gern!"}))):
        events = await chat(client, user_headers, cid, "Pendeln, 1000 €")
    assert events[-2] == {"type": "answer", "content": "Gern!"}
    consultation = (await client.get(f"/api/consultations/{cid}", headers=user_headers)).json()
    assert len(consultation["messages"]) == 4 and consultation["recommendation"] is None


async def test_chat_on_foreign_consultation_is_404(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    cid = await create_consultation(client, user_headers)
    await create_user("other@example.com")
    other = await auth_headers(client, "other@example.com")
    response = await client.post(f"/api/chat/{cid}", headers=other, json={"message": "Hallo"})
    assert response.status_code == 404
    assert (await client.delete(f"/api/consultations/{cid}", headers=user_headers)).status_code == 204


async def test_text_answer_rejected_when_ranking_due() -> None:
    """Beratung 9: Nach der abgelehnten Rückfrage kündigte das Modell per answer_user nur an, es recherchiere jetzt."""
    announcement = ("answer_user", {"message": "Ich recherchiere jetzt gezielt. Nur DE-Shops oder europaweit?"})
    steps = (SEARCH, RESEARCH, FRAME, announcement, ("final_recommendation", recommendation()))
    with recommender_agent.override(model=scripted_model(*steps)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(
                settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute", ranking_due=True
            )
            async with recommender_agent.run_stream_events("Trekking-E-Bike bis 3500 €", deps=deps) as events:
                result = [e async for e in events][-1].result
    assert isinstance(result.output, BikeRecommendation)
    assert any("wartet auf die Rangliste" in text for text in retry_prompts(result.all_messages(), "answer_user"))


async def test_text_answer_allowed_without_ranking_due() -> None:
    with recommender_agent.override(model=scripted_model(("answer_user", {"message": "Ein Gravelbike ist ..."}))):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
            async with recommender_agent.run_stream_events("Was ist ein Gravelbike?", deps=deps) as events:
                result = [e async for e in events][-1].result
    assert isinstance(result.output, TextAnswer)


async def test_earlier_pages_are_compacted_only_for_the_model() -> None:
    long_page = {"url": CUBE_URL, "text": "Cube Kathmandu " + "x" * 4000, "prices_eur": [3299.0], "truncated": False}
    history = [
        ModelRequest(parts=[UserPromptPart("Trekking-E-Bike bis 3500 €")]),
        ModelResponse(parts=[ToolCallPart("read_product_page", {"url": CUBE_URL}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart("read_product_page", long_page, tool_call_id="c1")]),
        ModelResponse(parts=[ToolCallPart("answer_user", {"message": "ok"}, tool_call_id="c2")]),
        ModelRequest(parts=[ToolReturnPart("answer_user", "Final result processed.", tool_call_id="c2")]),
    ]
    sent: list[Any] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        sent.extend(messages)
        return ModelResponse(parts=[ToolCallPart("answer_user", {"message": "Gern"})])

    with recommender_agent.override(model=FunctionModel(model)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
            result = await recommender_agent.run("Erklär mir Platz 1", deps=deps, message_history=history)

    def page_texts(messages: list[Any]) -> list[str]:
        return [
            part.content["text"]
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "read_product_page"
        ]

    assert [len(text) for text in page_texts(sent)] == [EARLIER_PAGE_CHARS]
    assert page_texts(result.all_messages()) == [long_page["text"]]  # gespeicherter Verlauf vollständig
    assert known_pages(result.all_messages())[normalize_url(CUBE_URL)].strip() == long_page["text"]


async def test_all_problems_reported_at_once_and_last_attempt_keeps_clean_bikes() -> None:
    """Beratung 24: Der Budgetfehler kam erst beim vierten Versuch ans Licht, dann brach der Lauf mit „Exceeded
    maximum output retries“ ab. Jetzt stehen alle Probleme in der ersten Rückmeldung, und der letzte Versuch liefert
    die belegten Räder statt eines Fehlers."""
    bad = ("final_recommendation", recommendation(bike(1), bike(2, price=3999)))
    result = await run(SEARCH, RESEARCH, FRAME, bad, bad, bad, bad)
    rejections = retry_prompts(result.all_messages())
    assert len(rejections) == 3  # gestrichen wird erst beim letzten erlaubten Versuch
    first = rejections[0]
    assert "3999" in first and "Budget" in first  # Preisbeleg und Budget in einer Rückmeldung
    assert "erneut aufzurufen" in first
    output = result.output
    assert isinstance(output, BikeRecommendation) and [b.rank for b in output.ranking] == [1]
    assert "nicht ausreichend belegen" in output.buying_tips[-1]


FIVE_ONE_OVER_BUDGET = (
    "final_recommendation",
    recommendation(bike(1), bike(2, price=3999), bike(3), bike(4), bike(5)),
)
ONE_BIKE = ("final_recommendation", recommendation(bike(1)))


async def test_shrunk_ranking_must_keep_sound_bikes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beratung 2: Statt das bemängelte Rad zu streichen, schrumpfte die Rangliste auf ein Rad und eine Rückfrage."""
    monkeypatch.setattr("app.agent.recommender.TOOL_BUDGETS", {"search_bikes": 1, "read_product_page": 0})
    four = ("final_recommendation", recommendation(bike(1), bike(2), bike(3), bike(4)))
    result = await run(SEARCH, RESEARCH, FRAME, FIVE_ONE_OVER_BUDGET, ONE_BIKE, four)
    rejections = retry_prompts(result.all_messages())
    assert len(rejections) == 2
    assert "Variante 3" in rejections[1] and "Variante 5" in rejections[1] and "fehlen" in rejections[1]
    assert isinstance(result.output, BikeRecommendation) and len(result.output.ranking) == 4


async def test_last_attempt_prefers_longer_previous_ranking() -> None:
    """Beratung 2: Im letzten Versuch kam ein einziges Rad; die vorige Rangliste hätte gerettet vier ergeben."""
    result = await run(
        SEARCH, RESEARCH, FRAME, FIVE_ONE_OVER_BUDGET, FIVE_ONE_OVER_BUDGET, FIVE_ONE_OVER_BUDGET, ONE_BIKE
    )
    output = result.output
    assert isinstance(output, BikeRecommendation)
    assert [b.model[-1] for b in output.ranking] == ["1", "3", "4", "5"]
    assert [b.rank for b in output.ranking] == [1, 2, 3, 4]


async def test_last_attempt_keeps_bikes_with_data_gaps() -> None:
    """Beratung 32: Fehlende Motorangaben und „(Angabe in Quelle)“ im Motor strichen vier von fünf Rädern."""
    specs = bike(1)["key_specs"]
    gap = bike(2, key_specs={**specs, "motor": None, "battery_wh": None})
    noted = bike(3, key_specs={**specs, "motor": "Bosch Performance Line (Angabe in Quelle)"})
    bad = ("final_recommendation", recommendation(bike(1), gap, noted))
    result = await run(SEARCH, RESEARCH, FRAME, bad, bad, bad, bad)
    output = result.output
    assert isinstance(output, BikeRecommendation) and [b.rank for b in output.ranking] == [1, 2, 3]
    assert output.ranking[2].key_specs.motor is None  # Anmerkung geleert, Akku bleibt
    assert output.ranking[2].key_specs.battery_wh == 750
    assert "Platz 2" in output.buying_tips[-1] and "Motor und Akku" in output.buying_tips[-1]


async def test_frame_size_tool_lists_sizes_already_determined() -> None:
    tool_def = ToolDefinition(name="recommend_frame_size", description="Empfiehlt die Rahmengröße.")
    call = ModelResponse(parts=[ToolCallPart("recommend_frame_size", {"category": "kids"}, tool_call_id="c1")])
    done = ModelRequest(
        parts=[ToolReturnPart("recommend_frame_size", {"category": "kids", "size": "24 Zoll"}, tool_call_id="c1")]
    )
    deps = SimpleNamespace(height_cm=None, inseam_cm=None, criteria={"height_cm": 126})
    messages = [ModelRequest(parts=[UserPromptPart("Kinderrad")]), call, done]
    prepared = await frame_size_budget(SimpleNamespace(messages=messages, deps=deps), tool_def)
    assert prepared is not None and "kids: 24 Zoll" in (prepared.description or "")


def test_rejected_question_does_not_count_for_ranking_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beratung 26: Rückfrage und eine Rangliste abgelehnt, dann galt ein einzelnes Rad mit „dummy“ als fertig."""
    monkeypatch.setattr("app.agent.recommender.MIN_RANKING_RETRIES", 2)
    prompt = ModelRequest(parts=[UserPromptPart("Gravelbike bis 2500 €")])
    rejected = ModelRequest(
        parts=[
            RetryPromptPart("nein", tool_name="ask_user", tool_call_id="q"),
            RetryPromptPart("nein", tool_name="final_recommendation", tool_call_id="r"),
        ]
    )
    search = ModelResponse(parts=[ToolCallPart("search_bikes", {}, tool_call_id="s")])
    ctx = SimpleNamespace(messages=[prompt, search, rejected], retry=2, max_retries=3)
    with pytest.raises(ModelRetry, match="es fehlen 4"):
        check_ranking_size(BikeRecommendation.model_validate(recommendation()), ctx)
    # Letzter Versuch: eine kürzere Liste gilt, damit die Antwort nicht scheitert
    check_ranking_size(
        BikeRecommendation.model_validate(recommendation()), SimpleNamespace(**{**vars(ctx), "retry": 3})
    )


def test_placeholder_texts_are_rejected() -> None:
    dummy = BikeRecommendation.model_validate(recommendation() | {"summary": "dummy"})
    with pytest.raises(ModelRetry, match="Platzhalter"):
        check_placeholders(dummy)
    check_placeholders(BikeRecommendation.model_validate(recommendation()))


async def test_parallel_calls_cannot_exceed_budget(monkeypatch: pytest.MonkeyPatch, tavily: respx.MockRouter) -> None:
    """Beratung 26: 11 statt 8 Seitenabrufe, weil das Budget nur vor einer Runde galt."""
    monkeypatch.setattr("app.agent.recommender.TOOL_BUDGETS", {"search_bikes": 2, "read_product_page": 8})
    searches = [("search_bikes", {"query": f"Trekking E-Bike {i}"}) for i in range(3)]
    result = await run(searches, RESEARCH, FRAME, ("final_recommendation", recommendation()))
    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "search_bikes"
    ]
    assert len(returns) == 3 and tavily.routes[0].call_count == 2
    assert sum("Budget" in str(content) for content in returns) == 1


async def test_frame_size_tool_hidden_after_rounds() -> None:
    """Beratung 26: fünf Runden mit je einem Aufruf statt einer Runde mit allen."""
    tool_def = ToolDefinition(name="recommend_frame_size")
    deps = SimpleNamespace(height_cm=172, inseam_cm=None, criteria={})

    def round_(i: int) -> list[Any]:
        call = ToolCallPart("recommend_frame_size", {"category": "gravel"}, tool_call_id=f"c{i}")
        done = ToolReturnPart("recommend_frame_size", {"category": "gravel", "size": "M"}, tool_call_id=f"c{i}")
        return [ModelResponse(parts=[call]), ModelRequest(parts=[done])]

    prompt = ModelRequest(parts=[UserPromptPart("Gravelbike")])
    one_round = SimpleNamespace(messages=[prompt, *round_(1)], deps=deps)
    assert await frame_size_budget(one_round, tool_def) is not None
    two_rounds = SimpleNamespace(messages=[prompt, *round_(1), *round_(2)], deps=deps)
    assert await frame_size_budget(two_rounds, tool_def) is None


async def test_invented_url_is_not_fetched_and_does_not_use_budget(
    monkeypatch: pytest.MonkeyPatch, tavily: respx.MockRouter
) -> None:
    """Beratung 29: Selbst gebaute URLs (404) verbrauchten die Seitenabrufe, danach kam keine Rangliste."""
    monkeypatch.setattr("app.agent.recommender.TOOL_BUDGETS", {"search_bikes": 10, "read_product_page": 1})
    invented = ("read_product_page", {"url": "https://www.cube.eu/de/bikes/kathmandu-hybrid-pro-750/"})
    steps = (SEARCH, invented, ("read_product_page", {"url": CUBE_URL}), RESEARCH, FRAME)
    result = await run(*steps, ("final_recommendation", recommendation()))
    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "read_product_page"
    ]
    assert "stammt nicht aus `search_bikes`" in returns[0]["error"]
    assert CUBE_URL in returns[0]["error"]  # Vorschlag: Treffer mit denselben Wörtern im Pfad
    assert returns[1]["url"] == CUBE_URL  # trotz Budget 1: der abgelehnte Aufruf zählt nicht
    assert tavily.routes[1].call_count == 1


def test_match_url_tolerates_umlaut_spelling_and_www() -> None:
    """Beratung 31: „lastenrader“ aus dem Treffer als „lastenraeder“ abgeschrieben, „www.“ weggelassen."""
    known = [
        "https://www.bike-magazin.de/lastenraeder/monty-v8-neue-e-lastenrader-aus-spanien",
        "https://www.nimms-rad.de/news/lastenraeder-unter-5000-euro",
    ]
    assert (
        match_url("https://www.bike-magazin.de/lastenraeder/monty-v8-neue-e-lastenraeder-aus-spanien", known)
        == (known[0])
    )
    assert match_url("https://nimms-rad.de/news/lastenraeder-unter-5000-euro/", known) == known[1]
    assert match_url("https://www.bike-magazin.de/lastenraeder/monty-v6", known) is None


def test_misspelled_source_url_is_corrected() -> None:
    misspelled = CUBE_URL.replace("www.", "")
    pages = {normalize_url(CUBE_URL): "Cube Kathmandu Hybrid Pro 750 Trekking, 3.299 €"}
    rec = BikeRecommendation.model_validate(recommendation(bike(1, url=misspelled)))
    fix_source_urls(rec, pages)
    assert rec.ranking[0].price_source_url == normalize_url(CUBE_URL)
    check_sources(rec, pages)


async def test_repeated_page_read_is_refused_and_does_not_use_budget(
    monkeypatch: pytest.MonkeyPatch, tavily: respx.MockRouter
) -> None:
    """Beratung 31: Dieselben Seiten wurden drei- und zweimal gelesen, danach fehlte das Budget für weitere Räder."""
    monkeypatch.setattr("app.agent.recommender.TOOL_BUDGETS", {"search_bikes": 10, "read_product_page": 2})
    read = ("read_product_page", {"url": CUBE_URL})
    steps = (SEARCH, read, read, ("read_product_page", {"url": KTM_URL}), RESEARCH, FRAME)
    result = await run(*steps, ("final_recommendation", recommendation()))
    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "read_product_page"
    ]
    assert "schon so gelesen" in returns[1]["error"]
    assert returns[2]["url"] == KTM_URL  # trotz Budget 2: die Wiederholung zählt nicht
    assert tavily.routes[1].call_count == 2


def test_invented_urls_count_only_after_free_ones() -> None:
    """Beratung 7: Sechs abgelehnte, selbst gebaute URLs zählten, nach vier gelesenen Seiten war das Budget weg."""
    search = ToolCallPart("search_bikes", {"query": "x"}, tool_call_id="s")
    invented = [
        ToolCallPart("read_product_page", {"url": f"https://erfunden.example/rad-{i}"}, tool_call_id=f"r{i}")
        for i in range(FREE_UNKNOWN_URLS + 2)
    ]
    messages: list[Any] = [
        ModelRequest(parts=[UserPromptPart("Gravelbike")]),
        ModelResponse(parts=[search]),
        ModelRequest(parts=[ToolReturnPart("search_bikes", SEARCH_OK, tool_call_id="s")]),
        ModelResponse(parts=invented),
    ]
    assert FREE_UNKNOWN_URLS >= 8
    assert calls_in_run(messages, "read_product_page") == 2


def test_visible_price_required_even_without_web_budget() -> None:
    """Beratung 7: Mit aufgebrauchtem Budget setzte das Modell auch belegte Preise auf null."""
    priced = {normalize_url(CUBE_URL): "Cube Kathmandu Hybrid Pro 750 Trekking, jetzt 3.299,00 €"}
    rec = BikeRecommendation.model_validate(recommendation(bike(1, price=None)))
    with pytest.raises(ModelRetry, match="nennt beim Modell 3299 €"):
        check_sources(rec, priced, web_budget=False)
    check_sources(rec, priced, web_budget=False, salvaging=True)  # beim Retten bleibt das Rad
    check_sources(rec, {normalize_url(CUBE_URL): "Cube Kathmandu Hybrid Pro 750 Trekking"}, web_budget=False)


async def test_long_page_can_be_read_for_one_model(tavily: respx.MockRouter) -> None:
    """Beratung 8: Ein Testbericht war nach 5000 Zeichen gekürzt; das Modell las ihn immer wieder neu, um an die
    übrigen Räder zu kommen. Mit `model` kommen dessen Abschnitte, und das ist keine Wiederholung."""
    filler = "Allgemeines zum Gravelbike-Fahren und zur Ausstattung. " * 120
    raw = f"Cube Kathmandu Hybrid Pro 750 für 3.299 €. {filler} Giant Revolt Advanced 2 kostet 1.699 €. {filler}"
    tavily.routes[1].respond(json={"results": [{"url": CUBE_URL, "raw_content": raw}], "failed_results": []})
    focused = ("read_product_page", {"url": CUBE_URL, "model": "Revolt Advanced 2"})
    steps = (SEARCH, ("read_product_page", {"url": CUBE_URL}), focused, focused, RESEARCH, FRAME)
    result = await run(*steps, ("final_recommendation", recommendation()))
    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "read_product_page"
    ]
    assert returns[0]["truncated"] and "`model`" in returns[0]["note"] and "Revolt" not in returns[0]["text"]
    assert "Revolt Advanced 2 kostet 1.699 €" in returns[1]["text"] and 1699 in returns[1]["prices_eur"]
    assert "schon so gelesen" in returns[2]["error"]
    assert "Revolt" in known_pages(result.all_messages())[normalize_url(CUBE_URL)]  # Validator sieht den Abschnitt


def test_focus_text_without_mention() -> None:
    assert focus_text("Cube Nuroad C:62 für 2.000 €", "Revolt 0") is None
    assert focus_text("Cube Nuroad C:62 für 2.000 €", "Nuroad C:62") == "Cube Nuroad C:62 für 2.000 €"


def test_repeated_reads_count_after_free_ones() -> None:
    """Beratung 33: Zählen Wiederholungen nie, bleibt das Tool angeboten und das Modell kann endlos wiederholen."""
    search = ToolCallPart("search_bikes", {"query": "x"}, tool_call_id="s")
    messages: list[Any] = [
        ModelRequest(parts=[UserPromptPart("Trekking-E-Bike")]),
        ModelResponse(parts=[search]),
        ModelRequest(parts=[ToolReturnPart("search_bikes", SEARCH_OK, tool_call_id="s")]),
        ModelResponse(
            parts=[ToolCallPart("read_product_page", {"url": CUBE_URL}, tool_call_id=f"r{i}") for i in range(6)]
        ),
    ]
    assert calls_in_run(messages, "read_product_page") == 1 + 6 - 1 - FREE_REPEATED_READS


async def test_endless_repeated_reads_end_the_loop(tavily: respx.MockRouter) -> None:
    """Beratung 33: Nach dem ersten Abruf las das Modell dieselbe Seite 23-mal neu, jedes Mal abgelehnt und ohne
    Budget zu verbrauchen, bis request_limit=40 die Beratung ohne Ergebnis abbrach. Ein Modell, das liest, solange
    das Tool angeboten wird, muss trotzdem zu einer Rangliste kommen."""
    steps = iter([SEARCH, RESEARCH, FRAME])
    reads = 0

    async def stream(messages: list[Any], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        nonlocal reads
        name, args = next(steps, None) or (None, None)
        if name is None and any(tool.name == "read_product_page" for tool in info.function_tools):
            reads += 1
            name, args = "read_product_page", {"url": CUBE_URL}
        elif name is None:
            name, args = "final_recommendation", recommendation()
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=f"call_{uuid.uuid4().hex[:12]}")}

    with recommender_agent.override(model=FunctionModel(stream_function=stream)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
            async with recommender_agent.run_stream_events("Trekking-E-Bike bis 3500 €", deps=deps) as events:
                result = [event.result async for event in events if hasattr(event, "result")][-1]
    assert isinstance(result.output, BikeRecommendation)
    assert reads == 1 + FREE_REPEATED_READS + TOOL_BUDGETS["read_product_page"] - 1
    assert tavily.routes[1].call_count == 1


async def test_url_from_user_message_may_be_read(tavily: respx.MockRouter) -> None:
    url = "https://www.example-shop.de/mein-wunschrad"
    steps = (("read_product_page", {"url": url}), ("answer_user", {"message": "Das Rad passt."}))
    with recommender_agent.override(model=scripted_model(*steps)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute")
            async with recommender_agent.run_stream_events(f"Passt {url} zu mir?", deps=deps) as events:
                [e async for e in events]
    assert tavily.routes[1].call_count == 1


async def test_only_ranking_offered_once_web_budget_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beratung 29: Mit leerem Web-Budget fragte das Modell per ask_user/answer_user, ob es weiter suchen darf."""
    monkeypatch.setattr("app.agent.recommender.TOOL_BUDGETS", {"search_bikes": 1, "read_product_page": 1})
    outputs: list[set[str]] = []
    steps = [
        SEARCH,
        ("read_product_page", {"url": CUBE_URL}),
        RESEARCH,
        FRAME,
        ("final_recommendation", recommendation()),
    ]

    def model(messages: list[Any], info: Any) -> ModelResponse:
        outputs.append({tool.name for tool in info.output_tools})
        name, args = steps[len(outputs) - 1]
        return ModelResponse(parts=[ToolCallPart(name, args)])

    with recommender_agent.override(model=FunctionModel(model)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(
                settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute", ranking_due=True
            )
            result = await recommender_agent.run("Trekking-E-Bike bis 3500 €", deps=deps)
    assert {"ask_user", "answer_user"} <= outputs[0]
    assert outputs[-1] == {"final_recommendation"}
    assert isinstance(result.output, BikeRecommendation)


def test_category_from_url_path() -> None:
    """Beratung 30: Rose-Seite ohne „Gravel“ im Text, aber mit dem Reifen „Continental Grand Prix Urban“."""
    text = (
        "Rose Bikes\n## Backroad AL GRX 610\n* €2,500.00\nFrame material: aluminium\nTyres: Continental Terra Adventure\n"
        "## Backroad AL EQ\nTyres: Continental Grand Prix Urban\nDynamo: inclusive"
    )
    gravel_url = "https://www.rosebikes.com/bikes/gravel/adventure/backroad-al/models"
    other_url = "https://www.rosebikes.com/p/backroad-al-models"
    pages = {normalize_url(gravel_url): text, normalize_url(other_url): text}
    specs = {
        "weight_kg": 10.5,
        "drivetrain": "1x12",
        "motor": None,
        "battery_wh": None,
        "suspension": "starr",
        "brakes": "Scheibenbremsen",
    }
    rose = {"brand": "Rose", "model": "Backroad AL GRX 610", "category": "gravel", "price": 2500, "key_specs": specs}
    rec = BikeRecommendation.model_validate(recommendation(bike(1, url=gravel_url, **rose), budget=2500))
    check_sources(rec, pages)  # Kategorie aus dem URL-Pfad
    rec = BikeRecommendation.model_validate(recommendation(bike(1, url=other_url, **rose), budget=2500))
    with pytest.raises(ModelRetry, match="nicht als gravel"):
        check_sources(rec, pages)


async def test_question_without_missing_fields_is_rejected_and_only_ranking_remains() -> None:
    """Beratung 30: „Soll ich jetzt fünf Neuräder recherchieren?“ ohne fehlende Angabe; die zweite Rückfrage galt."""
    outputs: list[set[str]] = []
    question = ("ask_user", {"question": "Soll ich jetzt recherchieren?", "missing_fields": []})
    steps = [
        SEARCH,
        RESEARCH,
        FRAME,
        question,
        ("final_recommendation", recommendation(*[bike(i) for i in range(1, 6)])),
    ]

    def model(messages: list[Any], info: Any) -> ModelResponse:
        outputs.append({tool.name for tool in info.output_tools})
        name, args = steps[len(outputs) - 1]
        return ModelResponse(parts=[ToolCallPart(name, args)])

    with recommender_agent.override(model=FunctionModel(model)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(
                settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute", ranking_due=True
            )
            result = await recommender_agent.run("Trekking-E-Bike bis 3500 €", deps=deps)
    assert any("missing_fields" in text for text in retry_prompts(result.all_messages(), "ask_user"))
    assert {"ask_user", "answer_user"} <= outputs[3]
    assert outputs[4] == {"final_recommendation"}
    assert isinstance(result.output, BikeRecommendation)
