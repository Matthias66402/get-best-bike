"""Berater, Output-Validator und Chat-Endpunkt mit geskriptetem Modell statt echtem LLM; Tavily per respx."""

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
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from app.agent.compaction import EARLIER_PAGE_CHARS
from app.agent.deps import AgentDeps
from app.agent.models import REQUEST_TIMEOUT_SECONDS
from app.agent.recommendation_schema import (
    MAX_RANKING,
    BikeCategory,
    BikeRecommendation,
    ClarificationQuestion,
    TextAnswer,
    clean_name,
)
from app.agent.recommender import (
    INSTRUCTIONS,
    PRICED_RANKS,
    category_conflict,
    check_sources,
    check_spec_notes,
    known_pages,
    model_on_page,
    recommender_agent,
)
from app.agent.tools.web import normalize_url
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


async def test_rejects_homepage_as_source() -> None:
    home = "https://www.cube.eu/"
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


async def test_question_rejected_when_use_case_and_budget_known() -> None:
    question = ("ask_user", {"question": "Wie groß bist Du?", "missing_fields": ["height_cm"]})
    with recommender_agent.override(model=scripted_model(question, question)):
        async with httpx.AsyncClient() as http:
            deps = AgentDeps(
                settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute",
                criteria={"use_case": "Kinder transportieren", "budget_eur": 6000},
            )  # fmt: skip
            async with recommender_agent.run_stream_events("E-Lastenrad bis 6000 €", deps=deps) as events:
                result = [e async for e in events][-1].result
    retries = [
        str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if getattr(part, "tool_name", None) == "ask_user" and getattr(part, "outcome", None) == "retried"
    ]
    assert len(retries) == 1 and "Einsatzzweck („Kinder transportieren“) und Budget (6000 €)" in retries[0]
    assert isinstance(result.output, ClarificationQuestion)  # beim zweiten Versuch gilt sie


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
    result = await run(page, RESEARCH, FRAME, ("final_recommendation", recommendation()))
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
