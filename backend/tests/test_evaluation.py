import json

from pydantic_ai import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from app.agent.recommendation_schema import BikeRecommendation, ClarificationQuestion
from app.agent.runner import RunOutcome
from app.evaluation.recommender import CASES_FILE, CaseRun, check_output, measure
from tests.helpers import CUBE_URL, SEARCH_OK, bike, recommendation

CASE = {
    "expected_output": "final_recommendation",
    "category": ["trekking"],
    "electric": True,
    "budget_eur": 3500,
    "frame_size": True,
}
MESSAGES = [
    ModelResponse(parts=[ToolCallPart("search_bikes", {"query": "x"}, tool_call_id="c1")]),
    ModelRequest(parts=[ToolReturnPart("search_bikes", SEARCH_OK, tool_call_id="c1")]),
]


def test_cases_file_is_valid() -> None:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    assert {c["expected_output"] for c in cases} <= {"final_recommendation", "ask_user"}
    assert len({c["id"] for c in cases}) == len(cases)


def test_check_output_for_recommendation() -> None:
    good = RunOutcome(recommendation=BikeRecommendation.model_validate(recommendation(bike(1))))
    assert all(check_output(CASE, good, MESSAGES).values())

    bad = recommendation(bike(1, url="https://anderswo.example/rad", frame_size=None, category="gravel", price=4000))
    checks = check_output(CASE, RunOutcome(recommendation=BikeRecommendation.model_validate(bad)), MESSAGES)
    assert not any(checks[name] for name in ("Quellen belegt", "Kategorie", "Budget", "Rahmengröße"))


def test_check_output_for_question() -> None:
    outcome = RunOutcome(question=ClarificationQuestion(question="Wofür?"))
    assert check_output({"expected_output": "ask_user"}, outcome, []) == {"Rückfrage": True}
    assert check_output(CASE, outcome, []) == {"Rangliste": False}


def test_measure_counts_tools() -> None:
    run = CaseRun(case="x")
    measure(run, [*MESSAGES, ModelResponse(parts=[ToolCallPart("final_recommendation", {}, tool_call_id="c2")])])
    assert run.tool_calls == {"search_bikes": 1} and run.web_calls == 1 and run.model_requests == 2
    assert CUBE_URL in json.dumps(SEARCH_OK)
