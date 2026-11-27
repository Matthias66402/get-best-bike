import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.agent.deps import AgentDeps
from app.agent.recommendation_schema import BikeCategory
from app.agent.tools.frame_size import NO_CHILD_HEIGHT, SizeChartRow, recommend_frame_size
from app.agent.tools.prices import price_in_text, prices_in_text
from app.agent.tools.web import clean_page_text, normalize_url, read_product_page, search_bikes
from app.config import get_settings
from app.db import SessionLocal
from tests.helpers import CUBE_URL, EXTRACT_OK, SEARCH_OK, TAVILY_URL


@pytest.fixture
def tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "tavily_api_key", "tvly-test")


async def context(height_cm: int | None = None, inseam_cm: int | None = None) -> SimpleNamespace:
    http = httpx.AsyncClient()
    deps = AgentDeps(
        settings=get_settings(), http=http, session_factory=SessionLocal, user_name="Ute",
        height_cm=height_cm, inseam_cm=inseam_cm,
    )  # fmt: skip
    return SimpleNamespace(deps=deps)


# --- Preise ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("jetzt nur 3.299,00 € statt 3.599 €", [3299.0, 3599.0]),
        ("Preis: EUR 2.499,-", [2499.0]),
        ("€ 4,999.00 incl. VAT", [4999.0]),
        ("1 499 Euro", [1499.0]),
        ("899€", [899.0]),
        ("Versand 5 € und 750 Wh, Modell 2026", []),
    ],
)
def test_prices_in_text(text: str, expected: list[float]) -> None:
    assert prices_in_text(text) == expected


def test_price_in_text_tolerates_rounding() -> None:
    assert price_in_text(3299, "ab 3.299,00 €") and price_in_text(3300, "ab 3.299,00 €")
    assert not price_in_text(3199, "ab 3.299,00 €")


# --- Rahmengröße ---


async def test_frame_size_mtb_formula() -> None:
    result = await recommend_frame_size(await context(), BikeCategory.mtb_fully, height_cm=182, inseam_cm=86)
    assert result["size"] == "L" and result["frame_height"] == '19.4"'


async def test_frame_size_uses_profile_and_estimates_inseam() -> None:
    result = await recommend_frame_size(await context(height_cm=170), BikeCategory.trekking)
    assert result["size"] == "M" and "geschätzt" in result["method"]


async def test_frame_size_prefers_size_chart() -> None:
    chart = [
        SizeChartRow(size="S", min_height_cm=160, max_height_cm=170),
        SizeChartRow(size="M", min_height_cm=168, max_height_cm=180),
        SizeChartRow(size="L", min_height_cm=178, max_height_cm=190),
    ]
    result = await recommend_frame_size(await context(), BikeCategory.trekking, height_cm=179, size_chart=chart)
    assert result["size"] == "M" and "M und L" in result["note"] and "Größentabelle" in result["method"]


async def test_frame_size_kids_cargo_and_missing_measures() -> None:
    assert (await recommend_frame_size(await context(), BikeCategory.kids, height_cm=120))["size"] == '20"'
    assert (await recommend_frame_size(await context(), BikeCategory.cargo, height_cm=170))["size"] == "Einheitsgröße"
    assert "error" in await recommend_frame_size(await context(), BikeCategory.road)


async def test_kids_frame_size_ignores_profile_of_user() -> None:
    """Das Profil gehört dem Nutzer: 182 cm ergäben für ein Kinderrad eine falsche Größe."""
    ctx = await context(height_cm=182, inseam_cm=86)
    assert (await recommend_frame_size(ctx, BikeCategory.kids))["error"] == NO_CHILD_HEIGHT
    assert (await recommend_frame_size(ctx, BikeCategory.kids, height_cm=126))["size"] == '20"'
    assert (await recommend_frame_size(ctx, BikeCategory.gravel))["size"]  # Erwachsene: Profil gilt weiter


async def test_kids_wheel_offers_neighbour_size_at_border() -> None:
    """Beratung 25: 126 cm ergab nur 20 Zoll, der Nutzer wollte 24 Zoll."""
    at_upper = await recommend_frame_size(await context(), BikeCategory.kids, height_cm=126)
    assert at_upper["size"] == '20"' and at_upper["alternative"] == '24"' and "Grenze" in at_upper["note"]
    at_lower = await recommend_frame_size(await context(), BikeCategory.kids, height_cm=133)
    assert at_lower["size"] == '24"' and at_lower["alternative"] == '20"'
    middle = await recommend_frame_size(await context(), BikeCategory.kids, height_cm=122)
    assert middle["size"] == '20"' and "alternative" not in middle


# --- Websuche ---


async def test_search_without_key_is_an_error() -> None:
    assert "TAVILY_API_KEY" in (await search_bikes(await context(), "Gravelbike"))["error"]


@respx.mock
async def test_search_returns_results_with_prices_and_caches(tavily_key: None) -> None:
    route = respx.post(f"{TAVILY_URL}/search").respond(json=SEARCH_OK)
    ctx = await context()
    result = await search_bikes(ctx, "Trekking E-Bike bis 3500 Euro")
    assert [r["url"] for r in result["results"]] == [CUBE_URL, SEARCH_OK["results"][1]["url"]]
    assert result["results"][0]["prices_eur"] == [3299.0, 3599.0]
    assert route.calls.last.request.headers["Authorization"] == "Bearer tvly-test"

    # Gleiche Anfrage (andere Schreibweise): aus dem Cache, ohne zweiten Aufruf
    again = await search_bikes(ctx, "  trekking e-bike bis 3500 euro ")
    assert again["results"] == result["results"] and route.call_count == 1


@respx.mock
async def test_shop_search_limits_domains_and_has_own_cache_entry(tavily_key: None) -> None:
    """Beratung 9: Die allgemeine Suche lieferte kaum Euro-Preise; `shops=True` sucht nur bei deutschen Händlern."""
    route = respx.post(f"{TAVILY_URL}/search").respond(json=SEARCH_OK)
    ctx = await context()
    await search_bikes(ctx, "Gravelbike Carbon 2x bis 2500 Euro")
    general = json.loads(route.calls.last.request.content)
    assert "include_domains" not in general and "tavily.com" in general["exclude_domains"]
    await search_bikes(ctx, "Gravelbike Carbon 2x bis 2500 Euro", shops=True)
    assert route.call_count == 2  # nicht aus dem Cache der allgemeinen Suche
    shop = json.loads(route.calls.last.request.content)
    assert "rosebikes.de" in shop["include_domains"] and "idealo.de" in shop["include_domains"]
    assert "country" not in shop  # mit country liefert Tavily bei include_domains nichts


@respx.mock
async def test_search_errors_are_results(tavily_key: None) -> None:
    respx.post(f"{TAVILY_URL}/search").respond(status_code=432, text="usage limit exceeded")
    assert "HTTP 432" in (await search_bikes(await context(), "Gravelbike"))["error"]
    respx.post(f"{TAVILY_URL}/search").respond(json={"results": []})
    assert "Keine Treffer" in (await search_bikes(await context(), "Fahrrad xyz"))["error"]


@respx.mock
async def test_read_product_page(tavily_key: None) -> None:
    respx.post(f"{TAVILY_URL}/extract").respond(json=EXTRACT_OK)
    result = await read_product_page(await context(), CUBE_URL)
    assert result["prices_eur"] == [3299.0] and "Größentabelle" in result["text"] and not result["truncated"]

    respx.post(f"{TAVILY_URL}/extract").respond(json={"results": [], "failed_results": [{"error": "403"}]})
    assert "nicht lesbar (403)" in (await read_product_page(await context(), "https://example.org/x"))["error"]
    assert "Ungültige URL" in (await read_product_page(await context(), "ftp://example.org"))["error"]


def test_normalize_url() -> None:
    assert normalize_url("HTTPS://Shop.de/Rad/#preis") == normalize_url("https://shop.de/Rad")


def test_clean_page_text_drops_images_links_and_json() -> None:
    raw = (
        "* [![](https://cdn.example/logo.png)](https://shop.example/)\n"
        'gravel","LoggedIn":"false","crmId":"","ABTesting":"false"\n\n\n\n'
        "[Grizl CF 7](https://www.canyon.com/grizl-cf-7) 2.499 €"
    )
    assert clean_page_text(raw) == "*\n\nGrizl CF 7 2.499 €"
