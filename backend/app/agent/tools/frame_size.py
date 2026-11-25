"""Rahmengröße aus Körpermaßen: Faustformeln je Kategorie oder die Größentabelle des Herstellers.

Deterministisch statt vom LLM geschätzt. Die Faustformeln sind Richtwerte; die Geometrie unterscheidet sich je
Hersteller, deshalb geht eine Größentabelle von der Produktseite immer vor.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.recommendation_schema import BikeCategory
from app.agent.tools.common import tool_error

# Schrittlänge ≈ 47 % der Körpergröße, wenn nur die Größe bekannt ist
INSEAM_SHARE = 0.47
NO_MEASUREMENTS = "Weder Körpergröße noch Schrittlänge bekannt. Den Nutzer fragen oder ohne Größe beraten."
NO_CHILD_HEIGHT = (
    "Für Kinderräder die Körpergröße des Kindes angeben (nicht die des Nutzers aus dem Profil); ist sie nicht "
    "bekannt, ohne Größe beraten."
)

# Faktor Schrittlänge → Sitzrohrlänge, Einheit, Größenstufen als (Obergrenze, Name)
MTB_SIZES = [(15.5, "XS"), (16.9, "S"), (18.4, "M"), (19.9, "L"), (21.4, "XL"), (99.0, "XXL")]
TREKKING_SIZES = [(45.5, "XS"), (49.5, "S"), (53.5, "M"), (57.5, "L"), (61.5, "XL"), (99.0, "XXL")]
ROAD_SIZES = [(48.5, "XS"), (51.5, "S"), (54.5, "M"), (57.5, "L"), (60.5, "XL"), (99.0, "XXL")]
FORMULAS: dict[BikeCategory, tuple[float, str, list[tuple[float, str]]]] = {
    BikeCategory.mtb_hardtail: (0.226, "Zoll", MTB_SIZES),
    BikeCategory.mtb_fully: (0.226, "Zoll", MTB_SIZES),
    BikeCategory.trekking: (0.66, "cm", TREKKING_SIZES),
    BikeCategory.city: (0.66, "cm", TREKKING_SIZES),
    BikeCategory.road: (0.665, "cm", ROAD_SIZES),
    BikeCategory.gravel: (0.665, "cm", ROAD_SIZES),
}
# Kinderräder nach Körpergröße: (Obergrenze cm, Laufradgröße)
KIDS_WHEELS = [(100, '12"'), (115, '16"'), (130, '20"'), (145, '24"'), (160, '26"'), (999, '27,5"/XS')]
# So nah an einer Grenze gilt die Angabe als "zwischen zwei Größen"
BORDER_SHARE = 0.03
# Kinderräder: So viele cm an einer Grenze der Laufradgröße passt auch die Nachbargröße (Beratung 25: 126 cm ergab
# 20 Zoll, obwohl der Nutzer ausdrücklich 24 Zoll wollte und 24 Zoll mit niedrigem Sattel passt)
KIDS_BORDER_CM = 5


def kids_wheel(height: int) -> dict[str, Any]:
    """Laufradgröße nach Körpergröße; nahe einer Grenze mit der Nachbargröße als Alternative."""
    index = next(i for i, (limit, _) in enumerate(KIDS_WHEELS) if height <= limit)
    limit, wheel = KIDS_WHEELS[index]
    result: dict[str, Any] = {"size": wheel, "method": f"Laufradgröße nach Körpergröße {height} cm"}
    if index + 1 < len(KIDS_WHEELS) and height > limit - KIDS_BORDER_CM:
        larger = KIDS_WHEELS[index + 1][1]
        result |= {
            "alternative": larger,
            "note": f"{height} cm liegt an der Grenze: {larger} passt mit niedrig eingestelltem Sattel und hält "
            "länger mit; Probefahrt machen.",
        }
    elif index > 0 and height <= KIDS_WHEELS[index - 1][0] + KIDS_BORDER_CM:
        smaller = KIDS_WHEELS[index - 1][1]
        result |= {
            "alternative": smaller,
            "note": f"{height} cm liegt an der Grenze: {wheel} nur mit niedrig eingestelltem Sattel, sonst "
            f"{smaller}; Probefahrt machen.",
        }
    return result


class SizeChartRow(BaseModel):
    size: str = Field(description='Größenbezeichnung wie auf der Seite, z. B. "M" oder "52"')
    min_height_cm: int
    max_height_cm: int


def _category_result(category: BikeCategory, inseam: float, estimated: bool) -> dict[str, Any]:
    factor, unit, sizes = FORMULAS[category]
    value = inseam * factor
    index = next(i for i, (limit, _) in enumerate(sizes) if value <= limit)
    size = sizes[index][1]
    lower = sizes[index - 1][0] if index else None
    upper = sizes[index][0]
    note = None
    if upper - value <= value * BORDER_SHARE and index + 1 < len(sizes):
        note = f"Zwischen {size} und {sizes[index + 1][1]}: sportlich die kleinere, aufrecht die größere Größe."
    elif lower is not None and value - lower <= value * BORDER_SHARE:
        note = f"Zwischen {sizes[index - 1][1]} und {size}: sportlich die kleinere, aufrecht die größere Größe."
    result: dict[str, Any] = {
        "size": size,
        "frame_height": f"{value:.0f} {unit}" if unit == "cm" else f'{value:.1f}"',
        "method": f"Faustformel Schrittlänge {inseam:.0f} cm × {factor}"
        + (" (Schrittlänge aus Körpergröße geschätzt)" if estimated else ""),
    }
    if note:
        result["note"] = note
    return result


async def recommend_frame_size(
    ctx: RunContext[AgentDeps],
    category: BikeCategory,
    height_cm: int | None = None,
    inseam_cm: int | None = None,
    size_chart: list[SizeChartRow] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Empfiehlt die Rahmengröße aus Körpergröße und Schrittlänge.

    Gibt es auf der Produktseite eine Größentabelle (Körpergröße je Rahmengröße), übergib sie als `size_chart`, dann
    gilt sie statt der Faustformel. Rufe das Tool für jedes Rad der Rangliste mit eigener Größentabelle bzw. einmal
    je Kategorie auf, alle in derselben Runde. Fehlende Maße ergänzt das Tool aus dem Profil des Nutzers,
    außer bei Kinderrädern: Dort die Körpergröße des Kindes übergeben.

    Args:
        category: Kategorie des Rads.
        height_cm: Körpergröße in cm.
        inseam_cm: Schrittlänge in cm (genauer als die Körpergröße).
        size_chart: Größentabelle des Herstellers, falls bekannt.
        model: Modellname, nur zur Zuordnung im Ergebnis.
    """
    if category == BikeCategory.kids:
        # Die Maße im Profil sind die des Nutzers, nicht die des Kindes: nur übergebene Maße zählen
        height, inseam = height_cm, inseam_cm
        if not height:
            return tool_error(NO_CHILD_HEIGHT)
    else:
        height = height_cm or ctx.deps.height_cm
        inseam = inseam_cm or ctx.deps.inseam_cm
    if not height and not inseam:
        return tool_error(NO_MEASUREMENTS)
    base: dict[str, Any] = {"model": model, "category": category.value} if model else {"category": category.value}

    if size_chart and height:
        rows = sorted(size_chart, key=lambda row: row.min_height_cm)
        matching = [row.size for row in rows if row.min_height_cm <= height <= row.max_height_cm]
        if matching:
            note = f"Laut Tabelle passen {' und '.join(matching)}." if len(matching) > 1 else None
            return (
                base
                | {"size": matching[0], "method": f"Größentabelle des Herstellers, Körpergröße {height} cm"}
                | ({"note": note} if note else {})
            )
        return base | {
            "size": rows[0].size if height < rows[0].min_height_cm else rows[-1].size,
            "method": f"Größentabelle des Herstellers, Körpergröße {height} cm",
            "note": "Körpergröße liegt außerhalb der Tabelle; Probefahrt dringend empfohlen.",
        }

    if category == BikeCategory.kids:
        assert height is not None  # oben geprüft
        return base | kids_wheel(height)

    if category in (BikeCategory.cargo, BikeCategory.folding, BikeCategory.other):
        return base | {
            "size": "Einheitsgröße",
            "method": "Lasten- und Falträder gibt es meist in einer Größe",
            "note": f"Auf den Einstellbereich von Sattel und Lenker achten (Körpergröße {height or '?'} cm).",
        }

    estimated = not inseam
    return base | _category_result(category, float(inseam or height * INSEAM_SHARE), estimated)
