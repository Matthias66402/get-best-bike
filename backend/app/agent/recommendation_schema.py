"""Ausgabeformate des Fahrradberaters (app.agent.recommender)."""

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_RANKING = 5


class BikeCategory(StrEnum):
    city = "city"  # Stadt- und Hollandrad
    trekking = "trekking"
    road = "road"  # Rennrad
    gravel = "gravel"
    mtb_hardtail = "mtb_hardtail"
    mtb_fully = "mtb_fully"
    cargo = "cargo"  # Lastenrad
    folding = "folding"  # Faltrad
    kids = "kids"
    other = "other"


class Requirements(BaseModel):
    """Vom Berater erkannte Kriterien; null, wenn der Nutzer nichts dazu gesagt hat."""

    use_case: str = Field(description='Einsatzzweck, z. B. "täglich 15 km pendeln, teils Schotter"')
    category: BikeCategory | None = Field(description="Gesuchte Kategorie")
    electric: bool | None = Field(description="E-Bike gewünscht (true), ausgeschlossen (false) oder offen (null)")
    budget_eur: float | None = Field(description="Höchstpreis in Euro")
    height_cm: int | None = Field(description="Körpergröße in cm")
    inseam_cm: int | None = Field(description="Schrittlänge in cm")
    other_wishes: list[str] = Field(description="Weitere Wünsche, z. B. Riemenantrieb, Tiefeinsteiger, < 12 kg")


class KeySpecs(BaseModel):
    weight_kg: float | None = Field(description="Gewicht laut Quelle, sonst null")
    drivetrain: str | None = Field(description='Schaltung bzw. Antrieb, z. B. "Shimano Deore 1x12" oder "Enviolo"')
    motor: str | None = Field(description='Motor bei E-Bikes, z. B. "Bosch Performance Line CX, 85 Nm"')
    battery_wh: int | None = Field(description="Akkukapazität in Wh bei E-Bikes")
    suspension: str | None = Field(description='Federung, z. B. "Federgabel 120 mm" oder "starr"')
    brakes: str | None = Field(description='Bremsen, z. B. "hydraulische Scheibenbremsen"')


# Sammeleinträge statt eines Modells (Beratungen 5 und 6): „Grizl (Aluminium-Varianten)“, „Grizl CF / AL (je nach
# Ausführung)“, „Nuroad / Nuroad-ähnliche Alu-Modelle“, Marke „Allgemeine Händlerangebote“
VAGUE_NAME = re.compile(
    r"varianten|ähnlich|verschiedene|je nach|ausführungen|builds\b|modelle\b|händler|angebote|allgemein|\boder\b|\s/\s",
    re.IGNORECASE,
)


def clean_name(name: str, *, brand: bool = False) -> str:
    """Entfernt Anmerkungen, die das Modell in Namen schreibt (Beratung 8): "Van Rysel (Decathlon)" → "Van Rysel",
    "ROSE / Backroad" → "ROSE", "Rose Bikes GmbH" → "Rose Bikes", "Checkpoint ALR (Vorjahresmodell)" → "Checkpoint ALR".
    Sonst lehnte der Validator die Rangliste ab, weil der Name so auf keiner Seite steht. Klammern mit Sammelangaben
    (VAGUE_NAME) bleiben stehen, damit `check_specific_models` sie zurückweist, statt einen Familiennamen zu prüfen."""
    cleaned = re.sub(r"\s*[(\[]([^)\]]*)[)\]]", lambda m: m[0] if VAGUE_NAME.search(m[1]) else "", name)
    if brand:
        cleaned = re.split(r"\s+/\s+|\s*/\s*(?=[A-ZÄÖÜ])", cleaned)[0]
        cleaned = re.sub(r"\s+(GmbH|AG|Inc\.?|Ltd\.?|S\.?A\.?S?\.?)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -–,/")
    return cleaned or name.strip()


class RankedBike(BaseModel):
    rank: int = Field(ge=1, le=MAX_RANKING, description="Platz in der Rangliste, 1 = beste Wahl")
    brand: str = Field(description='Nur der Markenname wie auf der Seite, z. B. "Van Rysel"')
    model: str = Field(description='Nur der Modellname wie auf der Seite, ohne Anmerkungen, z. B. "GRVL AF"')

    @field_validator("brand")
    @classmethod
    def clean_brand(cls, value: str) -> str:
        return clean_name(value, brand=True)

    @field_validator("model")
    @classmethod
    def clean_model(cls, value: str) -> str:
        return clean_name(value)

    model_year: int | None = Field(description="Modelljahr laut Quelle, sonst null")
    category: BikeCategory
    price_eur: float | None = Field(
        description="Preis in Euro, genau wie auf der Seite von price_source_url angegeben; unbekannt = null"
    )
    price_source_url: str = Field(
        description="URL aus search_bikes bzw. read_product_page, die dieses Modell (und den Preis) belegt"
    )
    key_specs: KeySpecs
    frame_size: str | None = Field(
        description='Empfohlene Rahmengröße aus recommend_frame_size, z. B. "M (52 cm)"; ohne Körpermaße null'
    )
    match_score: int = Field(ge=0, le=100, description="Wie gut das Rad zu den Kriterien passt (0-100)")
    why_fits: str = Field(description="1-2 Sätze, warum das Rad zu diesem Nutzer passt")
    pros: list[str] = Field(description="2-4 Vorteile")
    cons: list[str] = Field(description="1-3 Nachteile bzw. Kompromisse")


class BikeRecommendation(BaseModel):
    """Rangliste der 1 bis 5 am besten passenden Fahrräder."""

    title: str = Field(description='Kurzer Titel der Beratung, z. B. "Trekking-E-Bike bis 3.500 €"')
    summary: str = Field(description="2-3 Sätze: Was gesucht war und warum Platz 1 vorn liegt")
    requirements: Requirements
    ranking: list[RankedBike] = Field(min_length=1, max_length=MAX_RANKING)
    buying_tips: list[str] = Field(description="Kurze Kauftipps, z. B. Probefahrt, Vorjahresmodelle, Ausstattung")
    sources: list[str] = Field(description="Genutzte Quellen (Titel bzw. URLs) aus Websuche und Wissensbasis")

    @model_validator(mode="after")
    def ranks_in_order(self) -> "BikeRecommendation":
        self.ranking = sorted(self.ranking, key=lambda bike: bike.rank)
        expected = list(range(1, len(self.ranking) + 1))
        if [bike.rank for bike in self.ranking] != expected:
            raise ValueError(f"Die Ränge müssen lückenlos {expected} sein")
        models = [f"{b.brand} {b.model}".casefold().strip() for b in self.ranking]
        if len(set(models)) != len(models):
            raise ValueError("Jedes Modell darf nur einmal in der Rangliste stehen")
        return self


class TextAnswer(BaseModel):
    """Antwort ohne Rangliste, z. B. Erklärung eines Begriffs oder Antwort auf eine Frage zu einem Rad."""

    message: str = Field(description="Kurze Antwort auf Deutsch")


class ClarificationQuestion(BaseModel):
    """Rückfrage an den Nutzer, wenn entscheidende Angaben fehlen."""

    question: str = Field(description="Freundliche Rückfrage, die alle fehlenden Punkte auf einmal abdeckt")
    missing_fields: list[str] = Field(
        default_factory=list, description="z. B. use_case, budget_eur, category, electric, height_cm"
    )
