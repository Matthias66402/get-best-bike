"""Preise in Webtexten erkennen; Grundlage für die Prüfung, ob ein Preis der Empfehlung belegt ist."""

import re

# Zahl wie 4.999,00 | 4999 | 4 999 | 4,999.00 | 2.499,- | 899
NUMBER = r"\d{1,3}(?:[.,\s'’]\d{3})+(?:[.,]\d{1,2}|,-{1,2})?|\d+(?:[.,]\d{1,2}|,-{1,2})?"
CURRENCY = r"€|EUR\b|Euro\b"
PRICE = re.compile(rf"(?:(?:{CURRENCY})\s*(?P<a>{NUMBER}))|(?:(?P<b>{NUMBER})\s*(?:{CURRENCY}))", re.IGNORECASE)
# Abweichung durch Rundung ("ca. 3.500 €" für 3.499 €)
TOLERANCE_EUR = 1.5
# Typische Preisspanne von Fahrrädern; filtert z. B. "5 € Versand" aus
MIN_PRICE, MAX_PRICE = 50, 30_000


def parse_number(text: str) -> float | None:
    text = re.sub(r",-{1,2}$", "", text.strip())
    text = re.sub(r"[\s'’]", "", text)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d{1,2})?", text):  # deutsch: 4.999,00
        return float(text.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d{1,2})?", text):  # englisch: 4,999.00
        return float(text.replace(",", ""))
    if re.fullmatch(r"\d+,\d{1,2}", text):
        return float(text.replace(",", "."))
    try:
        return float(text)
    except ValueError:
        return None


def prices_in_text(text: str) -> list[float]:
    """Beträge mit Währungsangabe in plausibler Fahrrad-Preisspanne, in Reihenfolge ohne Duplikate."""
    found: dict[float, None] = {}
    for match in PRICE.finditer(text):
        value = parse_number(match.group("a") or match.group("b"))
        if value is not None and MIN_PRICE <= value <= MAX_PRICE:
            found.setdefault(value, None)
    return list(found)


def price_in_text(price: float, text: str) -> bool:
    return any(abs(price - found) <= TOLERANCE_EUR for found in prices_in_text(text))
