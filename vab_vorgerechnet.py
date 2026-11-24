"""Vorgerechnete Reihe: `tools/messreihen.json` enthält die Kennzahlen der Verwendungen V-a bis V-e für die Verteilungsmengen 1 bis 5 bei ε/D = 8, 3, 1 und 0,3 (erzeugt von
`tools/messreihen.py` mit dem Code der Demo). Die App zeigt daraus die Tabelle „dieselbe Zelle in fünf Mengen“; alles andere rechnet sie live."""

import json
from functools import lru_cache
from pathlib import Path

DATEI = Path(__file__).resolve().parent / "tools" / "messreihen.json"


@lru_cache(maxsize=1)
def lade() -> dict:
    try:
        return json.loads(DATEI.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def streuung(c: float):
    """Kennzahlen je Menge und Verwendung bei ε/D = c oder None, wenn c nicht vorgerechnet ist."""
    return lade().get("streuung", {}).get(repr(float(c)))
