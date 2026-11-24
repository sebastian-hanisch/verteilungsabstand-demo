"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte.

Rechenaufwand am Regler-Maximum (alles numpy, ein Kern, Zeiten auf dem Entwicklungsrechner; auf der Cloud wahrscheinlich 2- bis 3-mal länger):
- Experiment 1: 17 Verschiebungen × 5 Abstände auf höchstens 96 Zellen, O(Zellen) je Wert, Millisekunden.
- Experiment 2: höchstens 5 000 Zellen, ein Aufruf von math.erf je Zellgrenze, unter 0,1 s.
- Experiment 3: höchstens 300 Zufalls-Seeds × 2 × 1 000 Beobachtungen, vektorisiert (sortieren, zählen), unter 0,5 s.
- Experiment 4: 24 Verteilungen auf 48 Zellen, 300 Sinkhorn-Läufe (276 Paare, 24 Selbstläufe) im Stapel. Bei ε/D ≥ 1 unter 0,1 s, bei 0,3 etwa 1 s (wenige Paare mit fast
  getrennten Trägern brauchen Hunderttausende Produkte); V-a (Bisektion, 24 Stapelläufe) 2 bis 3 s. Die Ergebnisse stehen im Zwischenspeicher. Unter ε/D = 0,3 wird nicht angeboten
  (gemessen bei 0,1 mit Menge 1, ein Kern, bei gleichzeitiger anderer Last des Rechners: bis etwa eine Minute, dabei erreichen einzelne Läufe die Grenze von 200 000 Durchläufen).
- Experiment 5: bis 400 Zufalls-Seeds × 400 Beobachtungen × 5 Prognosen, geschlossene Formeln, vektorisiert, etwa 0,5 s."""

import math

TAG = 24.0                                         # Stunden
SEED_MAX, DEFAULT_SEED = 999999, 2026
SERIEN_SEED = 2026                                 # Zufalls-Seed der Reihen der README: nie der Regler „Zufalls-Seed“

# Experiment 1: Verschieben statt Überlappen
PROFIL1 = ("Kasten 3 h", "Zwei Spitzen")
S1_MIN, S1_MAX, S1_SCHRITT = 0.0, 8.0, 0.5         # Verschiebung in Stunden
S1_WERTE = tuple(S1_SCHRITT * i for i in range(int(S1_MAX / S1_SCHRITT) + 1))
BREITE1_MIN, BREITE1_MAX = 0.3, 2.0                # σ der Spitzen in Stunden
ZELLEN1 = (24, 48, 96)
ACHSE1 = ("Gerade", "Kreis")
GLAETTUNG1 = (0.0, 1e-12, 1e-9, 1e-6)
KASTEN_START, KASTEN_BREITE = 6.0, 3.0             # Kasten 6 bis 9 Uhr
SPITZEN_LAGEN, SPITZEN_GEWICHTE = (6.5, 14.5), (0.4, 0.6)   # Zellenmitten (24 Zellen: die Zellen 6 und 14)
S_ZEILEN = (1.0, 2.0, 3.0, 4.0, 6.0)               # die Zeilen der Tabelle „zwei Spitzen“

# Experiment 2: Das Raster macht die Zahl
PROFIL2 = ("Zwei Spitzen", "Glatte Gauß-Profile")
ZELLEN2 = (10, 20, 24, 48, 50, 96, 100, 200, 500, 1000, 5000)
ZELLEN2_REIHE = (10, 20, 50, 100, 200, 500, 1000, 5000)
ABSTAND2 = (0.005, 0.01, 0.0137, 0.02, 0.05, 0.1)  # Abstand der Spitzen in Bruchteilen des Tages
BREITE2 = (0.01, 0.02, 0.03, 0.04)                 # σ der glatten Profile in Bruchteilen des Tages (bei 0,07 ragt der Rand des Tages in die Normalverteilung)
LAGE2 = "3/10"                                     # erste Spitze bei 0,3 des Tages (als Bruch, damit die Zellennummer exakt ist)
W2_ZEILEN = ((25, 0.7), (100, 0.7), (400, 0.7), (28, 0.7), (112, 0.7), (25, 1.0), (100, 1.0), (400, 1.0), (28, 1.0), (112, 1.0))   # (Zellen, σ der zweiten Verteilung); 28 und 112: Verschiebung 1,5 ein Vielfaches von h
RUNDUNG_GRENZE = 1e-12                             # darunter nennt die Verdikt-Zeile zu den glatten Profilen den Fehler „Rundung“ (Rundung liegt bei etwa 1e-15 bis 1e-17)

# Experiment 3: Histogramm ohne Unterschied
N3 = (50, 100, 200, 500, 1000)
B3 = (10, 20, 50, 100, 200, 500, 1000)
B3_REIHE = (10, 50, 200, 1000)
D3 = (0.0, 0.02, 0.05)
RICHTUNG3 = ("x ‖ y", "y ‖ x")
SEEDS3 = (100, 300)
MU3, SIGMA3 = 0.3, 0.04                            # N(0,3; 0,04²) auf [0, 1]

# Experiment 4: Distanz oder Divergenz?
ZELLEN4, ANZAHL4 = 48, 24                          # 24 Verteilungen auf 48 Zellen à 30 Minuten
C4 = (0.3, 1.0, 3.0, 8.0, 20.0)                    # c = ε/D; darunter brauchen einzelne Paare mit fast getrennten Trägern Hunderttausende Durchläufe
VERW4 = ("V-a", "V-b", "V-c", "V-d", "V-e")
MENGE4_MIN, MENGE4_MAX = 1, 20
TOL4, TOL4_A = 1e-10, 1e-9
MAX_ITER4 = 200_000

# Experiment 5: Prognosen bewerten
NOBS5 = (25, 100, 400)
SEEDS5 = (100, 400)
F1_LAGE = (0.1, 0.25, 0.5, 0.75, 1.0)
F2_SIGMA = (1.2, 1.4, 1.6, 2.0)
F3_SIGMA = (0.4, 0.55, 0.7, 0.85)
F4_LAGE = (0.0, 0.15, 0.3, 0.5)
F4_SIGMA = (1.0, 1.15, 1.3, 1.6)

# getrennte Zufallsströme (Zweck-Nummern)
ZWECK_MENGE, ZWECK_HIST, ZWECK_PROGNOSE = 1, 2, 3

COLOR_A = "#4c78a8"
COLOR_B = "#e08a1e"
COLOR_W1 = "#2F6B65"
COLOR_TV = "#4c78a8"
COLOR_KS = "#7b6fb3"
COLOR_JS = "#e08a1e"
COLOR_KL = "#c0504d"
COLOR_GRAU = "rgba(150,150,150,0.55)"

DEFAULTS = {
    "seed_input": DEFAULT_SEED,
    "profil1": PROFIL1[0],
    "s1": 3.0,
    "breite1": 1.0,
    "zellen1": 24,
    "achse1": ACHSE1[0],
    "glaettung1": 0.0,
    "profil2": PROFIL2[0],
    "zellen2": 100,
    "abstand2": 0.0137,
    "breite2": 0.04,
    "n3": 500,
    "b3": 50,
    "d3": 0.0,
    "richtung3": RICHTUNG3[0],
    "seeds3": 300,
    "c4": 1.0,
    "verw4": "V-d",
    "menge4": 1,
    "f1_lage": 0.5,
    "f2_sigma": 1.6,
    "f3_sigma": 0.55,
    "f4_lage": 0.3,
    "f4_sigma": 1.3,
    "nobs5": 100,
    "seeds5": 400,
}

# Presets: jede Zeile legt ALLE Regler fest (Schlüssel wie in DEFAULTS); `aktives_preset` vergleicht jeden Regler.
PRESETS = {
    "Kasten 3 h": dict(DEFAULTS),
    "Zwei Spitzen": dict(DEFAULTS, profil1=PROFIL1[1], s1=4.0, glaettung1=1e-12),
    "Raster 24": dict(DEFAULTS, zellen1=24, zellen2=24),
    "Raster 96": dict(DEFAULTS, zellen1=96, zellen2=96),
    "Selbstabstand": dict(DEFAULTS, c4=8.0, verw4="V-b"),
    "Prognosen": dict(DEFAULTS, nobs5=25),
}
PRESET_HELP = {
    "Kasten 3 h": "Die Voreinstellung: ein Nachfragekasten von 6 bis 9 Uhr, um s Stunden verschoben, auf 24 Stundenzellen. Beim Kasten wächst W1 mit s, TV, KS und JS sättigen, und KL ist ohne Glättung unendlich.",
    "Zwei Spitzen": "Ein Tagesprofil mit zwei Spitzen (6:30 und 14:30 Uhr, σ = 1 Stunde), um 4 Stunden verschoben: Experiment 1 zeigt die Zeile dieser Verschiebung in der Tabelle.",
    "Raster 24": "Experiment 2 auf 24 Zellen (eine Zelle = eine Stunde bzw. 1/24 des Tages; Experiment 1 hat in der Voreinstellung schon 24): beim voreingestellten Abstand der Spitzen liegen beide in derselben Zelle.",
    "Raster 96": "Experiment 1 und 2 auf 96 Zellen (eine Zelle = 15 Minuten bzw. 1/96 des Tages).",
    "Selbstabstand": "Experiment 4 bei ε/D = 8 mit der Verwendung V-b: der Abstand einer Verteilung zu sich selbst ist nicht 0, und für manche Verteilung liegt eine andere näher.",
    "Prognosen": "Experiment 5 mit 25 Beobachtungen je Zufalls-Seed: wie oft die wahre Prognose Platz 1 belegt, je Bewertungsregel.",
}

LN2 = math.log(2.0)
