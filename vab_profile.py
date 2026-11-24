"""Tagesprofile und Verteilungsmengen (Objekt Zeit): Kasten, zwei Spitzen, Gauß-Hügel, Gauß-Zellenmassen, Stichproben.

Ein Tagesprofil ist die Verteilung der Nachfrage über 24 Stunden auf B Zellen der Breite h = 24/B; die Zellenmitte der Zelle k ist (k + ½) h. Alles ist deterministisch aus
(Zufalls-Seed, Zweck, Index) mit dem Hausgenerator SplitMix64 (`vab_rng.py`)."""

import math

import numpy as np

from vab_abstand import normal_cdf
from vab_rng import MASK, SplitMix64

TAG = 24.0


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Seed, Zweck, Index); die ersten Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def zellenmitten(zellen: int) -> np.ndarray:
    """Zellenmitten in Stunden: (k + ½) · 24/B."""
    return (np.arange(zellen) + 0.5) * (TAG / zellen)


def _normieren(d: np.ndarray) -> np.ndarray:
    return d / d.sum()


def kasten(zellen: int, start: float, breite: float, verschiebung: float, kreis: bool = False) -> np.ndarray:
    """Gleichverteilung auf dem Intervall [start + s, start + s + breite) (Stunden); Zellenwert = Überlapp der Zelle mit dem Intervall.

    Auf der Geraden wird, was über den Tag hinausragt, abgeschnitten und neu normiert; auf dem Kreis läuft es über Mitternacht."""
    h = TAG / zellen
    lo = start + verschiebung
    hi = lo + breite
    links = np.arange(zellen) * h
    d = np.zeros(zellen)
    for versatz in ((-TAG, 0.0, TAG) if kreis else (0.0,)):
        d += np.clip(np.minimum(links + h, hi + versatz) - np.maximum(links, lo + versatz), 0.0, None)
    return _normieren(d)


def gauss_dichte(zellen: int, zentren, sigmas, gewichte, kreis: bool = False, boden: float = 0.0) -> np.ndarray:
    """Mischung aus Gauß-Hügeln (Zentren und σ in Stunden), an den Zellenmitten ausgewertet und normiert (Punktproben der Dichte, nicht Zellenintegrale)."""
    t = zellenmitten(zellen)
    d = np.zeros(zellen)
    for m, s, w in zip(zentren, sigmas, gewichte):
        for versatz in ((-TAG, 0.0, TAG) if kreis else (0.0,)):
            d += w * np.exp(-0.5 * ((t - (m + versatz)) / s) ** 2) / s
    return _normieren(d + boden)


def zwei_spitzen(zellen: int, verschiebung: float, sigma: float, kreis: bool = False, lagen=(6.5, 14.5), gewichte=(0.4, 0.6)) -> np.ndarray:
    """Zwei Spitzen (Gewichte 0,4 und 0,6) bei den Zellenmitten 6,5 und 14,5 Uhr (24 Zellen: die Zellen 6 und 14), um `verschiebung` Stunden verschoben."""
    return gauss_dichte(zellen, [m + verschiebung for m in lagen], [sigma] * len(lagen), gewichte, kreis)


def profil_menge(menge: int, anzahl: int, zellen: int, zweck: int) -> np.ndarray:
    """`anzahl` zufällige Tagesprofile (Zeilen): Mischung aus 1 bis 3 Gauß-Hügeln, Zentren 0,15 + 0,7 U des Tages, Breite 0,03 + 0,07 U des Tages,
    Gewichte Dirichlet(1); Dichte plus Boden 1e-12, dann normiert. Eine Verteilungsmenge ist durch `menge` (ihren Zufalls-Seed) festgelegt."""
    return np.array([_profil(menge, zweck, index, zellen) for index in range(anzahl)])


def _profil(menge: int, zweck: int, index: int, zellen: int) -> np.ndarray:
    g = instanz_rng(menge, zweck, index)
    nc = g.below(3) + 1
    mu = [(0.15 + 0.7 * g.random()) * TAG for _ in range(nc)]
    sg = [(0.03 + 0.07 * g.random()) * TAG for _ in range(nc)]
    e = np.array([g.expovariate(1.0) for _ in range(nc)])
    w = e / e.sum()
    t = zellenmitten(zellen)
    d = sum(wi * np.exp(-0.5 * ((t - mi) / si) ** 2) / si for wi, mi, si in zip(w, mu, sg))
    return _normieren(d + 1e-12)


def gauss_zellenmassen(rand, mu: float, sigma: float) -> np.ndarray:
    """Zellenintegrale einer Normalverteilung N(mu, sigma²) über die Zellen mit den Rändern `rand` (B + 1 Werte), auf Summe 1 normiert."""
    c = normal_cdf((np.asarray(rand, dtype=float) - mu) / sigma)
    m = np.diff(c)
    return m / m.sum()


def zufall_gleich(seed: int, zweck: int, zeilen: int, spalten: int) -> np.ndarray:
    """Gleichverteilte Zahlen in [0, 1): `zeilen` eigene Ströme (Zeile j = Strom (seed, zweck, j)) mit je `spalten` Zahlen, vektorisiert mit uint64
    (dieselben Zahlen wie `SplitMix64.random()`, Test `test_profile.py`)."""
    z0 = np.array([instanz_rng(seed, zweck, j).state for j in range(zeilen)], dtype=np.uint64)
    i = np.arange(1, spalten + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z0[:, None] + i[None, :] * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def zufall_normal(seed: int, zweck: int, zeilen: int, spalten: int) -> np.ndarray:
    """Standardnormalzahlen (Box-Muller wie `SplitMix64.normal()`, aber mit numpy-Funktionen; auf der letzten Binärstelle plattformabhängig)."""
    u = zufall_gleich(seed, zweck, zeilen, 2 * spalten)
    u1 = np.maximum(u[:, 0::2], 1e-12)
    u2 = u[:, 1::2]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
