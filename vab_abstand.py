"""Abstände und Divergenzen zwischen Verteilungen auf einem Raster (nur numpy und math, ohne Streamlit).

Eine Verteilung ist ein Vektor p von B Zellenwerten mit Summe 1 und Zellenbreite h. Alle Funktionen sind kleine Einheiten mit expliziten Ein- und Ausgaben:
- `w1`: Wasserstein-Abstand W1 aus den Verteilungsfunktionen, h · Σ |F_k − G_k| über die Zellgrenzen (1D, Kosten |t − t'|).
- `w1_kreis`: W1 auf dem Kreis (Mitternacht), h · Σ |D_k − Median(D)| mit D = F − G.
- `w2_quadrat`: W2² (Kosten (t − t')²) über die Quantilfunktionen, exakt auf dem Raster.
- `tv`, `ks`, `kl`, `js`: Totalvariationsabstand, Kolmogorov-Smirnov-Abstand, Kullback-Leibler-Divergenz (mit Glättung g je Zelle, ohne Renormierung), Jensen-Shannon-Divergenz.
Alle Werte in nat, wo ein Logarithmus vorkommt."""

import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[float])


def normal_cdf(z):
    """Verteilungsfunktion der Standardnormalverteilung (math.erf)."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def normal_pdf(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def verteilungsfunktion(p) -> np.ndarray:
    """F_k = p_1 + ... + p_k (an den Zellgrenzen, die letzte ist 1)."""
    return np.cumsum(np.asarray(p, dtype=float))


def w1(p, q, h: float = 1.0) -> float:
    """W1 auf der Geraden: h · Σ_k |F_k − G_k|."""
    return float(h * np.sum(np.abs(verteilungsfunktion(p) - verteilungsfunktion(q))))


def w1_kreis(p, q, h: float = 1.0) -> float:
    """W1 auf dem Kreis mit B Zellen der Breite h: h · Σ_k |D_k − Median(D)|, D = F − G."""
    d = verteilungsfunktion(p) - verteilungsfunktion(q)
    return float(h * np.sum(np.abs(d - np.median(d))))


def tv(p, q) -> float:
    """Totalvariationsabstand ½ Σ |p_k − q_k|."""
    return float(0.5 * np.sum(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))))


def ks(p, q) -> float:
    """Kolmogorov-Smirnov-Abstand max_k |F_k − G_k|."""
    return float(np.max(np.abs(verteilungsfunktion(p) - verteilungsfunktion(q))))


def kl(p, q, glaettung: float = 0.0) -> float:
    """KL(p ‖ q) = Σ p_k ln((p_k + g) / (q_k + g)) über die Zellen mit p_k > 0.

    Mit g = 0 ist der Wert unendlich, sobald eine Zelle p_k > 0 und q_k = 0 hat (die Zelle ist dann wirklich leer, kein Unterlauf)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pos = p > 0
    if glaettung == 0.0 and np.any(pos & (q <= 0)):
        return math.inf
    return float(np.sum(p[pos] * np.log((p[pos] + glaettung) / (q[pos] + glaettung))))


def js(p, q) -> float:
    """Jensen-Shannon-Divergenz ½ KL(p ‖ m) + ½ KL(q ‖ m), m = (p + q)/2 (höchstens ln 2)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def w2_quadrat(x, a, b) -> float:
    """W2² auf gemeinsamem Raster x (Zellenmitten) über die Quantilfunktionen (stückweise konstant): Σ Länge · (Q_a − Q_b)²."""
    x = np.asarray(x, dtype=float)
    ca, cb = np.cumsum(a), np.cumsum(b)
    ca[-1] = cb[-1] = 1.0
    t = np.unique(np.concatenate([[0.0], ca, cb]))
    mid = 0.5 * (t[:-1] + t[1:])
    qa = x[np.minimum(np.searchsorted(ca, mid, side="left"), len(x) - 1)]
    qb = x[np.minimum(np.searchsorted(cb, mid, side="left"), len(x) - 1)]
    return float(np.sum(np.diff(t) * (qa - qb) ** 2))


def kendall_tau(u, v) -> float:
    """Rangübereinstimmung Kendall τ_b zweier Vektoren (alle Paare, O(m²), numpy)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    su = np.sign(u[:, None] - u[None, :])
    sv = np.sign(v[:, None] - v[None, :])
    iu = np.triu_indices(len(u), 1)
    zaehler = float(np.sum(su[iu] * sv[iu]))
    nenner = math.sqrt(float(np.sum(su[iu] != 0)) * float(np.sum(sv[iu] != 0)))
    return zaehler / nenner if nenner > 0 else float("nan")
