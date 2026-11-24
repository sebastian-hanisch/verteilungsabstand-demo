"""Sinkhorn-Iteration im Stapel (numpy, float64) und die fünf Verwendungen des Worts „Sinkhorn-Distanz“ (V-a bis V-e).

Gerechnet wird die naive Form im Kern-Bereich: K = exp(−C/ε), Start u = v = 1, abwechselnd v = b / (Kᵀu) und u = a / (K v). Ein Durchlauf sind **zwei Matrix-Vektor-Produkte**.
Abbruch je Paar, sobald der Zeilenfehler (L1-Abstand der Zeilensummen des Plans zur Randverteilung a) unter `tol` liegt; das Paar wird danach eingefroren (das Ergebnis eines
Paares hängt nicht davon ab, welche anderen Paare im Stapel stehen). Der Kern-Bereich ist bei max C = 1 sicher, solange K.min() > 1e-300 (ε > 1/690); das prüft die Funktion.

Verwendungen (Wortlaut der drei Arbeiten gelesen, README „Quellen“):
- V-a: Cuturi 2013, Definition 1: min ⟨P, C⟩ unter P ∈ U(a, b) und KL(P ‖ a bᵀ) ≤ α (Nebenbedingung), in der Lagrange-Form: das ε' je Paar, bei dem KL = α gilt.
- V-b: ⟨C, P_ε⟩ (Kosten unter dem weichen Plan, ohne Entropieterm).
- V-c: Genevay u. a. 2018, Faktor 2: 2 W(a, b) − W(a, a) − W(b, b) mit W = V-b.
- V-d: Feydy u. a. 2019, Faktor ½: S_ε = OT_ε(a, b) − ½ OT_ε(a, a) − ½ OT_ε(b, b), OT_ε = ⟨C, P⟩ + ε KL(P ‖ a bᵀ).
- V-e: OT_ε(a, b) allein."""

import math
from dataclasses import dataclass

import numpy as np

K_MIN_KERN = 1e-300
EPS_MIN_A = 0.003                  # kleinstes ε' der Bisektion von V-a (max C = 1)
VERWENDUNGEN = ("V-a", "V-b", "V-c", "V-d", "V-e")


@dataclass(frozen=True)
class Stapel:
    u: np.ndarray                 # (P, n)
    v: np.ndarray                 # (P, n)
    kern: np.ndarray              # K (n, n) für ein gemeinsames ε oder (P, n, n) bei ε je Paar
    iterationen: np.ndarray       # (P,) Durchläufe bis zum Abbruch (1-basiert); max_iter, wenn nicht erreicht
    konvergiert: np.ndarray       # (P,) bool

    @property
    def produkte(self) -> np.ndarray:
        """Matrix-Vektor-Produkte je Paar: zwei je Durchlauf."""
        return 2 * self.iterationen


def kern(kosten: np.ndarray, eps) -> np.ndarray:
    """K = exp(−C/ε); eps ein Skalar (K: n × n) oder ein Vektor je Paar (K: P × n × n). Wirft ValueError bei Unterlauf."""
    eps = np.asarray(eps, dtype=float)
    k = np.exp(-kosten / eps) if eps.ndim == 0 else np.exp(-kosten[None, :, :] / eps[:, None, None])
    if k.min() <= K_MIN_KERN:
        raise ValueError(f"ε zu klein für float64 im Kern-Bereich (kleinster Wert von K: {k.min():.1e})")
    return k


def _mal_kern(k: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x K für jede Zeile von x (K ist symmetrisch, also gilt das auch für Kᵀx)."""
    if k.ndim == 2:
        return x @ k
    return np.matmul(x[:, None, :], k)[:, 0, :]


def sinkhorn_stapel(a: np.ndarray, b: np.ndarray, kosten: np.ndarray, eps, tol: float = 1e-10, max_iter: int = 200_000) -> Stapel:
    """Sinkhorn für P Paare (Zeilen von a und b) mit gemeinsamen Kosten (symmetrisch) und einem ε (Skalar) oder einem ε je Paar (Vektor).

    Gerechnet wird nur mit den Paaren, die noch nicht fertig sind (wenige Paare brauchen sehr viel länger als die meisten); ein fertiges Paar ändert sich nicht mehr."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    k_alle = kern(kosten, eps)
    anzahl = len(a)
    u_alle = np.ones_like(a)
    v_alle = np.ones_like(b)
    fertig = np.zeros(anzahl, dtype=bool)
    iterationen = np.full(anzahl, max_iter, dtype=np.int64)
    aktiv = np.arange(anzahl)
    k, aa, bb, u = k_alle, a, b, u_alle.copy()
    for t in range(1, max_iter + 1):
        v = bb / _mal_kern(k, u)
        kv = _mal_kern(k, v)
        fehler = np.abs(u * kv - aa).sum(axis=1)               # Zeilenfehler des Plans u_i K_ij v_j (die Spalten stimmen nach dem v-Schritt)
        gut = fehler < tol
        if gut.any():
            fertig_idx = aktiv[gut]
            u_alle[fertig_idx] = u[gut]
            v_alle[fertig_idx] = v[gut]
            iterationen[fertig_idx] = t
            fertig[fertig_idx] = True
            if gut.all():
                aktiv = aktiv[:0]
                break
            rest = ~gut
            aktiv, aa, bb, u, kv = aktiv[rest], aa[rest], bb[rest], u[rest], kv[rest]
            k = k if k.ndim == 2 else k[rest]
        u = aa / kv
    if len(aktiv):                                             # Grenze erreicht: der letzte Stand der unfertigen Paare
        u_alle[aktiv] = u
        v_alle[aktiv] = bb / _mal_kern(k, u)
    return Stapel(u_alle, v_alle, k_alle, iterationen, fertig)


def plaene(st: Stapel) -> np.ndarray:
    """Die Pläne P_ij = u_i K_ij v_j, Form (P, n, n)."""
    k = st.kern if st.kern.ndim == 3 else st.kern[None, :, :]
    return st.u[:, :, None] * k * st.v[:, None, :]


def kosten_und_kl(st: Stapel, a: np.ndarray, b: np.ndarray, kosten: np.ndarray):
    """(⟨C, P⟩, KL(P ‖ a bᵀ)) je Paar. KL direkt als Σ P ln(P / (a_i b_j)) über die Einträge P > 0."""
    p = plaene(st)
    kosten_wert = (p * kosten[None, :, :]).sum(axis=(1, 2))
    ref = a[:, :, None] * b[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p / ref), 0.0)
    return kosten_wert, t.sum(axis=(1, 2))


def paarliste(anzahl: int):
    """Indexpaare (i < j) in fester Reihenfolge."""
    return [(i, j) for i in range(anzahl) for j in range(i + 1, anzahl)]


@dataclass(frozen=True)
class Werte:
    """V-b bis V-e (und KL, Produkte) als Matrizen über alle Verteilungen; die Diagonale ist der Selbstabstand."""
    eps: float
    kosten_plan: np.ndarray       # ⟨C, P_ε⟩ (V-b), K × K
    ot: np.ndarray                # OT_ε (V-e)
    kl_plan: np.ndarray           # KL(P_ε ‖ a bᵀ)
    v_c: np.ndarray               # Genevay (V-c)
    v_d: np.ndarray               # Feydy (V-d)
    produkte: np.ndarray          # Matrix-Vektor-Produkte je Paar (K × K, Diagonale: Selbstlauf)
    nicht_konvergiert: int


def werte_b_bis_e(profile: np.ndarray, kosten: np.ndarray, eps: float, tol: float = 1e-10, max_iter: int = 200_000) -> Werte:
    """Alle Paare und alle Selbstläufe in einem Stapel (ein gemeinsames ε)."""
    anzahl = len(profile)
    paare = paarliste(anzahl) + [(i, i) for i in range(anzahl)]
    ia = np.array([p[0] for p in paare])
    ib = np.array([p[1] for p in paare])
    a, b = profile[ia], profile[ib]
    st = sinkhorn_stapel(a, b, kosten, eps, tol, max_iter)
    c_wert, kl_wert = kosten_und_kl(st, a, b, kosten)
    ot_wert = c_wert + eps * kl_wert
    mat = {name: np.zeros((anzahl, anzahl)) for name in ("c", "ot", "kl", "prod")}
    for name, vec in (("c", c_wert), ("ot", ot_wert), ("kl", kl_wert), ("prod", st.produkte.astype(float))):
        mat[name][ia, ib] = vec
        mat[name][ib, ia] = vec
    diag_c, diag_ot = np.diag(mat["c"]), np.diag(mat["ot"])
    v_c = 2 * mat["c"] - diag_c[:, None] - diag_c[None, :]
    v_d = mat["ot"] - 0.5 * diag_ot[:, None] - 0.5 * diag_ot[None, :]
    np.fill_diagonal(v_c, 0.0)
    np.fill_diagonal(v_d, 0.0)
    return Werte(float(eps), mat["c"], mat["ot"], mat["kl"], v_c, v_d, mat["prod"], int((~st.konvergiert).sum()))


def wert_nebenbedingung(profile: np.ndarray, kosten: np.ndarray, alpha: float, tol: float = 1e-9, eps_min: float = EPS_MIN_A, eps_max: float = 5.0, schritte: int = 22,
                        max_iter: int = 200_000, mit_maske: bool = False):
    """V-a für alle Paare und Selbstläufe: je Paar das ε' (Bisektion auf ln ε'), bei dem KL(P ‖ a bᵀ) = α gilt; der Wert ist ⟨C, P_ε'⟩.

    Liegt KL schon beim kleinsten ε' (eps_min) unter α, ist die Nebenbedingung nicht wirksam; der Wert ist dann ⟨C, P_ε'⟩ beim kleinsten ε'. Dieser Plan ist zulässig, der Wert
    also eine obere Schranke des Optimums unter der Nebenbedingung (das Optimum selbst liegt zwischen W2² und diesem Wert, wird aber nicht gerechnet).
    Rückgabe: (Matrix K × K der Werte, Zahl der Paare a ≠ b mit unwirksamer Nebenbedingung, Zahl nicht konvergierter Läufe); mit `mit_maske` zusätzlich ein bool-Vektor über
    `paarliste(len(profile))` (True: Nebenbedingung bei diesem Paar nicht wirksam)."""
    anzahl = len(profile)
    paare = paarliste(anzahl) + [(i, i) for i in range(anzahl)]
    ia = np.array([p[0] for p in paare])
    ib = np.array([p[1] for p in paare])
    a, b = profile[ia], profile[ib]
    nicht_konv = 0

    def lauf(eps_vec):
        nonlocal nicht_konv
        st = sinkhorn_stapel(a, b, kosten, eps_vec, tol, max_iter)
        nicht_konv += int((~st.konvergiert).sum())
        return kosten_und_kl(st, a, b, kosten)

    lo = np.full(len(paare), math.log(eps_min))
    hi = np.full(len(paare), math.log(eps_max))
    c_lo, kl_lo = lauf(np.exp(lo))
    unwirksam = kl_lo <= alpha
    for _ in range(schritte):
        mitte = 0.5 * (lo + hi)
        _, kl_mitte = lauf(np.exp(mitte))
        zu_klein = kl_mitte > alpha                       # KL fällt mit ε': bei KL > α muss ε' größer werden
        lo = np.where(zu_klein, mitte, lo)
        hi = np.where(zu_klein, hi, mitte)
    c_hi, _ = lauf(np.exp(hi))
    wert = np.where(unwirksam, c_lo, c_hi)
    mat = np.zeros((anzahl, anzahl))
    mat[ia, ib] = wert
    mat[ib, ia] = wert
    neben = np.array([p[0] != p[1] for p in paare])
    if mit_maske:
        return mat, int(np.sum(unwirksam & neben)), nicht_konv, (unwirksam & neben)[neben]
    return mat, int(np.sum(unwirksam & neben)), nicht_konv


def dreiecksverletzungen(mat: np.ndarray, toleranz: float = 1e-12) -> tuple[int, int, int]:
    """(Zahl der Tripel i < j < k mit verletzter Dreiecksungleichung für d = √max(Wert, 0), Zahl der Tripel, Zahl negativer Paarwerte)."""
    anzahl = len(mat)
    d = np.sqrt(np.maximum(mat, 0.0))
    i, j, k = (np.array(x) for x in zip(*[(i, j, k) for i in range(anzahl) for j in range(i + 1, anzahl) for k in range(j + 1, anzahl)])) if anzahl >= 3 else (np.array([]),) * 3
    if len(i) == 0:
        return 0, 0, 0
    x, y, z = d[i, j], d[j, k], d[i, k]
    verletzt = (x > y + z + toleranz) | (y > x + z + toleranz) | (z > x + y + toleranz)
    obere = np.triu_indices(anzahl, 1)
    return int(verletzt.sum()), len(i), int(np.sum(mat[obere] < 0))
