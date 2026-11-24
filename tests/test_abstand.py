"""Handbeispiele zu den Abständen (in Brüchen, von Hand gerechnet) und die Einheiten `vab_abstand`: Kasten, Treppe, Tagesprofile mit zwei Spitzen, Kreis, KL ohne Unterschied, Rasterfehler von W2²."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import vab_abstand as A
import vab_evaluation as E
import vab_profile as PR


# ---- Bausteine in Brüchen (unabhängig von vab_abstand) -----------------------------------------------------------------------------

def w1_bruch(p, q, h=Fr(1)):
    f = g = s = Fr(0)
    for a, b in zip(p, q):
        f += a
        g += b
        s += abs(f - g)
    return s * h


def tv_bruch(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) / 2


def ks_bruch(p, q):
    f = g = m = Fr(0)
    for a, b in zip(p, q):
        f += a
        g += b
        m = max(m, abs(f - g))
    return m


def kasten_bruch(d):
    p = [Fr(0)] * 24
    for k in (6, 7, 8):
        p[k + d] = Fr(1, 3)
    return p


# ---- Handbeispiel 1: Kasten verschieben ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", range(0, 7))
def test_kasten_verschieben_in_bruechen(d):
    p, q = kasten_bruch(0), kasten_bruch(d)
    assert w1_bruch(p, q) == d and tv_bruch(p, q) == Fr(min(d, 3), 3) and ks_bruch(p, q) == Fr(min(d, 3), 3)
    fp, fq = [float(x) for x in p], [float(x) for x in q]
    assert A.w1(fp, fq, 1.0) == pytest.approx(d, abs=1e-12) and A.tv(fp, fq) == pytest.approx(min(d, 3) / 3, abs=1e-12) and A.ks(fp, fq) == pytest.approx(min(d, 3) / 3, abs=1e-12)
    kasten = PR.kasten(24, 6.0, 3.0, float(d))
    assert np.allclose(kasten, fq, atol=1e-15)
    if d == 0:
        assert A.kl(fq, fp) == 0.0 and A.js(fp, fq) == 0.0
    else:
        assert A.kl(fq, fp) == math.inf                                 # die verschobene Zelle ist im Ausgang leer
    if d >= 3:
        assert A.js(fp, fq) == pytest.approx(math.log(2), abs=1e-15)    # getrennte Träger
    elif d >= 1:
        assert 0 < A.js(fp, fq) < math.log(2)


@pytest.mark.parametrize("d", [Fr(1, 2), Fr(3, 2), Fr(5, 2)])
def test_kasten_verschieben_um_halbe_stunden_in_bruechen(d):
    """Handbeispiel Kasten für halbe Stunden: Zellenwert = Überlapp mit dem verschobenen Kasten; W1 = d, TV = KS = d/3, KL = ∞ schon für d > 0."""
    p = kasten_bruch(0)
    q = [Fr(0)] * 24
    for k in range(24):
        a = max(Fr(k), Fr(6) + d)
        b = min(Fr(k + 1), Fr(9) + d)
        q[k] = max(Fr(0), b - a) / 3
    assert sum(q) == 1
    assert w1_bruch(p, q) == d and tv_bruch(p, q) == d / 3 and ks_bruch(p, q) == d / 3
    fq = PR.kasten(24, 6.0, 3.0, float(d))
    assert np.allclose(fq, [float(x) for x in q], atol=1e-15)
    assert A.kl(fq, [float(x) for x in p]) == math.inf and 0 < A.js([float(x) for x in p], fq) < math.log(2)


def test_kasten_von_hand_bei_d_gleich_1():
    """Gemeinsame Zellen 7 und 8, TV = 1/3; die Verteilungsfunktionen unterscheiden sich an drei Zellgrenzen um je 1/3: W1 = 3 · 1/3 · 1 h = 1 h."""
    assert tv_bruch(kasten_bruch(0), kasten_bruch(1)) == Fr(1, 3)
    f = np.cumsum([float(x) for x in kasten_bruch(0)])
    g = np.cumsum([float(x) for x in kasten_bruch(1)])
    diff = np.abs(f - g)
    assert np.count_nonzero(diff > 1e-12) == 3 and np.allclose(diff[diff > 1e-12], 1 / 3)


# ---- Handbeispiel 2: Treppe ---------------------------------------------------------------------------------------------------------

def test_treppe_in_bruechen_und_groesster_fehler_17_zwanzigstel():
    erwartet = {10: Fr(0), 20: Fr(0), 50: Fr(0), 100: Fr(1, 100), 200: Fr(1, 100), 500: Fr(12, 1000), 1000: Fr(13, 1000), 5000: Fr(136, 10000)}
    wahr = Fr(137, 10000)
    groesster = Fr(0)
    for b, soll in erwartet.items():
        z = E.treppe_zeile(b, 0.0137)
        assert Fr(z["w1"]).limit_denominator(100000) == soll and abs(z["fehler"]) <= 1 / b + 1e-15
        groesster = max(groesster, abs(soll - wahr) * b)
        assert z["fehler_h"] == pytest.approx(float(abs(soll - wahr) * b), abs=1e-12)
    assert groesster == Fr(17, 20)
    z10, z20, z50 = (E.treppe_zeile(b, 0.0137) for b in (10, 20, 50))
    assert z10["zelle1"] == z10["zelle2"] == 3 and z20["zelle1"] == z20["zelle2"] == 6 and z50["zelle1"] == z50["zelle2"] == 15
    z500, z100 = E.treppe_zeile(500, 0.0137), E.treppe_zeile(100, 0.0137)
    assert (z500["zelle1"], z500["zelle2"]) == (150, 156) and (z100["zelle1"], z100["zelle2"]) == (30, 31)


def test_treppe_rechnet_mit_exakten_zellennummern_bei_ganzzahligen_grenzen():
    assert E.treppe_zeile(1000, 0.0137)["zelle1"] == 300           # 0,3 · 1000 ist als Gleitkommazahl nicht sicher 300; die Bruchrechnung schon
    assert E.treppe_zeile(20, 0.1)["zelle2"] == 8


# ---- Handbeispiel 3: Tagesprofile mit zwei Spitzen, KL gegen Glättung ---------------------------------------------------------------

SOLL3 = {1: (1.000, 0.399, 0.498), 2: (2.000, 0.639, 1.964), 3: (3.000, 0.878, 4.199), 4: (4.000, 0.911, 6.640), 6: (5.999, 0.840, 11.160)}


@pytest.mark.parametrize("s", list(SOLL3))
def test_tagesprofile_zwei_spitzen_gegen_die_nachgerechneten_werte(s):
    a, b = PR.zwei_spitzen(24, 0.0, 1.0), PR.zwei_spitzen(24, float(s), 1.0)
    w1, tvv, kl = A.w1(a, b, 1.0), A.tv(a, b), A.kl(b, a, 1e-12)
    assert abs(w1 - SOLL3[s][0]) < 6e-4 and abs(tvv - SOLL3[s][1]) < 6e-4 and abs(kl - SOLL3[s][2]) < 6e-3


def test_tv_ist_bei_s_gleich_4_groesser_als_bei_s_gleich_6_und_der_ueberlapp_waechst():
    a = PR.zwei_spitzen(24, 0.0, 1.0)
    assert A.tv(a, PR.zwei_spitzen(24, 4.0, 1.0)) > A.tv(a, PR.zwei_spitzen(24, 6.0, 1.0))
    assert np.minimum(a, PR.zwei_spitzen(24, 6.0, 1.0)).sum() > np.minimum(a, PR.zwei_spitzen(24, 4.0, 1.0)).sum()


@pytest.mark.parametrize("g,k4,k6", [(0.0, 6.641, 11.410), (1e-12, 6.640, 11.160), (1e-9, 6.618, 10.127), (1e-6, 6.312, 7.486)])
def test_kl_haengt_ab_s_gleich_4_an_der_glaettung(g, k4, k6):
    a = PR.zwei_spitzen(24, 0.0, 1.0)
    assert abs(A.kl(PR.zwei_spitzen(24, 4.0, 1.0), a, g) - k4) < 2e-3 and abs(A.kl(PR.zwei_spitzen(24, 6.0, 1.0), a, g) - k6) < 2e-3


def test_bis_s_gleich_3_aendert_die_glaettung_1e_12_die_zahl_nicht():
    a = PR.zwei_spitzen(24, 0.0, 1.0)
    assert max(abs(A.kl(PR.zwei_spitzen(24, float(s), 1.0), a, 0.0) - A.kl(PR.zwei_spitzen(24, float(s), 1.0), a, 1e-12)) for s in (1, 2, 3)) < 1e-3


def test_die_zellenlage_ist_teil_der_konvention():
    """Punktproben an den Zellgrenzen statt an den Zellenmitten geben bei s = 1 einen anderen TV-Wert (0,352 statt 0,399)."""
    stunden = np.arange(24) + 0.5

    def prof(m1, m2):
        h = 0.4 * np.exp(-0.5 * (stunden - m1) ** 2) + 0.6 * np.exp(-0.5 * (stunden - m2) ** 2)
        return h / h.sum()

    assert A.tv(prof(6.0, 14.0), prof(7.0, 15.0)) == pytest.approx(0.352, abs=1e-3)
    assert A.tv(PR.zwei_spitzen(24, 0.0, 1.0), PR.zwei_spitzen(24, 1.0, 1.0)) == pytest.approx(0.399, abs=1e-3)


# ---- Handbeispiel 4: Gerade gegen Kreis ---------------------------------------------------------------------------------------------

def lp_transport(a, b, kosten):
    from scipy.optimize import linprog
    n, m = kosten.shape
    gl = np.zeros((n + m, n * m))
    for i in range(n):
        gl[i, i * m:(i + 1) * m] = 1
    for j in range(m):
        gl[n + j, j::m] = 1
    r = linprog(kosten.ravel(), A_eq=gl, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs")
    assert r.status == 0
    return float(r.fun)


def test_23_uhr_gegen_1_uhr_ist_22_stunden_auf_der_geraden_und_2_stunden_auf_dem_kreis():
    p, q = np.zeros(24), np.zeros(24)
    p[23], q[1] = 1.0, 1.0
    assert A.w1(p, q, 1.0) == pytest.approx(22.0) and A.w1_kreis(p, q, 1.0) == pytest.approx(2.0)
    idx = np.arange(24)
    dgl = np.abs(idx[:, None] - idx[None, :])
    assert lp_transport(p, q, dgl.astype(float)) == pytest.approx(22.0, abs=1e-9) and lp_transport(p, q, np.minimum(dgl, 24 - dgl).astype(float)) == pytest.approx(2.0, abs=1e-9)


def test_kreisformel_gegen_das_lp_auf_20_zufallspaaren():
    idx = np.arange(24)
    dgl = np.abs(idx[:, None] - idx[None, :])
    kreis = np.minimum(dgl, 24 - dgl).astype(float)
    g = np.random.default_rng(np.random.SeedSequence([9411]))
    worst, worst_roh = 0.0, 0.0
    for _ in range(20):
        pa, pb = g.dirichlet(np.ones(24) * 0.7), g.dirichlet(np.ones(24) * 0.7)
        pa, pb = np.where(pa < 1e-9, 0.0, pa), np.where(pb < 1e-9, 0.0, pb)
        pa, pb = pa / pa.sum(), pb / pb.sum()
        formel = A.w1_kreis(pa, pb, 1.0)
        worst = max(worst, abs(formel - lp_transport(pa * 1e6, pb * 1e6, kreis) / 1e6))     # HiGHS meldet bei Massen unter 1e-7 „infeasible“: mit 1e6 skalieren
        worst_roh = max(worst_roh, abs(formel - lp_transport(pa, pb, kreis)))
    assert worst < 1e-8 and worst_roh < 1e-6


# ---- Handbeispiel 7: KL ohne Unterschied (Aufzählung und Formel) ---------------------------------------------------------------------

def aufzaehlung(b, n):
    traeger = [frozenset(f) for f in itertools.product(range(b), repeat=n)]
    return Fr(sum(1 for sp in traeger for sq in traeger if not sp <= sq), len(traeger) ** 2)


@pytest.mark.parametrize("b,n", [(4, 3), (3, 4), (2, 5), (5, 3), (6, 2)])
def test_kl_unendlich_exakt_gleicht_der_aufzaehlung(b, n):
    assert E.kl_unendlich_exakt(b, n) == aufzaehlung(b, n)


def test_kl_unendlich_789_durch_1024_und_steigt_mit_den_zellen():
    assert E.kl_unendlich_exakt(4, 3) == Fr(789, 1024)
    assert abs(float(E.kl_unendlich_exakt(8, 3)) - 0.9602) < 5e-5 and abs(float(E.kl_unendlich_exakt(16, 3)) - 0.9942) < 5e-5
    assert E.kl_unendlich_exakt(4, 3) < E.kl_unendlich_exakt(8, 3) < E.kl_unendlich_exakt(16, 3)


def test_beispiel_von_hand_p_folge_0_0_1_gegen_q_folge_1_2_3():
    p = np.bincount([0, 0, 1], minlength=4) / 3
    q = np.bincount([1, 2, 3], minlength=4) / 3
    assert A.kl(p, q) == math.inf and A.kl(q, p) == math.inf       # Zelle 0 fehlt in q, Zelle 3 fehlt in p: beide Richtungen unendlich


def test_der_simulierte_anteil_gleicht_dem_exakten_bei_4_zellen_und_3_beobachtungen():
    """Unabhängige Gegenprobe der Histogramm-Zählung: gleichverteilte Beobachtungen, 4 Zellen, n = 3, 6 000 Wiederholungen; erwartet 789/1 024 (Standardfehler 0,0054)."""
    u = PR.zufall_gleich(5, 9, 6000, 6)
    hx, hy = E.histogramme(u[:, :3], 4), E.histogramme(u[:, 3:], 4)
    anteil = float(np.mean(np.any((hx > 0) & (hy == 0), axis=1)))
    assert abs(anteil - 789 / 1024) < 0.02
    assert np.array_equal(E.histogramme(np.array([[0.05, 0.1, 0.9]]), 4), [[2, 0, 0, 1]])


# ---- Handbeispiel 8: Rasterfehler von W2² -------------------------------------------------------------------------------------------

def test_rasterfehler_geteilt_durch_h_quadrat_ist_positiv_und_nahe_ein_sechstel_bei_verschobenen_zellen():
    for b, soll in ((25, 0.1658), (100, 0.1672), (400, 0.1660)):
        z = E.w2_raster_zeile(b, 0.7)
        assert z["fehler"] > 0 and abs(z["fehler_h2"] - soll) < 5e-4 and z["wahr"] == pytest.approx(2.34) and not z["deckungsgleich"]


def test_bei_ungleicher_streuung_ist_der_fehler_auch_bei_deckungsgleichen_zellen_nahe_ein_sechstel():
    """Die Lage der Zellen ist nicht die Ursache: bei Streuung 1 gegen 0,7 liegt Fehler / h² auch bei deckungsgleichen Zellen zwischen 0,15 und 0,17."""
    for b, soll in ((28, 0.1533), (112, 0.1643)):
        z = E.w2_raster_zeile(b, 0.7)
        assert z["deckungsgleich"] and z["fehler"] > 0 and abs(z["fehler_h2"] - soll) < 5e-4 and 0.15 < z["fehler_h2"] < 0.17


def test_bei_gleicher_streuung_und_nicht_deckungsgleichen_zellen_ist_der_fehler_nicht_ein_sechstel():
    werte = {b: E.w2_raster_zeile(b, 1.0)["fehler_h2"] for b in (25, 100, 400)}
    assert all(not E.w2_raster_zeile(b, 1.0)["deckungsgleich"] for b in werte)
    assert werte[25] == pytest.approx(0.2181, abs=5e-4) and werte[100] == pytest.approx(0.2041, abs=5e-4) and werte[400] == pytest.approx(0.1224, abs=5e-4)
    assert max(werte.values()) - min(werte.values()) > 0.09                    # schwankt mit der Zellenzahl


def test_bei_deckungsgleichen_zellen_und_gleicher_streuung_ist_der_fehler_null_bis_auf_das_abschneiden_der_raender():
    for b in (28, 112):
        z = E.w2_raster_zeile(b, 1.0)
        assert abs(z["verschiebung_in_h"] - round(z["verschiebung_in_h"])) < 1e-12 and z["deckungsgleich"] and abs(z["fehler_h2"]) < 1e-4 and z["wahr"] == pytest.approx(2.25)


def test_glatte_gauss_profile_gleicher_breite_sind_ab_50_zellen_auf_rundung_genau():
    assert E.glatt_zeile(10, 0.0137, 0.04)["fehler"] == pytest.approx(1.03e-3, abs=2e-5)
    assert abs(E.glatt_zeile(20, 0.0137, 0.04)["fehler"]) < 1e-7
    for b in (50, 100, 200, 500, 1000, 5000):
        assert abs(E.glatt_zeile(b, 0.0137, 0.04)["fehler"]) < 1e-13, b


# ---- Einheiten ----------------------------------------------------------------------------------------------------------------------

def test_w1_in_einheit_der_zellbreite_und_bei_h():
    assert A.w1([1, 0, 0], [0, 0, 1], 2.0) == pytest.approx(4.0)


def test_kl_mit_glaettung_und_leere_zelle():
    assert A.kl([0.5, 0.5], [1.0, 0.0]) == math.inf
    assert math.isfinite(A.kl([0.5, 0.5], [1.0, 0.0], 1e-12)) and A.kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))
    assert A.kl([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_js_ist_symmetrisch_und_hoechstens_ln_2():
    g = np.random.default_rng(3)
    for _ in range(30):
        p, q = g.dirichlet(np.ones(10) * 0.3), g.dirichlet(np.ones(10) * 0.3)
        assert A.js(p, q) == pytest.approx(A.js(q, p), abs=1e-14) and 0 <= A.js(p, q) <= math.log(2) + 1e-12


def test_randfaelle_eine_zelle_ein_dirac_gleiche_verteilung():
    assert A.w1([1.0], [1.0], 1.0) == 0.0 and A.tv([1.0], [1.0]) == 0.0 and A.ks([1.0], [1.0]) == 0.0 and A.w2_quadrat([0.5], np.array([1.0]), np.array([1.0])) == 0.0
    p = np.array([0.0, 1.0, 0.0])
    assert A.w1(p, p, 1.0) == 0 and A.w1_kreis(p, p, 1.0) == 0 and A.js(p, p) == 0


def test_kendall_gleicht_scipy():
    from scipy.stats import kendalltau
    g = np.random.default_rng(4)
    for _ in range(10):
        u, v = g.normal(size=40), g.normal(size=40)
        assert A.kendall_tau(u, v) == pytest.approx(kendalltau(u, v)[0], abs=1e-12)
    assert A.kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0) and A.kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
