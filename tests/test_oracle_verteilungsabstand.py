"""Orakel: unabhängige Rechenwege gegen die Kerne der Demo (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5).

- W1 und W2² auf dem Raster über Verteilungsfunktionen und Quantile gegen das Linearprogramm (HiGHS, Atome unter 1e-9 abgeschnitten, Massen mit 1e6 skaliert) und gegen scipy.
- Gauß: W2² = Lage² + Streuung² gegen die Quadratur der Quantilfunktionen; W1 und Cramér-Abstand gegen scipy.integrate.quad.
- Prognosen gegen empirische Verteilungen: W1, W2², CRPS und E|X − y| gegen numerische Integration mit scipy.
- Sinkhorn gegen das Dual (L-BFGS) steht in test_sinkhorn.py, V-a gegen SLSQP ebenda."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, wasserstein_distance

import vab_abstand as A
import vab_constants as C
import vab_evaluation as E
import vab_profile as PR
from test_abstand import lp_transport

N_LP_PAARE = 12


def _lp_paar(a, b, kosten):
    a = np.where(a < 1e-9, 0.0, a)
    b = np.where(b < 1e-9, 0.0, b)
    a, b = a / a.sum(), b / b.sum()
    sa, sb = a > 0, b > 0
    return lp_transport(a[sa] * 1e6, b[sb] * 1e6, kosten[np.ix_(sa, sb)]) / 1e6, a, b


def test_w2_quantilformel_gegen_das_lp_auf_zwoelf_paaren_der_standardmenge():
    prof, x, kosten, _, paare = E.menge4(1)
    worst = 0.0
    for (i, j) in paare[::23][:N_LP_PAARE]:
        v_lp, a, b = _lp_paar(prof[i], prof[j], kosten)
        worst = max(worst, abs(v_lp - A.w2_quadrat(x, a, b) / (x[-1] - x[0]) ** 2))
    assert worst < 1e-12, worst


def test_w1_verteilungsfunktionen_gegen_das_lp_und_gegen_scipy_auf_zufallspaaren():
    g = np.random.default_rng(np.random.SeedSequence([4411]))
    idx = np.arange(24, dtype=float)
    kosten = np.abs(idx[:, None] - idx[None, :])
    for _ in range(15):
        a, b = g.dirichlet(np.ones(24) * 0.6), g.dirichlet(np.ones(24) * 0.6)
        v_lp, a2, b2 = _lp_paar(a, b, kosten)
        assert A.w1(a2, b2, 1.0) == pytest.approx(v_lp, abs=1e-9)
        assert A.w1(a, b, 1.0) == pytest.approx(wasserstein_distance(idx, idx, a, b), abs=1e-12)


def test_w1_aus_sortierten_stichproben_gleicht_scipy():
    x, y = E.stichproben(2026, 200, 0.02, 5)
    z = E.reihe3(2026, 200, 50, 0.02, 5)
    for j in range(5):
        assert z["w1_sort"][j] == pytest.approx(wasserstein_distance(x[j], y[j]), abs=1e-12)


@pytest.mark.parametrize("m,s", [(0.5, 1.0), (0.0, 1.6), (0.0, 0.55), (0.3, 1.3), (-1.2, 0.7)])
def test_gauss_w2_ist_lage_quadrat_plus_streuung_quadrat_gegen_quadratur_der_quantile(m, s):
    quad_wert = quad(lambda u: (m + s * norm.ppf(u) - norm.ppf(u)) ** 2, 0, 1, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    assert m * m + (s - 1) ** 2 == pytest.approx(quad_wert, abs=1e-9)


def test_tabelle5_gegen_die_nachgerechnete_tabelle_und_quadratur():
    tab = E.tabelle5(E.Settings())
    soll_w1 = [0.0, 0.5, 0.4787, 0.3590, 0.3500]
    soll_w2 = [0.0, 0.25, 0.36, 0.2025, 0.18]
    soll_crps = [0.5642, 0.6340, 0.6027, 0.6003, 0.5970]
    soll_exy = [1.1284, 1.1982, 1.5054, 0.9106, 1.3305]
    for z, w1, w2, cr, ex in zip(tab, soll_w1, soll_w2, soll_crps, soll_exy):
        assert abs(z["w1"] - w1) < 6e-5 and abs(z["w2"] - w2) < 1e-12 and abs(z["ecrps"] - cr) < 6e-5 and abs(z["exy"] - ex) < 6e-5
        assert z["ecrps"] == pytest.approx(z["cramer2"] + 1 / math.sqrt(math.pi), abs=1e-8)       # E[CRPS] = Cramér² + 1/√π (L2-Norm der Verteilungsfunktionsdifferenz)
        m, s = z["lage"], z["sigma"]
        w1_quad = quad(lambda t: abs(norm.cdf(t, m, s) - norm.cdf(t, 0, 1)), -12, 12, limit=400, points=[m, 0.0])[0]
        assert z["w1"] == pytest.approx(w1_quad, abs=2e-8)
        exy_quad = quad(lambda y: norm.pdf(y) * quad(lambda t: abs(t - y) * norm.pdf(t, m, s), -14, 14, points=[y], limit=200)[0], -8, 8, limit=200)[0]
        assert z["exy"] == pytest.approx(exy_quad, abs=1e-7)
    assert E.rangfolge(tab, "w2") == ["F0", "F4", "F3", "F1", "F2"] and E.rangfolge(tab, "w1") == ["F0", "F4", "F3", "F2", "F1"]
    assert E.rangfolge(tab, "ecrps") == ["F0", "F4", "F3", "F2", "F1"] and E.rangfolge(tab, "exy") == ["F3", "F0", "F1", "F4", "F2"]
    assert tab[1]["w1"] == pytest.approx(0.5, abs=1e-8) and tab[2]["w1"] == pytest.approx(0.6 * math.sqrt(2 / math.pi), abs=1e-7)       # W1 geschlossen: |Lage| und |σ − 1| √(2/π)


def test_w1_und_w2_ordnen_lage_und_streuung_entgegengesetzt():
    tab = E.tabelle5(E.Settings())
    assert tab[1]["w1"] > tab[2]["w1"] and tab[1]["w2"] < tab[2]["w2"]


def _stichprobe(n, seed):
    return np.sort(np.random.default_rng(seed).normal(size=n) * 1.1 + 0.2)


@pytest.mark.parametrize("n,m,s", [(5, 0.0, 1.0), (7, 0.5, 1.6), (12, -0.3, 0.55)])
def test_w1_gegen_stichprobe_gleicht_der_numerischen_integration(n, m, s):
    y = _stichprobe(n, n)
    f = lambda t: abs(norm.cdf(t, m, s) - np.searchsorted(y, t, side="right") / n)
    quad_wert = quad(f, -15, 15, points=list(y), limit=500, epsabs=1e-12, epsrel=1e-12)[0]
    assert E.w1_gegen_stichprobe(y[None, :], m, s)[0] == pytest.approx(quad_wert, abs=1e-8)


@pytest.mark.parametrize("n,m,s", [(5, 0.0, 1.0), (7, 0.5, 1.6), (12, -0.3, 0.55)])
def test_w2_gegen_stichprobe_gleicht_der_numerischen_integration(n, m, s):
    y = _stichprobe(n, n + 100)
    f = lambda u: (m + s * norm.ppf(u) - y[min(int(u * n), n - 1)]) ** 2
    quad_wert = quad(f, 0, 1, points=[i / n for i in range(1, n)], limit=500, epsabs=1e-12, epsrel=1e-12)[0]
    assert E.w2_gegen_stichprobe(y[None, :], m, s)[0] == pytest.approx(quad_wert, abs=1e-8)


@pytest.mark.parametrize("m,s", [(0.0, 1.0), (0.5, 1.6), (-0.3, 0.55)])
def test_crps_und_exy_gegen_numerische_integration(m, s):
    y = _stichprobe(6, 9)[None, :]
    crps, exy = E.crps_und_exy(y, m, s)
    exy_ref = np.mean([quad(lambda t: abs(t - v) * norm.pdf(t, m, s), -20, 20, points=[v], limit=200)[0] for v in y[0]])
    crps_ref = np.mean([quad(lambda t: (norm.cdf(t, m, s) - (t >= v)) ** 2, -20, 20, points=[v], limit=200)[0] for v in y[0]])
    assert exy[0] == pytest.approx(exy_ref, abs=1e-8) and crps[0] == pytest.approx(crps_ref, abs=1e-8)


def test_ableitung_von_exy_nach_sigma_ist_2_phi_und_positiv():
    """Der Satz hinter „E|X − y| stellt die zu schmale Prognose vor die wahre“: ∂/∂σ E|σZ − y| = 2 φ(y/σ) > 0 für jedes y, gegen den Differenzenquotienten."""
    def exy(sig, yy):
        return sig * math.sqrt(2 / math.pi) * math.exp(-yy * yy / (2 * sig * sig)) + yy * (1 - 2 * norm.cdf(-yy / sig))

    for yy in (0.0, 0.3, 1.0, 2.5, -1.7):
        for sig in (0.55, 1.0, 1.6):
            hh = 1e-6
            ableitung = (exy(sig + hh, yy) - exy(sig - hh, yy)) / (2 * hh)
            assert ableitung == pytest.approx(2 * norm.pdf(yy / sig), abs=1e-7) and ableitung > 0


def test_f3_liegt_bei_exy_vor_f0_bei_jeder_von_50_stichproben():
    g = np.random.default_rng(np.random.SeedSequence([9413]))
    for _ in range(50):
        y = g.normal(size=25)[None, :]
        assert crps_exy(y, 0.55) < crps_exy(y, 1.0)


def crps_exy(y, s):
    return float(E.crps_und_exy(y, 0.0, s)[1][0])


def test_mc5_zaehlt_wie_eine_einfache_schleife():
    """Zählung der Zufalls-Seeds, in denen F3 vor F0 liegt, und der Platz-1-Anteile gegen eine Schleife über die Zeilen mit den Einzelfunktionen."""
    s = E.Settings(nobs5=25, seeds5=40)
    mc = E.mc5(s)
    y = np.sort(PR.zufall_normal(s.seed_input, C.ZWECK_PROGNOSE, 40, 25), axis=1)
    prog = E.prognosen(s)
    platz, f3 = 0, 0
    for j in range(40):
        werte = [float(E.crps_und_exy(y[j:j + 1], m, sg)[1][0]) for _, m, sg in prog]
        f3 += werte[3] < werte[0]
        platz += int(np.argmin(werte) == 0)
    assert mc["f3_vor_f0"]["E|X − y|"] == f3 == 40 and mc["platz1"]["E|X − y|"] == platz / 40 == 0.0


def test_die_wahre_prognose_liegt_mit_mehr_beobachtungen_haeufiger_auf_platz_1():
    anteile = [E.mc5(E.Settings(nobs5=n))["platz1"] for n in C.NOBS5]
    for regel in ("W2²", "W1", "E[CRPS]"):
        assert anteile[0][regel] < anteile[1][regel] <= anteile[2][regel]
        assert 0.6 < anteile[0][regel] < 0.9 and anteile[2][regel] > 0.99
    assert all(a["E|X − y|"] == 0.0 for a in anteile)
