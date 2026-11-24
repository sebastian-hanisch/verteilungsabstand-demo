"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed): Aussagen über die Abstände, die ohne bekanntes Ergebnis gelten müssen."""

import math

import numpy as np
import pytest

import vab_abstand as A
import vab_constants as C
import vab_evaluation as E
import vab_profile as PR
import vab_sinkhorn as S

ABSTAENDE = {"W1": lambda p, q: A.w1(p, q, 1.0), "TV": A.tv, "KS": A.ks, "JS": A.js}


def _paare(n=24, zahl=25, seed=77, dichte=0.5):
    g = np.random.default_rng(np.random.SeedSequence([seed]))
    return [(g.dirichlet(np.ones(n) * dichte), g.dirichlet(np.ones(n) * dichte)) for _ in range(zahl)]


@pytest.mark.parametrize("name", list(ABSTAENDE))
def test_symmetrie_und_selbstabstand_null(name):
    f = ABSTAENDE[name]
    for p, q in _paare():
        assert f(p, q) == pytest.approx(f(q, p), abs=1e-12) and f(p, p) == 0.0 and f(p, q) >= 0.0


def test_kl_ist_nichtnegativ_null_bei_gleichheit_und_nicht_symmetrisch():
    nicht_symmetrisch = 0
    for p, q in _paare():
        assert A.kl(p, q) >= -1e-15 and A.kl(p, p) == 0.0
        nicht_symmetrisch += abs(A.kl(p, q) - A.kl(q, p)) > 1e-6
    assert nicht_symmetrisch >= 20


def test_pinsker_tv_hoechstens_wurzel_kl_halbe():
    for p, q in _paare(zahl=40):
        assert A.tv(p, q) <= math.sqrt(A.kl(p, q) / 2) + 1e-12


def test_tv_liegt_zwischen_0_und_1_und_ist_1_bei_getrennten_traegern():
    for p, q in _paare():
        assert 0 <= A.tv(p, q) <= 1 and A.ks(p, q) <= A.tv(p, q) + 1e-12
    p, q = np.array([0.5, 0.5, 0, 0]), np.array([0, 0, 0.3, 0.7])
    assert A.tv(p, q) == 1.0 and A.ks(p, q) == 1.0 and A.js(p, q) == pytest.approx(math.log(2), abs=1e-15) and A.kl(p, q) == math.inf


def test_w1_ist_mindestens_der_unterschied_der_mittel():
    t = np.arange(24) + 0.5
    for p, q in _paare():
        assert A.w1(p, q, 1.0) >= abs(float(t @ p - t @ q)) - 1e-12


def test_gemeinsames_verschieben_beider_profile_aendert_w1_nicht():
    for a_s, b_s in ((0.0, 1.0), (0.0, 2.5), (1.0, 4.0), (2.0, 2.5)):
        for t in (0.5, 1.0, 3.0):
            w = A.w1(PR.kasten(48, 6.0, 3.0, a_s), PR.kasten(48, 6.0, 3.0, b_s), 0.5)
            assert w == pytest.approx(A.w1(PR.kasten(48, 6.0, 3.0, a_s + t), PR.kasten(48, 6.0, 3.0, b_s + t), 0.5), abs=1e-12) and w == pytest.approx(abs(b_s - a_s), abs=1e-12)


def test_spiegeln_der_zeitachse_aendert_w1_tv_ks_js_nicht():
    for p, q in _paare():
        for name, f in ABSTAENDE.items():
            assert f(p[::-1], q[::-1]) == pytest.approx(f(p, q), abs=1e-12), name


def test_w1_auf_dem_kreis_ist_hoechstens_w1_auf_der_geraden_und_drehinvariant():
    for p, q in _paare():
        assert A.w1_kreis(p, q, 1.0) <= A.w1(p, q, 1.0) + 1e-12
        assert A.w1_kreis(np.roll(p, 5), np.roll(q, 5), 1.0) == pytest.approx(A.w1_kreis(p, q, 1.0), abs=1e-12)


def test_quantilformel_w2_ist_symmetrisch_und_erfuellt_die_dreiecksungleichung_der_wurzel():
    prof, x, _, _, _ = E.menge4(3)
    g = np.random.default_rng(5)
    for _ in range(40):
        i, j, k = g.choice(24, size=3, replace=False)
        d = lambda a, b: math.sqrt(A.w2_quadrat(x, prof[a], prof[b]))
        assert d(i, j) == pytest.approx(d(j, i), abs=1e-12) and d(i, k) <= d(i, j) + d(j, k) + 1e-12


def test_kosten_und_eps_mit_demselben_faktor_skalieren_v_b_v_c_v_d_mit():
    prof = PR.profil_menge(1, 5, 24, C.ZWECK_MENGE)
    x = PR.zellenmitten(24)
    kosten = (x[:, None] - x[None, :]) ** 2
    kosten = kosten / kosten.max()
    w1 = S.werte_b_bis_e(prof, kosten, 0.06, tol=1e-12)
    w2 = S.werte_b_bis_e(prof, 3.0 * kosten, 0.18, tol=1e-12)
    for a, b in ((w1.kosten_plan, w2.kosten_plan), (w1.v_c, w2.v_c), (w1.v_d, w2.v_d), (w1.ot, w2.ot)):
        assert np.allclose(3.0 * a, b, rtol=1e-9, atol=1e-12)


def test_im_grenzfall_kleines_eps_ist_v_c_etwa_das_doppelte_von_v_d():
    """Band: Median V-c / V-d über die Paare bei ε/D = 0,3 liegt zwischen 1,9 und 2,2 (die Definitionen unterscheiden sich um den Faktor 2 gegen ½)."""
    res = E.analyse4(1, 0.3)
    c, d = res["matrizen"]["V-c"], res["matrizen"]["V-d"]
    v = [c[i, j] / d[i, j] for i, j in res["paare"]]
    assert 1.9 < float(np.median(v)) < 2.2


def test_gleiche_zufalls_seeds_gleiche_stichproben_und_zeilen_unabhaengig_von_der_zeilenzahl():
    u5, u10 = PR.zufall_gleich(11, 2, 5, 30), PR.zufall_gleich(11, 2, 10, 30)
    assert np.array_equal(u5, u10[:5])                              # Zeile j ist ein eigener Strom: unabhängig von der Zahl der Zeilen
    assert np.array_equal(E.reihe3(11, 100, 50, 0.02, 20)["w1_hist"], E.reihe3(11, 100, 50, 0.02, 20)["w1_hist"]) and not np.array_equal(u5, PR.zufall_gleich(12, 2, 5, 30))


def test_kendall_aendert_sich_nicht_unter_monotoner_umrechnung():
    g = np.random.default_rng(8)
    for _ in range(10):
        u, v = g.random(30), g.random(30)
        assert A.kendall_tau(u, v) == pytest.approx(A.kendall_tau(u, np.exp(3 * v)), abs=1e-12)


def test_ein_dirac_und_eine_zelle_als_randfaelle_der_experimente():
    z = E.treppe_zeile(1, 0.0137)
    assert z["w1"] == 0.0 and z["zelle1"] == z["zelle2"] == 0
    assert E.glatt_zeile(1, 0.0137, 0.04)["w1"] == pytest.approx(0.0, abs=1e-15)
    r = E.reihe3(3, 50, 1, 0.0, 10)
    assert r["anteil_xy"] == 0.0 and r["anteil_yx"] == 0.0 and np.all(r["w1_hist"] == 0.0)           # eine Zelle: beide Histogramme sind gleich
