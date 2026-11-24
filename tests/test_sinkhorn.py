"""Sinkhorn im Stapel und die Verwendungen V-a bis V-e: Handbeispiel 5 (Selbstabstand), Orakel gegen ein unabhängiges Dual (L-BFGS) und die Nebenbedingung gegen SLSQP, Stapel gegen Einzellauf."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

import vab_abstand as A
import vab_constants as C
import vab_evaluation as E
import vab_profile as PR
import vab_sinkhorn as S

ROOT = Path(__file__).resolve().parent.parent
KOSTEN3 = (np.arange(3.0)[:, None] - np.arange(3.0)[None, :]) ** 2 / 4
A3 = np.array([1 / 3, 1 / 3, 1 / 3])
B3 = np.array([0.4, 0.2, 0.4])


def einzel(a, b, kosten, eps, tol=1e-13):
    st = S.sinkhorn_stapel(a[None, :], b[None, :], kosten, eps, tol=tol)
    assert st.konvergiert.all()
    return S.plaene(st)[0], st


def dual_lbfgs(a, b, kosten, eps):
    """Unabhängiger Rechenweg: die duale Funktion der weichen Aufgabe mit L-BFGS (keine Sinkhorn-Iteration)."""
    n, m = kosten.shape

    def f(z):
        e = np.exp((z[:n, None] + z[None, n:] - kosten) / eps)
        return eps * e.sum() - z[:n] @ a - z[n:] @ b, np.concatenate([e.sum(1) - a, e.sum(0) - b])

    r = minimize(f, np.zeros(n + m), jac=True, method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-16, "gtol": 1e-13})
    return np.exp((r.x[:n, None] + r.x[None, n:] - kosten) / eps)


# ---- Handbeispiel 5 --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [1.0, 0.5, 0.2])
def test_selbstabstand_2x2_geschlossen_q_durch_1_plus_q(eps):
    a2 = np.array([0.5, 0.5])
    c2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    p, _ = einzel(a2, a2, c2, eps, 1e-15)
    q = math.exp(-1 / eps)
    assert float((p * c2).sum()) == pytest.approx(q / (1 + q), abs=1e-14)
    assert math.exp(-1) / (1 + math.exp(-1)) == pytest.approx(1 / (1 + math.e), abs=1e-16) and 1 / (1 + math.e) == pytest.approx(0.2689, abs=5e-5)


def test_drei_zellen_die_andere_verteilung_ist_naeher_als_die_verteilung_selbst():
    assert A.w2_quadrat(np.arange(3.0) / 2, A3, B3) == pytest.approx(1 / 30, abs=1e-12)
    eps = 0.25
    p_ab, _ = einzel(A3, B3, KOSTEN3, eps)
    p_aa, _ = einzel(A3, A3, KOSTEN3, eps)
    k_ab, k_aa = float((p_ab * KOSTEN3).sum()), float((p_aa * KOSTEN3).sum())
    assert k_ab == pytest.approx(0.08553, abs=5e-5) and k_aa == pytest.approx(0.08693, abs=5e-5) and k_ab < k_aa
    assert np.allclose(p_ab, dual_lbfgs(A3, B3, KOSTEN3, eps), atol=1e-8) and np.allclose(p_aa, dual_lbfgs(A3, A3, KOSTEN3, eps), atol=1e-8)
    kl = float(np.sum(p_ab * np.log(p_ab / np.outer(A3, B3))))
    assert kl == pytest.approx(0.3851, abs=5e-4) and 0 < kl < math.log(3)


def test_faktor_2_gegen_faktor_halb_bei_drei_zellen_und_eps_0_02():
    """Genevay-Verlust / W2² = 2,00, Feydy S_ε / W2² = 0,80, beide zueinander 2,51; S_ε(a, a) = 0 genau."""
    eps = 0.02
    w2 = 1 / 30
    prof = np.array([A3, B3])
    paare = [(0, 1)]
    ergebnis = S.werte_b_bis_e(prof, KOSTEN3, eps, tol=1e-13)
    g, s = ergebnis.v_c[0, 1], ergebnis.v_d[0, 1]
    assert g / w2 == pytest.approx(2.0, abs=0.01) and 0.75 < s / w2 < 0.85 and 2.4 < g / s < 2.6 and paare == [(0, 1)]
    assert ergebnis.v_d[0, 0] == 0.0 and ergebnis.v_c[1, 1] == 0.0


def test_nebenbedingung_ist_die_lagrange_form_gleicht_slsqp_auf_6_paaren():
    """V-a gegen einen unabhängigen Löser (SLSQP, Zulässigkeit selbst geprüft): SLSQP liegt nie unter der Lagrange-Lösung (sie ist zulässig und optimal)."""
    g = np.random.default_rng(np.random.SeedSequence([9402]))
    x5 = (np.arange(5) + 0.5) / 5
    c5 = (x5[:, None] - x5[None, :]) ** 2
    c5 = c5 / c5.max()
    alpha = 0.5
    geprueft = 0
    for _ in range(6):
        a, b = g.dirichlet(np.ones(5)), g.dirichlet(np.ones(5))
        mat, unwirksam, nk = S.wert_nebenbedingung(np.array([a, b]), c5, alpha, tol=1e-12, eps_min=0.002, eps_max=50.0, schritte=50)
        v_lagrange = mat[0, 1]
        ref = np.outer(a, b).ravel()
        zeilen = np.zeros((5, 25))
        for i in range(5):
            zeilen[i, i * 5:(i + 1) * 5] = 1
        spalten = np.zeros((4, 25))
        for j in range(4):
            spalten[j, j::5] = 1
        aeq, beq = np.vstack([zeilen, spalten]), np.concatenate([a, b[:4]])
        cons = [{"type": "eq", "fun": lambda p: aeq @ p - beq, "jac": lambda p: aeq},
                {"type": "ineq", "fun": lambda p: alpha - float(np.sum(p * np.log(p / ref))), "jac": lambda p: -(np.log(p / ref) + 1.0)}]
        assert nk == 0 and unwirksam == 0
        beste = None
        for lam in (0.0, 0.3, 0.6, 0.85):                    # mehrere Startpunkte: SLSQP-Ergebnisse hängen von der Plattform ab (CI-Lauf 2026-11-24: auf einem Runner kein zulässiger Start)
            p0 = (1.0 - lam) * ref + lam * np.full(25, 1.0 / 25)
            r = minimize(lambda p: float((p.reshape(5, 5) * c5).sum()), p0, jac=lambda p: c5.ravel(), constraints=cons, bounds=[(1e-12, 1)] * 25, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-15})
            pm = r.x.reshape(5, 5)
            if np.abs(pm.sum(1) - a).max() < 1e-6 and np.abs(pm.sum(0) - b).max() < 1e-6 and float(np.sum(r.x * np.log(r.x / ref))) <= alpha + 1e-6:
                beste = r.fun if beste is None else min(beste, r.fun)
        if beste is None:
            continue                                         # kein zulässiger SLSQP-Lauf: dieses Paar bleibt ungeprüft (unten: mindestens 4 von 6 verglichen)
        geprueft += 1
        assert beste - v_lagrange > -1e-8 and beste - v_lagrange < 1e-4       # SLSQP nie unter der Lagrange-Lösung, darüber nur Restfehler des Lösers
    assert geprueft >= 4, geprueft


# ---- Stapel gegen Einzellauf, Orakel ------------------------------------------------------------------------------------------------

def test_stapel_gleicht_dem_einzellauf_und_dem_dual_und_haengt_nicht_von_den_anderen_paaren_ab():
    prof = PR.profil_menge(1, 6, 24, C.ZWECK_MENGE)
    x = PR.zellenmitten(24)
    kosten = (x[:, None] - x[None, :]) ** 2
    kosten = kosten / kosten.max()
    eps = 0.05
    prof = 0.9 * prof + 0.1 / 24                  # strikt positive Massen: das Dual per L-BFGS konvergiert bei Massen um 1e-12 nicht
    a, b = prof[:5], prof[1:6]
    st = S.sinkhorn_stapel(a, b, kosten, eps, tol=1e-12)
    pl = S.plaene(st)
    for i in range(5):
        ein = S.sinkhorn_stapel(a[i:i + 1], b[i:i + 1], kosten, eps, tol=1e-12)
        assert np.allclose(S.plaene(ein)[0], pl[i], rtol=1e-9, atol=1e-18) and abs(int(ein.iterationen[0]) - int(st.iterationen[i])) <= 1       # eingefroren: bis auf die Summenreihenfolge der Matrixprodukte gleich
        assert np.allclose(pl[i], dual_lbfgs(a[i], b[i], kosten, eps), atol=1e-8)
        assert np.allclose(pl[i].sum(1), a[i], atol=1e-11) and np.allclose(pl[i].sum(0), b[i], atol=1e-14)
    assert st.konvergiert.all() and np.array_equal(st.produkte, 2 * st.iterationen)


def test_eps_je_paar_gleicht_dem_gemeinsamen_eps():
    prof = PR.profil_menge(1, 4, 24, C.ZWECK_MENGE)
    x = PR.zellenmitten(24)
    kosten = (x[:, None] - x[None, :]) ** 2
    kosten = kosten / kosten.max()
    a, b = prof[:3], prof[1:4]
    gemeinsam = S.plaene(S.sinkhorn_stapel(a, b, kosten, 0.07, tol=1e-12))
    je_paar = S.plaene(S.sinkhorn_stapel(a, b, kosten, np.array([0.07, 0.07, 0.07]), tol=1e-12))
    assert np.allclose(gemeinsam, je_paar, atol=1e-15, rtol=0)


def test_eine_iteration_sind_zwei_produkte_beim_handbeispiel_mit_toleranz_unendlich():
    st = S.sinkhorn_stapel(A3[None, :], A3[None, :], KOSTEN3, 0.5, tol=1e9)
    assert st.iterationen[0] == 1 and st.produkte[0] == 2 and st.konvergiert[0]


def test_unterlauf_wird_gemeldet_statt_still_falsch_zu_rechnen():
    with pytest.raises(ValueError):
        S.kern(np.array([[0.0, 1.0], [1.0, 0.0]]), 1 / 800)
    assert S.kern(np.array([[0.0, 1.0], [1.0, 0.0]]), 1 / 600).min() > 1e-300


def test_nicht_konvergierte_paare_werden_gemeldet():
    st = S.sinkhorn_stapel(A3[None, :], B3[None, :], KOSTEN3, 0.05, tol=1e-30, max_iter=50)
    assert not st.konvergiert[0] and st.iterationen[0] == 50


def test_zentrierte_verwendungen_haben_selbstabstand_0_und_V_b_V_e_nicht():
    prof = PR.profil_menge(1, 6, 48, C.ZWECK_MENGE)
    x = PR.zellenmitten(48)
    kosten = (x[:, None] - x[None, :]) ** 2
    kosten = kosten / kosten.max()
    w = S.werte_b_bis_e(prof, kosten, 0.05)
    assert np.all(np.diag(w.v_c) == 0) and np.all(np.diag(w.v_d) == 0) and np.all(np.diag(w.kosten_plan) > 0) and np.all(np.diag(w.ot) > 0)
    assert np.allclose(w.v_c, w.v_c.T) and np.allclose(w.v_d, w.v_d.T) and w.nicht_konvergiert == 0


def test_dreiecksverletzungen_zaehlt_an_einem_handbeispiel():
    # drei Punkte auf einer Geraden mit d = Wurzel(Wert): Werte 1, 1, 9 (d = 1, 1, 3): 3 > 1 + 1 verletzt; Werte 1, 1, 4 (d = 1, 1, 2) nicht
    m = np.array([[0, 1, 9], [1, 0, 1], [9, 1, 0]], dtype=float)
    assert S.dreiecksverletzungen(m) == (1, 1, 0)
    m2 = np.array([[0, 1, 4], [1, 0, 1], [4, 1, 0]], dtype=float)
    assert S.dreiecksverletzungen(m2) == (0, 1, 0)
    m3 = np.array([[0, -1, 4], [-1, 0, 1], [4, 1, 0]], dtype=float)
    assert S.dreiecksverletzungen(m3)[2] == 1
    assert S.dreiecksverletzungen(np.zeros((2, 2))) == (0, 0, 0)


def test_exaktes_w2_verletzt_die_dreiecksungleichung_nie_und_alle_werte_sind_nichtnegativ():
    """Kontrolle der Zählung: das exakte W2 ist eine Metrik (0 von 2 024 Tripeln) in fünf Verteilungsmengen; V-a und V-b sind in der Standardmenge ohne Verletzung."""
    for menge in range(1, 6):
        _, _, _, exakt, paare = E.menge4(menge)
        mat = np.zeros((24, 24))
        for (i, j), v in zip(paare, exakt):
            mat[i, j] = mat[j, i] = v
        assert S.dreiecksverletzungen(mat) == (0, 2024, 0), menge
    res = E.analyse4(1, 3.0)
    for name in ("V-a", "V-b"):
        assert res["kennzahlen"][name]["verletzt"] == 0 and res["kennzahlen"][name]["negativ"] == 0
    for name in C.VERW4:
        assert res["kennzahlen"][name]["negativ"] == 0


def test_kopie_gegen_die_quelle_sinkhorn_demo_gleicher_plan():
    """Querprobe gegen die Log-Domänen-Iteration der Demo `sinkhorn-demo` (andere Rechenform, andere Datei), wenn der Ordner vorhanden ist."""
    quelle = ROOT.parent / "sinkhorn-demo" / "snk_sinkhorn.py"
    if not quelle.exists():
        pytest.skip("sinkhorn-demo nicht vorhanden")
    import importlib.util
    import sys
    sys.path.insert(0, str(quelle.parent))
    try:
        spec = importlib.util.spec_from_file_location("snk_sinkhorn_quelle", quelle)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(quelle.parent))
    prof = PR.profil_menge(1, 3, 24, C.ZWECK_MENGE)
    x = PR.zellenmitten(24)
    kosten = (x[:, None] - x[None, :]) ** 2
    kosten = kosten / kosten.max()
    ref = mod.loese_sinkhorn(prof[0], prof[1], kosten, 0.05, tol=1e-12, max_iter=100000)
    plan_ref = mod.plan_aus_potentialen(prof[0], prof[1], kosten, 0.05, ref.f, ref.g) if hasattr(ref, "f") else None
    eigen = S.plaene(S.sinkhorn_stapel(prof[:1], prof[1:2], kosten, 0.05, tol=1e-12))[0]
    if plan_ref is not None:
        assert np.allclose(eigen, plan_ref, atol=1e-9)
    else:
        pytest.skip("Schnittstelle der Quelle unbekannt")


def test_unwirksame_nebenbedingung_zeigt_den_wert_beim_kleinsten_eps_als_obere_schranke():
    """Ist KL schon beim kleinsten ε' unter α, steht ⟨C, P_ε'⟩ bei ε' = EPS_MIN_A in der Matrix: der Plan ist zulässig, der Wert also eine obere Schranke des Optimums unter der Nebenbedingung
    (dieses liegt zwischen W2² = 1/30 und dem Wert); die Maske nennt genau diese Paare."""
    prof = np.array([A3, B3])
    mat, unwirksam, nk, maske = S.wert_nebenbedingung(prof, KOSTEN3, 50.0, mit_maske=True)
    plan, _ = einzel(A3, B3, KOSTEN3, S.EPS_MIN_A)
    assert unwirksam == 1 and maske.tolist() == [True] and S.EPS_MIN_A == 0.003
    assert mat[0, 1] == pytest.approx(float((plan * KOSTEN3).sum()), abs=1e-7) and mat[0, 1] >= 1 / 30 - 1e-9
    mat2, wirksam, nk2, maske2 = S.wert_nebenbedingung(prof, KOSTEN3, 1e-6, mit_maske=True)
    assert wirksam == 0 and maske2.tolist() == [False] and mat2[0, 1] > mat[0, 1]          # bei wirksamer Nebenbedingung wird ε' groß und der Wert größer
    assert len(S.wert_nebenbedingung(prof, KOSTEN3, 0.5)) == 3                                           # ohne mit_maske bleibt die Rückgabe dreiteilig
