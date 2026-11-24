"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a).

Dieser Test macht vier Dinge:
(1) die Tabellen und Sätze der README werden aus dem Code der Demo mit denselben Formatfunktionen wie in der App gebildet (`tests/readme_zeilen.py`) und müssen wörtlich in der README stehen;
(2) die Aussagen, die die README über diese Zahlen macht („in allen Mengen“, „keine Verletzung“), werden an der ganzen Reihe der Datei `tools/messreihen.json` gezählt;
(3) die Menge 1 der Datei wird neu gerechnet und als Band verglichen (Zählungen ±1 bis ±2, Kendall ±0,02), weil Gleitkomma-Summen auf anderen Plattformen abweichen können;
(4) Konstanten, Kantenwortlaut, Testzahl und verbotene Wörter."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import readme_zeilen as R
import vab_constants as C
import vab_evaluation as E

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
MESS = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))
DATEN = {c: MESS["streuung"][c]["daten"] for c in MESS["streuung"]}


def _zellen(verw):
    """Alle 20 Zellen (4 ε/D × 5 Mengen) einer Verwendung."""
    return [(c, m, DATEN[c][m][verw]) for c in DATEN for m in DATEN[c]]


def test_tabellen_und_saetze_der_readme():
    for name, f in R.STUECKE.items():
        assert f() in README, name


def test_konstanten_der_readme():
    for text in ("24 Tagesprofile aus Mischungen von 1 bis 3 Gauß-Hügeln", "auf 48 Zellen à 30 Minuten, 276 Paare, 2 024 Tripel", "Zufalls-Seed 2026", "höchstens 200 000 Durchläufe",
                 "Zeilenfehler unter 1e-10 (V-a: 1e-9)", "Verschiebung 0 bis 8 Stunden in Schritten von 0,5", "n = 500", "300 Zufalls-Seeds", "400 Zufalls-Seeds"):
        assert text in README, text
    assert C.ANZAHL4 == 24 and C.ZELLEN4 == 48 and C.MAX_ITER4 == 200_000 and C.TOL4 == 1e-10 and C.TOL4_A == 1e-9 and C.SERIEN_SEED == 2026 and len(C.S1_WERTE) == 17 and C.S1_SCHRITT == 0.5
    assert C.SEEDS3[-1] == 300 and C.SEEDS5[-1] == 400 and C.DEFAULTS["n3"] == 500 and len(C.VERW4) == 5 and min(C.C4) == 0.3


def test_testzahl_der_readme():
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    zahl = int(re.search(r"(\d+) tests? collected", r.stdout).group(1))
    assert f"- **Testzahl:** {zahl} Tests (`pytest --collect-only -q tests`)." in README, zahl


def test_readme_und_app_nennen_keine_planbegriffe_und_keine_umgebungsaussagen():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (README, app):
        for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Saatwert", "trägt", "tragen", "Vormessung", "Mutationstest", "Stück", "V1", "O6", "TODO ", "Plan-", "PLAN-",
                         "wasserstein-demo", "Neuheit ist", "erstmals"):
            if verboten == "TODO " and text is README:
                continue                                                # der Live-Link-Kommentar bleibt bis zum Deploy
            assert verboten not in text, verboten
    assert "Orakel" not in app


def test_readme_nennt_die_kanten_im_wortlaut():
    for satz in ("*Lager, Kunden und Kosten sind gegeben; der Optimalwert ist das Ende der Rechnung, ein Preis für einen Auftrag.*", "*⟨C, P_ε⟩ ist eine Näherung des Transportwerts, die mit ε → 0 besser wird.*",
                 "*Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle.*",
                 "*Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1.*"):
        assert satz in README, satz
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for satz in ("Lager, Kunden und Kosten sind gegeben; der Optimalwert ist das Ende der Rechnung, ein Preis für einen Auftrag.", "⟨C, P_ε⟩ ist eine Näherung des Transportwerts, die mit ε → 0 besser wird.",
                 "Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle.",
                 "Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1."):
        assert satz in app, satz


def test_die_zahl_zu_test_tabelle_nennt_vorhandene_tests():
    for datei, name in (("test_claims.py", "test_tabellen_und_saetze_der_readme"), ("test_claims.py", "test_konstanten_der_readme")):
        assert f"def {name}" in (ROOT / "tests" / datei).read_text(encoding="utf-8")
    for datei in ("test_abstand.py", "test_sinkhorn.py", "test_oracle_verteilungsabstand.py", "test_profile.py", "test_eigenschaften.py"):
        assert (ROOT / "tests" / datei).exists() and f"`tests/{datei}" in README, datei


def test_zaehlungen_der_verifikation_stehen_in_den_tests():
    quelle = {d: (ROOT / "tests" / d).read_text(encoding="utf-8") for d in ("test_oracle_verteilungsabstand.py", "test_abstand.py", "test_sinkhorn.py")}
    assert "N_LP_PAARE = 12" in quelle["test_oracle_verteilungsabstand.py"] and "range(15)" in quelle["test_oracle_verteilungsabstand.py"] and "range(20)" in quelle["test_abstand.py"]
    assert "for i in range(5)" in quelle["test_sinkhorn.py"] and "for _ in range(6)" in quelle["test_sinkhorn.py"] and '[(4, 3), (3, 4), (2, 5), (5, 3), (6, 2)]' in quelle["test_abstand.py"]
    for text in ("auf 12 Paaren", "auf 15 Zufallspaaren", "auf 20 Zufallspaaren", "(5 Paare, 1e-8)", "(6 Zufallsinstanzen mit 5 Zellen", "5 Instanzen"):
        assert text in README, text


# ---- Aussagen über die Reihe der fünf Mengen ----------------------------------------------------------------------------------------------

def test_mengen_der_datei_sind_1_bis_5_und_vier_eps_durch_d():
    assert set(DATEN) == {repr(c) for c in (8.0, 3.0, 1.0, 0.3)} and all(set(DATEN[c]) == {"1", "2", "3", "4", "5"} for c in DATEN)
    assert len(MESS["d_mengen"]) == 20


def test_nicht_zentrierte_verwendungen_haben_in_allen_zwanzig_zellen_einen_selbstabstand_ueber_null():
    for verw in ("V-a", "V-b", "V-e"):
        assert all(z["selbst_mittel"] > 0 for _, _, z in _zellen(verw)), verw


def test_zentrierte_verwendungen_haben_in_allen_zellen_selbstabstand_null_und_keine_verteilung_mit_naeherer_anderer():
    for verw in ("V-c", "V-d"):
        assert all(z["selbst_mittel"] == 0 and z["nicht_naechster"] == 0 for _, _, z in _zellen(verw)), verw


def test_bei_eps_durch_d_gleich_8_hat_in_jeder_menge_mehr_als_die_haelfte_eine_naehere_andere():
    for verw in ("V-a", "V-b", "V-e"):
        assert all(DATEN["8.0"][m][verw]["nicht_naechster"] > 12 for m in DATEN["8.0"]), verw


def test_v_c_und_v_d_ordnen_in_jeder_zelle_besser_als_v_b_und_v_e():
    for c in DATEN:
        for m in DATEN[c]:
            z = DATEN[c][m]
            assert min(z["V-c"]["kendall"], z["V-d"]["kendall"]) > max(z["V-b"]["kendall"], z["V-e"]["kendall"]), (c, m)


def test_v_c_liegt_in_zehn_von_zehn_zellen_vor_v_d_und_die_differenz_ist_bei_kleinem_eps_klein():
    gross = [DATEN[c][m]["V-c"]["kendall"] - DATEN[c][m]["V-d"]["kendall"] for c in ("8.0", "3.0") for m in DATEN[c]]
    klein = [DATEN[c][m]["V-c"]["kendall"] - DATEN[c][m]["V-d"]["kendall"] for c in ("1.0", "0.3") for m in DATEN[c]]
    assert min(gross) > 0 and min(klein) > 0 and max(klein) < min(gross)


def test_dreiecksverletzungen_von_v_a_v_b_v_e_sind_null_und_von_v_c_v_d_nicht():
    for verw in ("V-a", "V-b", "V-e"):
        assert all(z["verletzt"] == 0 and z["negativ"] == 0 for _, _, z in _zellen(verw)), verw
    assert any(z["verletzt"] > 0 for _, _, z in _zellen("V-c")) and any(z["verletzt"] > 0 for _, _, z in _zellen("V-d"))
    assert max(z["verletzt"] for _, _, z in _zellen("V-c") + _zellen("V-d")) < 40


def test_v_d_unterschaetzt_bei_grossem_eps_und_v_c_ueberschaetzt():
    for m in DATEN["8.0"]:
        assert DATEN["8.0"][m]["V-d"]["verhaeltnis"] < 1.0 < DATEN["8.0"][m]["V-c"]["verhaeltnis"]


def test_v_c_und_v_d_sind_bei_kleinem_eps_etwa_das_doppelte_und_das_einfache_von_w2():
    for m in DATEN["0.3"]:
        assert 1.95 < DATEN["0.3"][m]["V-c"]["verhaeltnis"] < 2.05 and 0.97 < DATEN["0.3"][m]["V-d"]["verhaeltnis"] < 1.01


def test_d_haengt_an_der_menge_in_den_zwanzig_mengen():
    d = [MESS["d_mengen"][str(m)] for m in range(1, 21)]
    assert 2.0 < max(d) / min(d) < 2.5


@pytest.mark.parametrize("c", [8.0, 1.0])
def test_menge_1_der_datei_neu_gerechnet_im_band(c):
    res = E.analyse4(1, c)
    for verw in C.VERW4:
        neu, alt = res["kennzahlen"][verw], DATEN[repr(c)]["1"][verw]
        assert abs(neu["kendall"] - alt["kendall"]) < 0.02 and abs(neu["nicht_naechster"] - alt["nicht_naechster"]) <= 2 and abs(neu["verhaeltnis"] - alt["verhaeltnis"]) < 0.05, (verw, neu, alt)
        assert abs(neu["verletzt"] - alt["verletzt"]) <= 3 and abs(neu["selbst_mittel"] - alt["selbst_mittel"]) < 1e-4
    assert abs(res["d"] - DATEN[repr(c)]["1"]["d"]) < 1e-9


def test_alle_laeufe_von_v_b_bis_v_e_konvergieren_und_v_a_hat_hoechstens_zwei_unfertige_laeufe_je_zelle():
    assert all(DATEN[c][m]["nicht_konvergiert"] == 0 for c in DATEN for m in DATEN[c])
    assert all(DATEN[c][m]["nicht_konvergiert_a"] <= 2 for c in DATEN for m in DATEN[c])


# ---- Befunde der Aussagenprüfung ---------------------------------------------------------------------------------------------------------

def test_v_c_und_v_d_ordnen_in_jeder_zelle_auch_besser_als_v_a():
    for c in DATEN:
        for m in DATEN[c]:
            z = DATEN[c][m]
            assert min(z["V-c"]["kendall"], z["V-d"]["kendall"]) > z["V-a"]["kendall"], (c, m)


def test_d_der_mengen_unterscheidet_sich_um_den_faktor_2_2():
    d = [MESS["d_mengen"][str(m)] for m in range(1, 21)]
    assert f"{max(d) / min(d):.1f}" == "2.2" and "(Faktor 2,2)" in README and "Faktor etwa 2" not in README


def test_v_a_bei_unwirksamer_nebenbedingung_zeigt_den_wert_beim_kleinsten_eps_und_die_readme_nennt_es():
    """Menge 1, ε/D = 0,3: 18 Paare mit unwirksamer Nebenbedingung (Band ± 2); dort steht ⟨C, P_ε'⟩ bei ε' = 0,003 = 0,11 D, im Median das 1,04-Fache von W2², höchstens das 10,97-Fache (Band)."""
    res = E.analyse4(1, 0.3)
    assert abs(res["unwirksam_a"] - 18) <= 2 and abs(res["unwirksam_median"] - 1.04) < 0.05 and abs(res["unwirksam_max"] - 10.97) < 1.0 and res["unwirksam_median"] >= 1.0
    assert E.zahl(res["eps_min_durch_d"], 2) == "0,11" and E.unwirksam_satz(res) in README
    assert "obere Schranke des Werts unter der Nebenbedingung" in README
    res8 = E.analyse4(1, 8.0)
    assert res8["unwirksam_a"] == 0 and res8["unwirksam_median"] is None and "obere Schranke" not in E.unwirksam_satz(res8)


def test_readme_erklaert_v_d_mit_dem_gewicht_des_kreuzterms_und_nicht_mit_dem_faktor_einhalb():
    assert "(die Faktoren 2 und ½ der Definitionen)" not in README
    assert "V-d: der Kreuzterm hat das Gewicht 1, die Hälften stehen nur vor den Selbsttermen" in README and "V-c: der Faktor 2 der Definition steht vor dem Kreuzterm" in README


def test_readme_nennt_den_erwarteten_crps_als_quadrat_der_l2_norm_plus_konstante():
    assert "Der erwartete CRPS ist die L2-Norm" not in README and "Quadrat der L2-Norm der Verteilungsfunktionsdifferenz plus eine Konstante (1/√π = 0,5642" in README
    assert "Quadrat der L2-Norm der Verteilungsfunktionsdifferenz $F - G$ plus eine Konstante" in (ROOT / "app.py").read_text(encoding="utf-8")


def test_readme_und_app_sagen_zur_konstante_ein_sechstel_nicht_mehr_nur_verschobene_zellen():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (README, app):
        assert "nur für gegeneinander verschobene Zellen" not in text
    assert "bei ungleicher Streuung" in app and "auch bei deckungsgleichen Zellen" in app and "Für gleiche Streuung erklärt sie nichts" in README


def test_readme_nennt_die_grenzen_der_verifikation_und_der_plattform_genau():
    assert "(6 Zufallsinstanzen mit 5 Zellen, α = 0,5 und erweitertem ε-Bereich von 0,002 bis 50, nicht in der Konfiguration der App;" in README and "(6 Paare;" not in README
    assert "ohne Einfluss auf die Zählungen" not in README and "auf einer zweiten Plattform nicht geprüft" in README
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "ohne Einfluss auf Zählungen" not in app and "auf einer zweiten Plattform nicht geprüft" in app
    assert "rechnet nur die Menge 1 bei ε/D = 8 und 1 neu nach" in README and "der Test vergleicht die Spannen und Zählungen über die Mengen 1 bis 5 mit dieser Datei" in README


def test_bei_eps_durch_d_0_1_laufen_einzelne_paare_bis_zur_grenze_und_bei_0_05_meldet_die_demo_den_unterlauf():
    """Die Grenzen-Aussage der README (Menge 1, ein Kern; Rechenzeit dort rund eine Minute, nicht geprüft): 1 Lauf bei V-b bis V-e und 21 Läufe von V-a ohne Konvergenz (Band ± 2)."""
    res = E.analyse4(1, 0.1)
    assert abs(res["nicht_konvergiert"] - 1) <= 2 and res["nicht_konvergiert"] >= 1 and abs(res["nicht_konvergiert_a"] - 21) <= 2
    assert "1 Lauf bei V-b bis V-e sowie 21 Läufe der Bisektion von V-a ohne Konvergenz innerhalb von 200 000 Durchläufen" in README
    with pytest.raises(ValueError):
        E.analyse4(1, 0.05)
    assert min(C.C4) == 0.3
