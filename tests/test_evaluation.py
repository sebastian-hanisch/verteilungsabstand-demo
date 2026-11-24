"""Verdikte und Formatierung: reine Funktionen an handgesetzten Werten (Schwellen, jede Zahl im Text), dazu die Kennzahlen von Experiment 4 an einer Handmatrix."""

import math

import numpy as np
import pytest

import vab_constants as C
import vab_evaluation as E


# ---- Formatierung ---------------------------------------------------------------------------------------------------------------------

def test_zahl_hat_dezimalkomma_unendlich_und_kein_minus_null():
    assert E.zahl(0.3989, 3) == "0,399" and E.zahl(math.inf) == "∞" and E.zahl(-math.inf) == "−∞" and E.zahl(float("nan")) == "–"
    assert E.zahl(-0.0001, 3) == "0,000" and E.zahl(-1.5, 1) == "−1,5" and E.zahl(3.0, 0) == "3"


def test_ganz_und_prozent_und_exp():
    assert E.ganz(1000) == "1 000" and E.ganz(2024) == "2 024" and E.ganz(7) == "7"
    assert E.prozent(0.587, 1) == "58,7 %" and E.exp_text(1.2e-12, 1) == "1,2e-12" and E.exp_text(0) == "0" and E.exp_text(6.4e-8, 0) == "6e-8"
    assert E.glaettung_text(0.0) == "keine Glättung (g = 0)" and E.glaettung_text(1e-12) == "g = 1e-12"


def test_settings_aus_zustand_nimmt_fehlende_schluessel_vom_standard():
    s = E.settings_aus_zustand({"c4": 8.0})
    assert s.c4 == 8.0 and s.zellen1 == C.DEFAULTS["zellen1"] and E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


# ---- Experiment 1 ---------------------------------------------------------------------------------------------------------------------

def test_verdikt1_kasten_nennt_die_gemessenen_saetze():
    s = E.Settings()
    kind, text = E.verdikt1(s, E.reihe1(s))
    assert kind == "info" and "W1 = s Stunden in allen 17 Verschiebungen" in text and "TV = KS in allen Verschiebungen" in text
    assert "TV sättigt bei 1 ab s = 3,0 Stunden" in text and "JS erreicht ln 2 = 0,693 ab s = 3,0 Stunden" in text and "KL ist ohne Glättung schon ab s = 0,5 Stunden unendlich" in text


def test_verdikt1_zwei_spitzen_nennt_eine_zeile_je_verschiebung_und_die_nichtmonotonie():
    s = E.Settings(profil1=C.PROFIL1[1], glaettung1=1e-12)
    kind, text = E.verdikt1(s, E.reihe1(s))
    assert "bei s = 1 / 2 / 3 / 4 / 6 Stunden" in text and "W1 = 1,000 / 2,000 / 3,000 / 4,000 / 5,999 Stunden" in text and "TV = 0,399 / 0,639 / 0,878 / 0,911 / 0,840" in text
    assert "KL (mit g = 1e-12) = 0,498, 1,964, 4,199, 6,640, 11,160" in text and "TV ist nicht monoton" in text


def test_verdikt1_aendert_sich_mit_der_einstellung():
    kreis = E.Settings(profil1=C.PROFIL1[1], achse1="Kreis", glaettung1=1e-12)
    gerade = E.Settings(profil1=C.PROFIL1[1], glaettung1=1e-12)
    assert E.verdikt1(kreis, E.reihe1(kreis))[1] != E.verdikt1(gerade, E.reihe1(gerade))[1]
    ohne = E.Settings(profil1=C.PROFIL1[1], glaettung1=0.0)
    assert E.reihe1(ohne)["kl"][8] == pytest.approx(E.reihe1(ohne)["kl_roh"][8]) and E.zeile1(E.reihe1(ohne), 6.0)["kl"] == pytest.approx(11.410, abs=2e-3)


def test_glaettung_regler_ist_wirksam_beim_zwei_spitzen_profil_und_beim_kasten():
    spitzen = E.Settings(profil1=C.PROFIL1[1])
    werte = [E.kl_gegen_glaettung(spitzen, 6.0)[g] for g in C.GLAETTUNG1]
    assert werte[0] > werte[1] > werte[2] > werte[3]                           # 11,41 > 11,16 > 10,13 > 7,49
    kasten = E.kl_gegen_glaettung(E.Settings(), 3.0)
    assert kasten[0.0] == math.inf and all(math.isfinite(kasten[g]) for g in C.GLAETTUNG1[1:])


def test_kreis_aendert_nur_w1_und_die_profile():
    s_g, s_k = E.Settings(profil1=C.PROFIL1[1]), E.Settings(profil1=C.PROFIL1[1], achse1="Kreis")
    rg, rk = E.reihe1(s_g), E.reihe1(s_k)
    assert rk["w1"][-1] != rg["w1"][-1] and rk["w1"][1] == pytest.approx(0.4982, abs=1e-3) and max(rk["w1"]) < max(rg["w1"])


def test_reihe1_hat_je_zellenzahl_17_zeilen_und_kasten_ist_von_der_zellenzahl_unabhaengig_fuer_ganze_zellen():
    for zellen in C.ZELLEN1:
        r = E.reihe1(E.Settings(zellen1=zellen))
        assert len(r["s"]) == 17 and all(abs(w - sv) < 1e-9 for w, sv in zip(r["w1"], r["s"]))


# ---- Experiment 2 ---------------------------------------------------------------------------------------------------------------------

def test_verdikt2_nennt_die_nullzeilen_und_den_groessten_fehler():
    s = E.Settings()
    z = E.reihe2(s)
    kind, text = E.verdikt2(s, z)
    assert "W1 = 0 bei 10, 20, 50 Zellen" in text and "der größte Fehler ist 0,85 Zellenbreiten bei 500 Zellen" in text and "unter einer Zellenbreite" in text
    assert [x["zellen"] for x in z] == sorted(set(C.ZELLEN2_REIHE) | {s.zellen2})


def test_regler_zellen2_wirkt_die_gewaehlte_zahl_steht_in_der_reihe():
    for b in (24, 96):
        z = E.reihe2(E.Settings(zellen2=b))
        assert b in [x["zellen"] for x in z]
    assert E.treppe_zeile(24, 0.0137)["w1"] == 0.0 and E.treppe_zeile(96, 0.0137)["w1"] == pytest.approx(2 / 96)


def test_verdikt2_glatt_und_dirac_unterscheiden_sich():
    glatt = E.Settings(profil2=C.PROFIL2[1])
    kind, text = E.verdikt2(glatt, E.reihe2(glatt))
    assert "Glatte Gauß-Profile gleicher Breite" in text and "ab 50 Zellen ist der größte Fehler" in text


def test_verdikt2_w2_nennt_beide_faelle():
    zeilen = [E.w2_raster_zeile(b, sg) for b, sg in C.W2_ZEILEN]
    kind, text = E.verdikt2_w2(zeilen)
    assert "ungleicher Streuung (σ₂ = 0,7)" in text and "0,1658, 0,1672, 0,1660 bei 25, 100, 400 Zellen" in text and "0,1533, 0,1643 bei 28, 112 Zellen" in text
    assert "gleicher Streuung (σ₂ = 1) ist er 0,2181, 0,2041, 0,1224 bei 25, 100, 400 Zellen" in text and "nur 6e-8, 1e-6" in text
    assert "nur für gegeneinander verschobene Zellen" not in text and "bei ungleicher Streuung, unabhängig von der Lage der Zellen" in text


# ---- Experiment 3 ---------------------------------------------------------------------------------------------------------------------

def test_reihe3_beide_richtungen_und_zahlen_im_band():
    zeilen = [E.reihe3(2026, 500, b, 0.0, 300) for b in C.B3_REIHE]
    xy = [z["anteil_xy"] for z in zeilen]
    assert 0.03 < xy[0] < 0.15 and 0.5 < xy[1] < 0.72 and xy[2] > 0.95 and xy[3] == 1.0 and xy == sorted(xy)       # steigt mit den Zellen
    assert zeilen[0]["w1_hist_median"] > 0 and 0.0020 < zeilen[0]["w1_sort_median"] < 0.0042


def test_bei_kleinem_b_haengt_der_anteil_an_der_richtung():
    z = E.reihe3(2026, 500, 10, 0.02, 300)
    assert z["anteil_xy"] > 0.35 and z["anteil_yx"] < 0.2 and z["anteil_xy"] - z["anteil_yx"] > 0.25


def test_verdikt3_nennt_beide_richtungen_und_die_grenze_des_beispiels():
    s = E.Settings()
    zeilen = [E.reihe3(s.seed_input, s.n3, b, s.d3, s.seeds3) for b in C.B3_REIHE]
    kind, text = E.verdikt3(s, zeilen)
    assert "KL(x ‖ y) = ∞ in" in text and "KL(y ‖ x) in" in text and "10 / 50 / 200 / 1 000 Zellen" in text and "wie der Rauschboden mit n und Dimension wächst, zeigt diese Demo nicht" in text


def test_w1_histogramm_bleibt_endlich_wo_kl_unendlich_ist():
    z = E.reihe3(2026, 500, 1000, 0.0, 50)
    assert z["anteil_xy"] == 1.0 and np.all(np.isfinite(z["w1_hist"]))


# ---- Experiment 4: Kennzahlen an einer Handmatrix ---------------------------------------------------------------------------------------

def test_kennzahlen4_an_einer_handmatrix():
    # drei Verteilungen: Selbstabstand 0,5 bei Verteilung 0 (Verteilung 1 ist näher: 0,2), 0 bei den anderen
    mat = np.array([[0.5, 0.2, 0.9], [0.2, 0.0, 0.4], [0.9, 0.4, 0.0]])
    paare = [(0, 1), (0, 2), (1, 2)]
    exakt = np.array([0.1, 0.45, 0.2])
    nn_exakt = np.array([1, 0, 1])
    k = E.kennzahlen4(mat, exakt, paare, nn_exakt)
    assert k["nicht_naechster"] == 1 and k["selbst_mittel"] == pytest.approx(0.5 / 3) and k["kendall"] == pytest.approx(1.0)
    assert k["nn_wie_exakt"] == 3 and k["verhaeltnis"] == pytest.approx(float(np.median([2.0, 2.0, 2.0]))) and (k["verletzt"], k["tripel"], k["negativ"]) == (0, 1, 0)


def test_analyse4_standardmenge_im_band_der_messung():
    res = E.analyse4(1, 8.0)
    k = res["kennzahlen"]
    assert res["pruefsumme"] == "303ccccd8b59ce8c" and res["d"] == pytest.approx(0.0268, abs=1e-4)
    assert k["V-b"]["nicht_naechster"] > 10 and k["V-e"]["nicht_naechster"] > 10 and k["V-c"]["nicht_naechster"] == 0 and k["V-d"]["nicht_naechster"] == 0
    assert k["V-c"]["selbst_mittel"] == 0.0 and k["V-d"]["selbst_mittel"] == 0.0 and k["V-b"]["selbst_mittel"] > 0.01
    assert k["V-c"]["kendall"] > k["V-d"]["kendall"] > k["V-b"]["kendall"] > k["V-e"]["kendall"] - 0.05
    assert res["nicht_konvergiert"] <= 2 and res["produkte_median"] < 100


def test_verdikt4_nennt_verwendung_menge_und_grenzen():
    kind, text = E.verdikt4(E.analyse4(1, 1.0))
    assert "Verteilungsmenge 1, 24 Verteilungen, 276 Paare" in text and "V-c und V-d haben den Selbstabstand 0 durch Konstruktion" in text and "Dreiecksungleichung der Wurzel" in text
    assert "nicht als Aussage über alle Verteilungen" in text


def test_analyse4_zu_kleines_eps_wird_gemeldet():
    with pytest.raises(ValueError):
        E.analyse4(1, 0.01)


# ---- Experiment 5 ---------------------------------------------------------------------------------------------------------------------

def test_verdikt5_nennt_beide_ordnungen_und_den_satz():
    s = E.Settings()
    tab = E.tabelle5(s)
    mc = E.mc5(s)
    kind, text = E.verdikt5(s, tab, mc)
    assert "W1 ordnet die zu breite Prognose F2 vor die verschobene Prognose F1 (0,479 gegen 0,500)" in text and "W2² ordnet die verschobene Prognose F1 vor die zu breite Prognose F2 (0,250 gegen 0,360)" in text
    assert "400 von 400" in text and "E|X − y| wächst mit σ" in text and "bekannte Wahrheit N(0, 1)" in text


def test_verdikt5_eine_prognose_die_nicht_zu_schmal_ist_bekommt_keine_behauptung():
    s = E.Settings(f3_sigma=0.85)
    tab = E.tabelle5(s)
    # σ = 0,85 ist noch zu schmal: E|X - y| stellt sie weiterhin vor die wahre (der Satz gilt für jedes σ < 1)
    assert E.rangfolge(tab, "exy").index("F3") < E.rangfolge(tab, "exy").index("F0")


# ---- Befunde der Aussagenprüfung ------------------------------------------------------------------------------------------------------

def test_verdikt1_bei_glaettung_aus_sagt_ohne_glaettung_statt_mit_keine_glaettung():
    s = E.settings_aus_zustand(dict(C.PRESETS["Zwei Spitzen"], glaettung1=0.0))
    text = E.verdikt1(s, E.reihe1(s))[1]
    assert "KL ohne Glättung = 0,498, 1,964, 4,199, 6,641, 11,410" in text and "mit keine" not in text
    s = E.settings_aus_zustand(C.PRESETS["Zwei Spitzen"])
    assert "KL (mit g = 1e-12) = 0,498, 1,964, 4,199, 6,640, 11,160" in E.verdikt1(s, E.reihe1(s))[1]


@pytest.mark.parametrize("breite, rundung", [(0.01, False), (0.02, False), (0.03, True), (0.04, True)])
def test_verdikt2_nennt_rundung_nur_wenn_der_fehler_unter_der_rundungsgrenze_liegt(breite, rundung):
    s = E.Settings(profil2=C.PROFIL2[1], breite2=breite)
    zeilen = E.reihe2(s)
    groesster = max(abs(z["fehler"]) for z in zeilen if z["zellen"] >= 50)
    text = E.verdikt2(s, zeilen)[1]
    assert (groesster < C.RUNDUNG_GRENZE) is rundung and ("(Rundung)" in text) is rundung and E.exp_text(groesster, 1) in text


def test_verdikt2_bei_sigma_0_01_nennt_den_fehler_4_2e_5_ohne_rundung():
    s = E.Settings(profil2=C.PROFIL2[1], breite2=0.01)
    text = E.verdikt2(s, E.reihe2(s))[1]
    assert "ab 50 Zellen ist der größte Fehler 4,2e-5, " in text and "Rundung" not in text


def test_die_einleitung_der_app_gilt_nur_fuer_die_genannten_faelle():
    """W1 wächst beim Kasten (Gerade und Kreis, 24/48/96 Zellen) und bei zwei Spitzen auf der Geraden, bei zwei Spitzen auf dem Kreis nicht;
    TV, KS, JS sättigen beim Kasten und steigen und fallen bei zwei Spitzen."""
    for zellen in C.ZELLEN1:
        for achse in C.ACHSE1:
            r = E.reihe1(E.Settings(profil1=C.PROFIL1[0], zellen1=zellen, achse1=achse))
            for k in ("w1", "tv", "ks", "js"):
                assert np.all(np.diff(r[k]) >= -1e-9), (zellen, achse, k)
            assert np.allclose(r["w1"], r["s"], atol=1e-9) and r["tv"][-1] == pytest.approx(1.0) and r["js"][-1] == pytest.approx(C.LN2) and math.isinf(r["kl_roh"][1])
            r = E.reihe1(E.Settings(profil1=C.PROFIL1[1], zellen1=zellen, achse1=achse, glaettung1=1e-12))
            assert (bool(np.all(np.diff(r["w1"]) > 0))) is (achse == "Gerade"), (zellen, achse)
            for k in ("tv", "ks", "js"):
                assert max(r[k]) > r[k][-1] + 0.01, (zellen, achse, k)        # steigt und fällt wieder
            assert max(r["tv"]) - r["tv"][-1] > 0.25 and max(r["js"]) - r["js"][-1] > 0.25


def test_tv_bei_zwei_spitzen_ist_nicht_monoton_mit_hoechstwert_bei_s_4_5():
    r = E.reihe1(E.Settings(profil1=C.PROFIL1[1], glaettung1=1e-12))
    i = int(np.argmax(r["tv"]))
    assert r["s"][i] == 4.5 and r["tv"][i] == pytest.approx(0.9345, abs=5e-4) and E.zeile1(r, 4.0)["tv"] < E.zeile1(r, 3.5)["tv"] < r["tv"][i] and E.zeile1(r, 6.0)["tv"] == pytest.approx(0.8399, abs=5e-4)


def test_der_erwartete_crps_ist_das_quadrat_der_l2_norm_plus_eine_konstante():
    """E[CRPS] = ∫(F − G)² dx + 1/√π gegen die Quadratur des Cramér-Abstands² in der Tabelle (jede der fünf Prognosen); F0 hat deshalb E[CRPS] = 1/√π und nicht 0."""
    tab = E.tabelle5(E.Settings())
    for z in tab:
        assert z["ecrps"] == pytest.approx(z["cramer2"] + 1.0 / math.sqrt(math.pi), abs=1e-6), z["name"]
    assert tab[0]["cramer2"] == pytest.approx(0.0, abs=1e-12) and tab[0]["ecrps"] == pytest.approx(0.5642, abs=5e-5)


def test_standardabweichung_der_anteile_aus_300_und_400_zufalls_seeds():
    assert E.sd_prozentpunkte(0.5, 300) == pytest.approx(2.887, abs=1e-3) and E.sd_prozentpunkte(0.08, 300) == pytest.approx(1.566, abs=1e-3) and E.sd_prozentpunkte(0.99, 300) == pytest.approx(0.574, abs=1e-3)
    assert E.sd_prozentpunkte(0.5, 400) == pytest.approx(2.5, abs=1e-9)
    s = E.Settings()
    text3 = E.verdikt3(s, [E.reihe3(s.seed_input, s.n3, b, s.d3, s.seeds3) for b in C.B3_REIHE])[1]
    assert "Standardabweichung von höchstens 2,9 Prozentpunkten (bei 8 % etwa 1,6, bei 99 % etwa 0,6)" in text3
    text5 = E.verdikt5(s, E.tabelle5(s), E.mc5(s))[1]
    assert "Standardabweichung von höchstens 2,5 Prozentpunkten" in text5 and "keine Rangordnung der Regeln" in text5


def test_unwirksam_satz_nennt_den_wert_beim_kleinsten_eps_nur_wenn_es_unwirksame_paare_gibt():
    ohne = {"unwirksam_a": 0}
    assert E.unwirksam_satz(ohne) == "bei 0 von 276 Paaren ist die Nebenbedingung nicht wirksam"
    mit = {"unwirksam_a": 18, "eps_min_durch_d": 0.1121, "unwirksam_median": 1.0423, "unwirksam_max": 10.974}
    satz = E.unwirksam_satz(mit)
    assert satz.startswith("bei 18 von 276 Paaren ist die Nebenbedingung nicht wirksam (dort zeigt die Tabelle den Wert beim kleinsten ε′ = 0,003, das sind 0,11 D: eine obere Schranke")
    assert "im Median das 1,04-Fache, höchstens das 10,97-Fache des exakten W2²" in satz


def test_preset_hilfetexte_stimmen_mit_den_verdikt_zeilen_ueberein():
    """Je betroffenem Preset die erwartete Zeile: Kasten 3 h (sättigt, KL = ∞), Raster 24 (beide Spitzen in einer Zelle beim eingestellten Abstand), Raster 96."""
    s = E.settings_aus_zustand(C.PRESETS["Kasten 3 h"])
    text = E.verdikt1(s, E.reihe1(s))[1]
    assert "TV sättigt bei 1 ab s = 3,0 Stunden" in text and "KL ist ohne Glättung schon ab s = 0,5 Stunden unendlich" in text
    assert "wächst W1 mit s" in C.PRESET_HELP["Kasten 3 h"] and "sättigen" in C.PRESET_HELP["Kasten 3 h"] and "alle abstände wachsen" not in C.PRESET_HELP["Kasten 3 h"].lower()
    s = E.settings_aus_zustand(C.PRESETS["Raster 24"])
    zeile = next(z for z in E.reihe2(s) if z["zellen"] == 24)
    assert zeile["w1"] == 0.0 and "Das Raster meldet W1 = 0 bei 10, 20, 24, 50 Zellen (beide Spitzen in einer Zelle)" in E.verdikt2(s, E.reihe2(s))[1]
    assert "Experiment 2 auf 24 Zellen" in C.PRESET_HELP["Raster 24"] and "Experiment 1 hat in der Voreinstellung schon 24" in C.PRESET_HELP["Raster 24"]
    assert C.PRESETS["Raster 24"]["zellen1"] == C.DEFAULTS["zellen1"] == 24
    s = E.settings_aus_zustand(C.PRESETS["Raster 96"])
    assert "Das Raster meldet W1 = 0 bei 10, 20, 50 Zellen" in E.verdikt2(s, E.reihe2(s))[1] and "eine Zelle = 15 Minuten" in C.PRESET_HELP["Raster 96"]
