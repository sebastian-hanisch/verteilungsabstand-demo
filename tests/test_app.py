"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen und prüft, dass kein Python-Fehler
auftritt (insbesondere `StreamlitDuplicateElementId` bei mehreren st.plotly_chart ohne key= und `StreamlitAPIException` bei Reglern mit falschen Grenzen). Dazu: je Preset die erwartete
Aussage, Permalink, Randfälle, Worst-Case am Regler-Maximum.

Laufzeit: Experiment 4 (Sinkhorn-Stapel) rechnet bei ε/D ≥ 1 unter einer Sekunde, bei kleinerem ε/D und für V-a einige Sekunden; der Zwischenspeicher gilt im Prozess und wärmt spätere Tests."""

import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import vab_constants as C
import vab_evaluation as E
import vab_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio")


def _app(**adresse) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in adresse.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _setze(at, **werte):
    for key, wert in werte.items():
        _regler(at, key).set_value(wert)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (werte, [str(e) for e in at.exception])
    return at


def _texte(at) -> str:
    return " ".join(str(e.value) for kind in ("success", "info", "warning", "markdown", "caption") for e in getattr(at, kind))


def _metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_all_sections():
    at = _app()
    titel = " ".join(s.value for s in at.subheader)
    for abschnitt in ("Experiment 1: Verschieben statt Überlappen", "Experiment 2: Das Raster macht die Zahl", "Experiment 3: Histogramm ohne Unterschied", "Experiment 4: Distanz oder Divergenz?",
                      "Experiment 5: Prognosen bewerten", "Wo die Annahmen enden"):
        assert abschnitt in titel, abschnitt
    labels = [e.label for e in at.expander]
    assert [l for l in labels if "Mathematische Formulierung" in l] == ["📐 Mathematische Formulierung"] and any("Annahmen, Grenzen und bekannte Probleme" in l for l in labels)
    assert any("So rechnen die Abstände" in l for l in labels)
    assert len(at.metric) == 8 and len(at.info) >= 9 and not at.warning
    schluessel = [p.key for p in at.get("plotly_chart")]
    assert len(set(schluessel)) == len(schluessel) >= 9          # jedes Diagramm hat einen eigenen Schlüssel
    alles = " ".join(m.value for m in at.markdown)
    assert "Kernfragen" in alles and "Wächst der Abstand mit der Verschiebung?" in alles and "Hängt die Zahl am Raster?" in alles and "Ist „Sinkhorn-Distanz“ (V-a bis V-e) ein Abstand?" in alles


def test_every_button_click_does_not_raise():
    at = _app()
    labels = [b.label for b in at.button]
    assert len(labels) == len(C.PRESETS) + 1, labels                      # sechs Presets und der Zufalls-Seed-Button
    for index, label in enumerate(labels):
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    at = _app()
    schluessel = [s.key for s in at.sidebar.slider]
    assert schluessel == ["s1", "breite1", "menge4"], schluessel
    _setze(at, profil1=C.PROFIL1[1])                      # die Breite der Spitzen ist beim Kasten deaktiviert
    for key in schluessel:
        for edge in ("min", "max"):
            regler = _regler(at, key)
            regler.set_value(regler.min if edge == "min" else regler.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


# ----------------------------------------------------------------------------------------------------------------------------------
# Auswahlregler und Presets
# ----------------------------------------------------------------------------------------------------------------------------------

def _ecken(optionen):
    """Erste, letzte und eine mittlere Option (bei kurzen Listen alle)."""
    optionen = list(optionen)
    return optionen if len(optionen) <= 4 else [optionen[0], optionen[len(optionen) // 2], optionen[-1]]


@pytest.mark.parametrize("key", [k for k, s in P.SETTING_SPECS.items() if s.options is not None])
def test_optionen_der_auswahlregler_laufen_ohne_fehler(key):
    at = _app()
    if key == "breite2":
        _setze(at, profil2=C.PROFIL2[1])                  # die Breite ist bei zwei Spitzen deaktiviert
    for option in _ecken(P.SETTING_SPECS[key].options):
        _setze(at, **{key: option})


def test_alle_stufen_von_eps_durch_d_und_alle_verwendungen_laufen():
    at = _app()
    for c in C.C4:
        _setze(at, c4=c)
    for v in C.VERW4:
        _setze(at, verw4=v)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler_und_zeigt_seine_einstellung(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, wert in C.PRESETS[name].items():
        assert at.session_state[key] == wert, (name, key)


def test_preset_kasten_zeigt_kl_unendlich_und_w1_gleich_s():
    at = _app()
    m = _metrics(at)
    assert m["W1 (Stunden)"] == "3,000" and m["KL (nat)"] == "∞" and m["TV"] == "1,000" and m["JS (nat)"] == "0,693"
    assert "W1 = s Stunden in allen 17 Verschiebungen" in _texte(at)


def test_preset_zwei_spitzen_zeigt_die_zeile_s_gleich_4():
    at = _app()
    at.sidebar.button(key="preset_Zwei Spitzen").click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert m["W1 (Stunden)"] == "4,000" and m["TV"] == "0,911" and m["KL (nat)"] == "6,640"
    assert "TV ist nicht monoton" in _texte(at) and "| 4 | 4,000 | 0,911 |" in _texte(at)


def test_preset_raster_24_meldet_w1_gleich_0_bei_24_zellen():
    at = _app()
    at.sidebar.button(key="preset_Raster 24").click().run(timeout=TIMEOUT)
    assert at.session_state["zellen2"] == 24 and "| 24 | 0,0000 | 0,0137 |" in _texte(at)
    at.sidebar.button(key="preset_Raster 96").click().run(timeout=TIMEOUT)
    assert "| 96 | 0,0208 | 0,0137 |" in _texte(at)


def test_preset_selbstabstand_meldet_fuer_v_c_und_v_d_keinen_selbstabstand():
    at = _app()
    at.sidebar.button(key="preset_Selbstabstand").click().run(timeout=TIMEOUT)
    text = _texte(at)
    assert at.session_state["c4"] == 8.0 and at.session_state["verw4"] == "V-b"
    k = E.analyse4(1, 8.0)["kennzahlen"]
    assert f"| V-c Genevay (Faktor 2) | 0,0000 (Konstruktion) | 0 von 24 (Konstruktion) |" in text
    assert f"| V-b ⟨C, P_ε⟩ | {E.zahl(k['V-b']['selbst_mittel'], 4)} | {k['V-b']['nicht_naechster']} von 24 |" in text


def test_preset_prognosen_nennt_25_beobachtungen():
    at = _app()
    at.sidebar.button(key="preset_Prognosen").click().run(timeout=TIMEOUT)
    assert at.session_state["nobs5"] == 25 and "bei 25 Beobachtungen auf Platz 1" in _texte(at)


def test_regler_profil1_kasten_deaktiviert_die_breite_der_spitzen():
    at = _app()
    assert _regler(at, "breite1").proto.disabled is True
    _setze(at, profil1=C.PROFIL1[1])
    assert _regler(at, "breite1").proto.disabled is False
    _setze(at, breite1=2.0)
    assert "σ = 2,0 Stunden" in _texte(at)


def test_breite2_ist_nur_bei_glatten_profilen_wirksam_und_sonst_deaktiviert():
    at = _app()
    assert _regler(at, "breite2").proto.disabled is True
    _setze(at, profil2=C.PROFIL2[1])
    assert _regler(at, "breite2").proto.disabled is False


def test_richtung_der_kl_wechselt_die_angezeigte_zahl():
    at = _app()
    a = _metrics(at)["Anteil der Zufalls-Seeds mit KL(x ‖ y) = ∞"]
    _setze(at, richtung3=C.RICHTUNG3[1])
    assert _metrics(at)["Anteil der Zufalls-Seeds mit KL(y ‖ x) = ∞"] != a


def test_zu_kleines_eps_durch_d_wird_nicht_angeboten():
    assert min(C.C4) >= 0.1 and C.C4 == tuple(sorted(C.C4))


# ----------------------------------------------------------------------------------------------------------------------------------
# Permalink, Randfälle, Worst-Case, Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_permalink_uebernimmt_werte_begrenzt_und_eingerastet():
    at = _app(s1="99", z1="50", g1="0.000001", z2="70", d2="0.0138", b3="1001", c4="5", v4="V-e", m4="99", n5="1000", p1="unbekannt", r3="x ‖ y")
    s = at.session_state
    assert s["s1"] == 8.0 and s["zellen1"] == 48 and s["glaettung1"] == 1e-6 and s["zellen2"] in (50, 96) and s["abstand2"] == 0.0137 and s["b3"] == 1000 and s["c4"] in (3.0, 8.0)
    assert s["verw4"] == "V-e" and s["menge4"] == C.MENGE4_MAX and s["nobs5"] == 400 and s["profil1"] == C.DEFAULTS["profil1"] and s["richtung3"] == "x ‖ y"


def test_adresszeile_spiegelt_die_einstellungen():
    at = _setze(_app(), s1=2.5, c4=3.0, zellen2=500)
    for param, wert in (("s1", "2.5"), ("c4", "3.0"), ("z2", "500")):
        assert at.query_params[param] in ([wert], wert)


def test_randfaelle_alle_minima_und_seed_0():
    at = _app()
    _setze(at, profil1=C.PROFIL1[1], profil2=C.PROFIL2[1])
    _setze(at, s1=0.0, breite1=C.BREITE1_MIN, zellen1=C.ZELLEN1[0], achse1="Kreis", glaettung1=C.GLAETTUNG1[0], zellen2=C.ZELLEN2[0], abstand2=C.ABSTAND2[0], breite2=C.BREITE2[0],
           n3=C.N3[0], b3=C.B3[0], d3=C.D3[0], seeds3=C.SEEDS3[0], c4=C.C4[0], menge4=C.MENGE4_MIN, f1_lage=C.F1_LAGE[0], f2_sigma=C.F2_SIGMA[0], f3_sigma=C.F3_SIGMA[0], f4_lage=C.F4_LAGE[0],
           f4_sigma=C.F4_SIGMA[0], nobs5=C.NOBS5[0], seeds5=C.SEEDS5[0], seed_input=0)
    assert len(at.metric) == 8


def test_worst_case_alle_maxima_zugleich_ohne_fehler():
    """Regler-Maximum: 96 Zellen, 5 000 Zellen im Raster, n = 1 000 mit 1 000 Zellen und 300 Zufalls-Seeds, ε/D = 0,3 (am längsten), Menge 20, 400 Beobachtungen und 400 Zufalls-Seeds.
    Die Zeit wird nicht geprüft (Abschnitt 1c)."""
    at = _app()
    _setze(at, profil1=C.PROFIL1[1], profil2=C.PROFIL2[1])
    _setze(at, s1=C.S1_MAX, breite1=C.BREITE1_MAX, breite2=C.BREITE2[-1], zellen1=C.ZELLEN1[-1], zellen2=C.ZELLEN2[-1], n3=C.N3[-1], b3=C.B3[-1], d3=C.D3[-1], seeds3=C.SEEDS3[-1], c4=C.C4[0], verw4="V-a", menge4=C.MENGE4_MAX,
           nobs5=C.NOBS5[-1], seeds5=C.SEEDS5[-1])
    assert len(at.metric) == 8


def test_die_ansicht_nennt_keine_planbegriffe_und_keine_umgebungsaussagen():
    at = _app()
    gesamt = _texte(at) + " " + " ".join(e.label for e in at.expander) + " " + " ".join(s.value for s in at.subheader) + " " + " ".join(m.value for m in at.markdown)
    for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Orakel", "Saatwert", "Stück", "Vormessung", "Prüfung B", "wasserstein"):
        assert verboten not in gesamt, verboten
    assert not re.search(r"\btr[äa]gt\b|\btragen\b", gesamt)
    assert "Zufalls-Seed" in " ".join(w.label for w in at.sidebar.number_input)


def test_jeder_regler_der_seitenleiste_hat_einen_hilfetext():
    at = _app()
    for art in REGLER_ARTEN:
        for w in getattr(at.sidebar, art):
            if w.key in P.SETTING_SPECS:
                assert w.help, w.key


def test_sinkhorn_distanz_steht_nie_ohne_verwendung():
    """Jedes Auftreten von „Sinkhorn-Distanz“ in der Ansicht steht in einem Absatz, der V-a bis V-e nennt, von fünf verschiedenen Größen spricht oder die Frage stellt."""
    at = _app()
    for m in list(at.markdown) + list(at.info) + list(at.caption):
        if "Sinkhorn-Distanz" in str(m.value):
            assert re.search(r"V-[a-e]|Verwendung|fünf verschiedene Größen", str(m.value)), str(m.value)[:200]      # „fünf verschiedene Größen“: der erzeugte Expander „💬 In einfachen Worten“


def test_kernfragen_nennen_tv_bei_zwei_spitzen_als_nicht_monoton_und_die_glatten_profile_mit_sigma():
    text = _texte(_app())
    assert "TV aber ist nicht monoton: Höchstwert 0,935 bei s = 4,5, danach 0,840 bei s = 6 Stunden" in text and "TV aber fällt" not in text
    assert "Glatte Profile gleicher Breite (σ = 0,04 des Tages) sind dagegen ab 50 Zellen auf Rundung genau" in text
    assert max(abs(z["fehler"]) for z in E.reihe2(E.Settings(profil2=C.PROFIL2[1], breite2=0.04)) if z["zellen"] >= 50) < C.RUNDUNG_GRENZE


def test_einleitung_nennt_die_faelle_in_denen_die_annahme_gilt():
    alles = " ".join(" ".join(m.value for m in _app().markdown).split())          # Zeilenumbrüche des Quelltexts zählen nicht
    assert "beim Kasten und bei zwei Spitzen auf der Geraden, auf dem Kreis bei zwei Spitzen nicht immer" in alles and "beim Kasten sättigen sie, bei zwei Spitzen steigen sie und fallen wieder" in alles


def test_verdikt_zu_glatten_profilen_mit_sigma_0_01_nennt_keine_rundung():
    at = _app()
    _setze(at, profil2=C.PROFIL2[1])
    _setze(at, breite2=0.01)
    text = _texte(at)
    assert "ab 50 Zellen ist der größte Fehler 4,2e-5, " in text and "4,2e-5 (Rundung)" not in text


def test_expander_in_einfachen_worten_verlinkt_die_erklaerseite_und_die_lesehilfe_steht_einmal():
    at = _app()
    expander = [e for e in at.expander if e.label == "💬 In einfachen Worten"]
    assert len(expander) == 1
    inhalt = " ".join(m.value for m in expander[0].markdown)
    assert "](https://sebastianhanisch.net/abstand-zwischen-verteilungen-erklaert.html)" in inhalt
    alles = " ".join(m.value for m in at.markdown)
    assert "der Expander **💬 In einfachen Worten**" in alles                      # die Einleitung nennt ihn
    assert alles.count("**Lesehilfe.**") == 1 and "](https://sebastianhanisch.net/abstand-zwischen-verteilungen-erklaert.html#fachleute)" in alles      # eine Lesehilfe, mit Link auf den Fachteil
