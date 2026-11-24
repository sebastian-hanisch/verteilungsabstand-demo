"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzen und einrasten; jedes Preset lässt sich rechnen und zeigt die erwartete Aussage."""

import re

import pytest

import vab_constants as C
import vab_evaluation as E
import vab_presets as P

NAMEN = ["Kasten 3 h", "Zwei Spitzen", "Raster 24", "Raster 96", "Selbstabstand", "Prognosen"]


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and list(C.PRESETS) == NAMEN
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.SETTING_SPECS), name          # jedes Preset legt ALLE Regler fest
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_no_two_presets_are_identical_and_the_default_is_exactly_one_preset():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)
    assert [n for n, p in C.PRESETS.items() if p == C.DEFAULTS] == ["Kasten 3 h"]
    assert E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


def test_setting_specs_cover_every_preset_key_and_every_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(C.DEFAULTS)
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)          # keine doppelten Adressparameter
    assert all(spec.default == C.DEFAULTS[key] for key, spec in P.SETTING_SPECS.items())


def test_preset_hilfetexte_nennen_keine_messergebnisse():
    """Die Hilfetexte der Presets nennen keine gemessenen Zahlen (jede Zahl der App steht in der README oder in einem Test): keine Prozent, keine Zählungen „x von y“."""
    for name, text in C.PRESET_HELP.items():
        assert "%" not in text and not re.search(r"\d+ von \d+", text), (name, text)


def test_hilfetexte_der_presets_nennen_keine_planbegriffe():
    for text in C.PRESET_HELP.values():
        for verboten in ("Phase", "Pilot", "Stück", "Plan-", "Orakel"):
            assert verboten not in text


def test_jedes_preset_zeigt_die_erwartete_aussage():
    kasten = C.PRESETS["Kasten 3 h"]
    s = E.settings_aus_zustand(kasten)
    text = E.verdikt1(s, E.reihe1(s))[1]
    assert "KL ist ohne Glättung schon ab s = 0,5 Stunden unendlich" in text                       # das Preset „Kasten 3 h“ meldet KL = ∞

    s = E.settings_aus_zustand(C.PRESETS["Zwei Spitzen"])
    text = E.verdikt1(s, E.reihe1(s))[1]
    assert "TV = 0,399 / 0,639 / 0,878 / 0,911 / 0,840" in text and "g = 1e-12" in text

    for name, b, null in (("Raster 24", 24, True), ("Raster 96", 96, False)):
        s = E.settings_aus_zustand(C.PRESETS[name])
        assert s.zellen1 == b and s.zellen2 == b
        zeile = next(z for z in E.reihe2(s) if z["zellen"] == b)
        assert (zeile["w1"] == 0.0) is null

    s = E.settings_aus_zustand(C.PRESETS["Selbstabstand"])
    k = E.analyse4(s.menge4, s.c4)["kennzahlen"]
    assert s.c4 == 8.0 and s.verw4 == "V-b" and k["V-b"]["nicht_naechster"] > 10 and k["V-c"]["nicht_naechster"] == 0 and k["V-d"]["nicht_naechster"] == 0       # V-c und V-d melden keinen Selbstabstand

    s = E.settings_aus_zustand(C.PRESETS["Prognosen"])
    mc = E.mc5(s)
    assert s.nobs5 == 25 and 0.6 < mc["platz1"]["W1"] < 0.9 and mc["platz1"]["E|X − y|"] == 0.0


def test_einrasten_und_begrenzen_des_permalinks():
    sp = P.SETTING_SPECS
    assert P._eingerastet(sp["s1"], 99.0) == 8.0 and P._eingerastet(sp["s1"], -3.0) == 0.0 and P._eingerastet(sp["s1"], 3.2) == 3.0 and P._eingerastet(sp["s1"], 3.3) == 3.5
    assert P._eingerastet(sp["breite1"], 0.04) == 0.3 and P._eingerastet(sp["breite1"], 1.04) == 1.0
    assert P._eingerastet(sp["zellen1"], 50) == 48 and P._eingerastet(sp["zellen2"], 70) in (50, 96) and P._eingerastet(sp["zellen2"], 1e9) == 5000
    assert P._eingerastet(sp["glaettung1"], 0.0) == 0.0 and P._eingerastet(sp["glaettung1"], 3e-10) == 1e-12 and P._eingerastet(sp["d3"], 0.03) in (0.02, 0.05)
    assert P._eingerastet(sp["profil1"], "Kasten 3 h") == "Kasten 3 h" and P._eingerastet(sp["profil1"], "unbekannt") is None and P._eingerastet(sp["verw4"], "V-z") is None
    assert P._eingerastet(sp["c4"], 5.0) in (3.0, 8.0) and P._eingerastet(sp["c4"], 0.01) == 0.3 and P._eingerastet(sp["menge4"], 99) == C.MENGE4_MAX and P._eingerastet(sp["menge4"], 0) == C.MENGE4_MIN
    assert P._eingerastet(sp["seed_input"], -5) == 0 and P._eingerastet(sp["c4"], float("nan")) is None and P._eingerastet(sp["nobs5"], 1000) == 400
