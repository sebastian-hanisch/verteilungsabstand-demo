"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler hat einen Eintrag in SETTING_SPECS (Name im Session State = Schlüssel in `vab_constants.DEFAULTS`). Reglerwerte aus der Adresszeile werden auf die
Grenzen begrenzt (Zahlen, bei Stufenreglern auf das Raster der Stufe gerundet) beziehungsweise auf den nächsten erlaubten Wert eingerastet (Auswahllisten); ungültige Werte werden ignoriert."""

import random
from dataclasses import dataclass

import streamlit as st

import vab_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # erlaubte Werte (Auswahllisten und Stufen)
    step: float | None = None         # Stufe eines Zahlenreglers


def _opt(url, caster, key, options):
    return SettingSpec(url, caster, C.DEFAULTS[key], options=options)


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULTS["seed_input"], 0, C.SEED_MAX),
    "profil1": _opt("p1", str, "profil1", C.PROFIL1),
    "s1": SettingSpec("s1", float, C.DEFAULTS["s1"], C.S1_MIN, C.S1_MAX, step=C.S1_SCHRITT),
    "breite1": SettingSpec("w1", float, C.DEFAULTS["breite1"], C.BREITE1_MIN, C.BREITE1_MAX, step=0.1),
    "zellen1": _opt("z1", int, "zellen1", C.ZELLEN1),
    "achse1": _opt("a1", str, "achse1", C.ACHSE1),
    "glaettung1": _opt("g1", float, "glaettung1", C.GLAETTUNG1),
    "profil2": _opt("p2", str, "profil2", C.PROFIL2),
    "zellen2": _opt("z2", int, "zellen2", C.ZELLEN2),
    "abstand2": _opt("d2", float, "abstand2", C.ABSTAND2),
    "breite2": _opt("w2", float, "breite2", C.BREITE2),
    "n3": _opt("n3", int, "n3", C.N3),
    "b3": _opt("b3", int, "b3", C.B3),
    "d3": _opt("d3", float, "d3", C.D3),
    "richtung3": _opt("r3", str, "richtung3", C.RICHTUNG3),
    "seeds3": _opt("s3", int, "seeds3", C.SEEDS3),
    "c4": _opt("c4", float, "c4", C.C4),
    "verw4": _opt("v4", str, "verw4", C.VERW4),
    "menge4": SettingSpec("m4", int, C.DEFAULTS["menge4"], C.MENGE4_MIN, C.MENGE4_MAX),
    "f1_lage": _opt("f1", float, "f1_lage", C.F1_LAGE),
    "f2_sigma": _opt("f2", float, "f2_sigma", C.F2_SIGMA),
    "f3_sigma": _opt("f3", float, "f3_sigma", C.F3_SIGMA),
    "f4_lage": _opt("f4l", float, "f4_lage", C.F4_LAGE),
    "f4_sigma": _opt("f4s", float, "f4_sigma", C.F4_SIGMA),
    "nobs5": _opt("n5", int, "nobs5", C.NOBS5),
    "seeds5": _opt("s5", int, "seeds5", C.SEEDS5),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _eingerastet(spec: SettingSpec, wert):
    """Wert auf Grenzen begrenzt (und auf die Stufe gerundet) beziehungsweise auf die nächste erlaubte Option eingerastet; None, wenn nichts passt."""
    if spec.options is not None:
        if spec.caster is str:
            return wert if wert in spec.options else None
        if not isinstance(wert, (int, float)) or wert != wert:
            return None
        return min(spec.options, key=lambda o: abs(o - float(wert)))
    if isinstance(wert, float) and wert != wert:
        return None
    if spec.lo is not None:
        wert = max(spec.lo, wert)
    if spec.hi is not None:
        wert = min(spec.hi, wert)
    if spec.step:
        wert = min(spec.hi, max(spec.lo, round(round((wert - spec.lo) / spec.step) * spec.step + spec.lo, 10)))
    return wert


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (begrenzt beziehungsweise eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                wert = _eingerastet(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if wert is not None:
                st.session_state[state_key] = wert
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
