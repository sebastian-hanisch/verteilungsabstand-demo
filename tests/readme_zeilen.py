"""Tabellen und Sätze der README, mit denselben Formatfunktionen wie in der App aus dem Code der Demo (und der Datei `tools/messreihen.json`) gebildet.

Die README muss jedes dieser Stücke wörtlich enthalten (`tests/test_claims.py`); `python tests/readme_zeilen.py` druckt sie zum Einfügen."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import vab_abstand as A  # noqa: E402
import vab_constants as C  # noqa: E402
import vab_evaluation as E  # noqa: E402

MESS = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))


def md(kopf, zeilen) -> str:
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


def _kl(x) -> str:
    return "∞" if x == float("inf") else E.zahl(x, 3)


# ---- Experiment 1 --------------------------------------------------------------------------------------------------------------------

def exp1_kasten() -> str:
    r = E.reihe1(E.Settings())
    zeilen = []
    for sv in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
        z = E.zeile1(r, sv)
        zeilen.append([E.zahl(sv, 0), E.zahl(z["w1"], 3), E.zahl(z["tv"], 3), E.zahl(z["ks"], 3), E.zahl(z["js"], 3), _kl(z["kl_roh"])])
    return md(["s (Stunden)", "W1 (Stunden)", "TV", "KS", "JS (nat)", "KL ohne Glättung (nat)"], zeilen)


def exp1_kasten_verdikt() -> str:
    s = E.Settings()
    return E.verdikt1(s, E.reihe1(s))[1]


def exp1_spitzen() -> str:
    s = E.Settings(profil1=C.PROFIL1[1], glaettung1=1e-12)
    r = E.reihe1(s)
    zeilen = []
    for sv in C.S_ZEILEN:
        z = E.zeile1(r, sv)
        zeilen.append([E.zahl(sv, 0), E.zahl(z["w1"], 3), E.zahl(z["tv"], 3), E.zahl(z["ks"], 3), E.zahl(z["js"], 3), _kl(z["kl"]), _kl(z["kl_roh"])])
    return md(["s (Stunden)", "W1 (Stunden)", "TV", "KS", "JS (nat)", "KL mit g = 1e-12 (nat)", "KL ohne Glättung (nat)"], zeilen)


def exp1_spitzen_verdikt() -> str:
    s = E.Settings(profil1=C.PROFIL1[1], glaettung1=1e-12)
    return E.verdikt1(s, E.reihe1(s))[1]


def exp1_glaettung() -> str:
    s = E.Settings(profil1=C.PROFIL1[1])
    zeilen = []
    for sv in (4.0, 6.0):
        gl = E.kl_gegen_glaettung(s, sv)
        zeilen.append([E.zahl(sv, 0)] + [E.zahl(gl[g], 3) for g in C.GLAETTUNG1])
    return md(["s (Stunden)", "g = 0", "g = 1e-12", "g = 1e-9", "g = 1e-6"], zeilen)


def exp1_kreis() -> str:
    p, q = np.zeros(24), np.zeros(24)
    p[23], q[1] = 1.0, 1.0
    return f"Eine Spitze um 23:30 Uhr gegen eine Spitze um 1:30 Uhr (24 Zellen): W1 ist {E.zahl(A.w1(p, q, 1.0), 0)} Stunden auf der Geraden und {E.zahl(A.w1_kreis(p, q, 1.0), 0)} Stunden auf dem Kreis."


def exp1_breite() -> str:
    """Zahl der Zellen in der Zeile „Zwei Spitzen“: W1 bei s = 6 hängt an den abgeschnittenen Rändern (5,999 statt 6)."""
    z = E.zeile1(E.reihe1(E.Settings(profil1=C.PROFIL1[1])), 6.0)
    return f"W1 = {E.zahl(z['w1'], 3)} Stunden bei s = 6 statt 6,000: die Zellen sind bei 0 Uhr und 24 Uhr abgeschnitten."


# ---- Experiment 2 --------------------------------------------------------------------------------------------------------------------

def exp2_dirac() -> str:
    zeilen = E.reihe2(E.Settings(zellen2=100))
    return md(["Zellen B", "Zelle der ersten Spitze", "Zelle der zweiten Spitze", "W1 im Raster", "W1 wahr", "Betrag des Fehlers / h"],
              [[E.ganz(z["zellen"]), str(z["zelle1"]), str(z["zelle2"]), E.zahl(z["w1"], 4), E.zahl(z["wahr"], 4), E.zahl(z["fehler_h"], 3)] for z in zeilen])


def exp2_dirac_verdikt() -> str:
    s = E.Settings(zellen2=100)
    return E.verdikt2(s, E.reihe2(s))[1]


def exp2_glatt() -> str:
    zeilen = E.reihe2(E.Settings(profil2=C.PROFIL2[1], zellen2=100))
    return md(["Zellen B", "W1 im Raster", "W1 wahr", "Betrag des Fehlers"], [[E.ganz(z["zellen"]), E.zahl(z["w1"], 6), E.zahl(z["wahr"], 4), E.exp_text(abs(z["fehler"]), 1)] for z in zeilen])


def exp2_glatt_verdikt() -> str:
    s = E.Settings(profil2=C.PROFIL2[1], zellen2=100)
    return E.verdikt2(s, E.reihe2(s))[1]


def exp2_w2() -> str:
    zeilen = [E.w2_raster_zeile(b, sg) for b, sg in C.W2_ZEILEN]
    return md(["Zellen", "σ der zweiten Verteilung", "Zellen deckungsgleich", "h", "Fehler Raster − geschlossen", "Fehler / h²"],
              [[str(z["zellen"]), E.zahl(z["sigma2"], 1), "ja" if z["deckungsgleich"] else "nein", E.zahl(z["h"], 3), E.exp_text(z["fehler"], 2), E.zahl(z["fehler_h2"], 4) if abs(z["fehler_h2"]) > 1e-3 else E.exp_text(z["fehler_h2"], 1)] for z in zeilen])


def exp2_w2_verdikt() -> str:
    return E.verdikt2_w2([E.w2_raster_zeile(b, sg) for b, sg in C.W2_ZEILEN])[1]


# ---- Experiment 3 --------------------------------------------------------------------------------------------------------------------

def exp3_tabelle(d: float) -> str:
    zeilen = []
    for b in C.B3_REIHE:
        z = E.reihe3(C.SERIEN_SEED, 500, b, d, 300)
        zeilen.append([E.ganz(b), E.prozent(z["anteil_xy"], 1), E.prozent(z["anteil_yx"], 1), E.zahl(z["w1_hist_median"], 4), E.zahl(z["w1_sort_median"], 4)])
    return md(["Zellen B", "KL(x ‖ y) = ∞", "KL(y ‖ x) = ∞", "W1 aus dem Histogramm (Median)", "W1 aus sortierten Stichproben (Median)"], zeilen)


def exp3_verdikt() -> str:
    s = E.Settings()
    zeilen = [E.reihe3(s.seed_input, s.n3, b, s.d3, s.seeds3) for b in C.B3_REIHE]
    return E.verdikt3(s, zeilen)[1]


def exp3_klein() -> str:
    k = [E.kl_unendlich_exakt(b, 3) for b in (4, 8, 16)]
    return (f"P(KL = ∞) ist {k[0].numerator} / {E.ganz(k[0].denominator)} = {E.zahl(float(k[0]), 4)} bei B = 4, {E.zahl(float(k[1]), 4)} bei B = 8 und {E.zahl(float(k[2]), 4)} bei B = 16")


# ---- Experiment 4 --------------------------------------------------------------------------------------------------------------------

NAMEN4 = {"V-a": "V-a Nebenbedingung KL ≤ α", "V-b": "V-b ⟨C, P_ε⟩", "V-c": "V-c Genevay (Faktor 2)", "V-d": "V-d Feydy (Faktor ½)", "V-e": "V-e OT_ε ohne Zentrieren"}


def exp4_tabelle(c: float) -> str:
    k = E.analyse4(1, c)["kennzahlen"]
    zeilen = []
    for v in C.VERW4:
        z = k[v]
        kons = v in ("V-c", "V-d")
        zeilen.append([NAMEN4[v], E.zahl(z["selbst_mittel"], 4) + (" (Konstruktion)" if kons else ""), f"{z['nicht_naechster']} von 24" + (" (Konstruktion)" if kons else ""), E.zahl(z["kendall"], 3),
                       f"{z['nn_wie_exakt']} von 24", E.zahl(z["verhaeltnis"], 2), f"{E.ganz(z['verletzt'])} von {E.ganz(z['tripel'])}"])
    return md(["Verwendung", "Selbstabstand (Mittel)", "selbst nicht am nächsten", "Kendall τ gegen W2²", "nächster Nachbar wie exakt", "Median Wert / W2²", "Dreiecksungleichung verletzt"], zeilen)


def exp4_kopfzeile(c: float) -> str:
    res = E.analyse4(1, c)
    return (f"D = {E.zahl(res['d'], 4)}, ε = {E.zahl(res['eps'], 4)}; V-a: α = {E.zahl(res['alpha'], 4)}, {E.unwirksam_satz(res)}; "
            f"Läufe ohne Konvergenz innerhalb von {E.ganz(C.MAX_ITER4)} Durchläufen: {res['nicht_konvergiert']} bei V-b bis V-e, {res['nicht_konvergiert_a']} in den Bisektionsläufen von V-a")


def exp4_aufwand(c: float) -> str:
    res = E.analyse4(1, c)
    return f"im Median {E.ganz(res['produkte_median'])} Matrix-Vektor-Produkte je Paar, am größten {E.ganz(res['produkte_max'])}"


def exp4_verdikt(c: float) -> str:
    return E.verdikt4(E.analyse4(1, c))[1]


def spanne(werte, fmt) -> str:
    lo, hi = min(werte), max(werte)
    return fmt(lo) if lo == hi else f"{fmt(lo)} bis {fmt(hi)}"


def exp4_streuung(c: float) -> str:
    """Je Verwendung die Spanne über die Mengen 1 bis 5: Kendall τ und Zahl der Verteilungen, für die eine andere näher liegt als sie selbst."""
    daten = MESS["streuung"][repr(float(c))]["daten"]
    zeilen = []
    for v in C.VERW4:
        tau = [daten[str(m)][v]["kendall"] for m in daten]
        nn = [daten[str(m)][v]["nicht_naechster"] for m in daten]
        ver = [daten[str(m)][v]["verhaeltnis"] for m in daten]
        zeilen.append([v, spanne(tau, lambda x: E.zahl(x, 3)), spanne(nn, lambda x: f"{x}") + " von 24", spanne(ver, lambda x: E.zahl(x, 2))])
    return md(["Verwendung", "Kendall τ (Mengen 1 bis 5)", "selbst nicht am nächsten (Mengen 1 bis 5)", "Median Wert / W2² (Mengen 1 bis 5)"], zeilen)


def exp4_dreieck_streuung() -> str:
    """Verletzungen der Dreiecksungleichung der Wurzel je Verwendung und ε/D über die Mengen 1 bis 5 (Spanne)."""
    zeilen = []
    for v in C.VERW4:
        zel = [v]
        for c in (8.0, 3.0, 1.0, 0.3):
            daten = MESS["streuung"][repr(c)]["daten"]
            ver = [daten[str(m)][v]["verletzt"] for m in daten]
            zel.append(spanne(ver, str))
        zeilen.append(zel)
    return md(["Verwendung", "ε/D = 8", "ε/D = 3", "ε/D = 1", "ε/D = 0,3"], zeilen)


def exp4_c_vor_d() -> str:
    """Zählung: V-c liegt im Kendall-Rang vor V-d (je Menge 1 bis 5 und ε/D), getrennt für ε/D ≥ 3 und ε/D ≤ 1."""
    teile = []
    for namen, cs in (("ε/D = 8 und 3", (8.0, 3.0)), ("ε/D = 1 und 0,3", (1.0, 0.3))):
        diff = [MESS["streuung"][repr(c)]["daten"][m]["V-c"]["kendall"] - MESS["streuung"][repr(c)]["daten"][m]["V-d"]["kendall"] for c in cs for m in MESS["streuung"][repr(c)]["daten"]]
        vor = sum(1 for d in diff if d > 0)
        teile.append(f"bei {namen} liegt V-c in {vor} von {len(diff)} Zellen (Mengen 1 bis 5) im Kendall-Rang vor V-d, die Differenz V-c minus V-d ist {E.zahl(min(diff), 3)} bis {E.zahl(max(diff), 3)}")
    return "; ".join(teile)


def exp4_d_mengen() -> str:
    d = [MESS["d_mengen"][str(m)] for m in range(1, 21)]
    d5 = d[:5]
    return (f"D liegt in den Mengen 1 bis 20 zwischen {E.zahl(min(d), 4)} und {E.zahl(max(d), 4)} (Faktor {E.zahl(max(d) / min(d), 1)}), "
            f"in den hier gezeigten Mengen 1 bis 5 zwischen {E.zahl(min(d5), 4)} und {E.zahl(max(d5), 4)}")


# ---- Experiment 5 --------------------------------------------------------------------------------------------------------------------

def exp5_tabelle() -> str:
    tab = E.tabelle5(E.Settings())
    return md(["Prognose", "Lage²", "Streuung²", "W2²", "W1", "E[CRPS]", "E\\|X − y\\|"],
              [[z["name"], E.zahl(z["lage2"], 4), E.zahl(z["streu2"], 4), E.zahl(z["w2"], 4), E.zahl(z["w1"], 4), E.zahl(z["ecrps"], 4), E.zahl(z["exy"], 4)] for z in tab])


def exp5_raenge() -> str:
    tab = E.tabelle5(E.Settings())
    return "; ".join(f"{regel}: " + " < ".join(E.rangfolge(tab, key)) for regel, key in (("W2²", "w2"), ("W1", "w1"), ("E[CRPS]", "ecrps"), ("E\\|X − y\\|", "exy")))


def exp5_mc() -> str:
    zeilen = []
    for n in C.NOBS5:
        mc = E.mc5(E.Settings(nobs5=n, seeds5=400))
        zeilen.append([str(n)] + [E.prozent(mc["platz1"][r], 1) for r in ("W2²", "W1", "E[CRPS]", "E|X − y|")] + [f"{mc['f3_vor_f0']['E|X − y|']} von {mc['seeds']}"])
    return md(["Beobachtungen n_obs", "Platz 1: W2²", "Platz 1: W1", "Platz 1: E[CRPS]", "Platz 1: E\\|X − y\\|", "F3 vor F0 bei E\\|X − y\\|"], zeilen)


def exp5_verdikt() -> str:
    s = E.Settings()
    return E.verdikt5(s, E.tabelle5(s), E.mc5(s))[1]


STUECKE = {
    "exp1_kasten": exp1_kasten, "exp1_kasten_verdikt": exp1_kasten_verdikt, "exp1_spitzen": exp1_spitzen, "exp1_spitzen_verdikt": exp1_spitzen_verdikt, "exp1_glaettung": exp1_glaettung,
    "exp1_kreis": exp1_kreis, "exp1_breite": exp1_breite, "exp2_dirac": exp2_dirac, "exp2_dirac_verdikt": exp2_dirac_verdikt, "exp2_glatt": exp2_glatt, "exp2_glatt_verdikt": exp2_glatt_verdikt,
    "exp2_w2": exp2_w2, "exp2_w2_verdikt": exp2_w2_verdikt, "exp3_tabelle_0": lambda: exp3_tabelle(0.0), "exp3_tabelle_002": lambda: exp3_tabelle(0.02), "exp3_tabelle_005": lambda: exp3_tabelle(0.05),
    "exp3_verdikt": exp3_verdikt, "exp3_klein": exp3_klein, "exp4_tabelle_8": lambda: exp4_tabelle(8.0), "exp4_kopf_8": lambda: exp4_kopfzeile(8.0), "exp4_aufwand_8": lambda: exp4_aufwand(8.0),
    "exp4_verdikt_8": lambda: exp4_verdikt(8.0), "exp4_tabelle_1": lambda: exp4_tabelle(1.0), "exp4_kopf_1": lambda: exp4_kopfzeile(1.0), "exp4_aufwand_1": lambda: exp4_aufwand(1.0),
    "exp4_verdikt_1": lambda: exp4_verdikt(1.0), "exp4_tabelle_03": lambda: exp4_tabelle(0.3), "exp4_kopf_03": lambda: exp4_kopfzeile(0.3), "exp4_streuung_8": lambda: exp4_streuung(8.0),
    "exp4_streuung_1": lambda: exp4_streuung(1.0), "exp4_dreieck_streuung": exp4_dreieck_streuung, "exp4_d_mengen": exp4_d_mengen, "exp4_c_vor_d": exp4_c_vor_d, "exp5_tabelle": exp5_tabelle, "exp5_raenge": exp5_raenge,
    "exp5_mc": exp5_mc, "exp5_verdikt": exp5_verdikt,
}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for name, f in STUECKE.items():
        print(f"##### {name}\n{f()}\n")
