"""Kennzahlen, Reihen und Verdikte der fünf Experimente (reine Funktionen, ohne Streamlit).

Jede Zahl, die die App als Befund zeigt, wird hier aus der Rechnung gebildet (nie hartcodiert). Verdikte sind Funktionen von Zahlen und damit an handgesetzten Werten testbar.
Formatierung: Dezimalkomma, Tausender mit Leerzeichen."""

import math
from dataclasses import dataclass, fields
from fractions import Fraction
from statistics import NormalDist

import numpy as np

import vab_abstand as A
import vab_constants as C
import vab_profile as PR
import vab_sinkhorn as S


# ---------------------------------------------------------------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    seed_input: int = C.DEFAULTS["seed_input"]
    profil1: str = C.DEFAULTS["profil1"]
    s1: float = C.DEFAULTS["s1"]
    breite1: float = C.DEFAULTS["breite1"]
    zellen1: int = C.DEFAULTS["zellen1"]
    achse1: str = C.DEFAULTS["achse1"]
    glaettung1: float = C.DEFAULTS["glaettung1"]
    profil2: str = C.DEFAULTS["profil2"]
    zellen2: int = C.DEFAULTS["zellen2"]
    abstand2: float = C.DEFAULTS["abstand2"]
    breite2: float = C.DEFAULTS["breite2"]
    n3: int = C.DEFAULTS["n3"]
    b3: int = C.DEFAULTS["b3"]
    d3: float = C.DEFAULTS["d3"]
    richtung3: str = C.DEFAULTS["richtung3"]
    seeds3: int = C.DEFAULTS["seeds3"]
    c4: float = C.DEFAULTS["c4"]
    verw4: str = C.DEFAULTS["verw4"]
    menge4: int = C.DEFAULTS["menge4"]
    f1_lage: float = C.DEFAULTS["f1_lage"]
    f2_sigma: float = C.DEFAULTS["f2_sigma"]
    f3_sigma: float = C.DEFAULTS["f3_sigma"]
    f4_lage: float = C.DEFAULTS["f4_lage"]
    f4_sigma: float = C.DEFAULTS["f4_sigma"]
    nobs5: int = C.DEFAULTS["nobs5"]
    seeds5: int = C.DEFAULTS["seeds5"]


def settings_aus_zustand(zustand) -> Settings:
    """Settings aus einem Mapping (Session State oder Preset); fehlende Schlüssel nehmen den Standard."""
    return Settings(**{f.name: zustand.get(f.name, getattr(Settings(), f.name)) for f in fields(Settings)})


# ---------------------------------------------------------------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------------------------------------------------------------

def zahl(x: float, stellen: int = 3) -> str:
    """Dezimalkomma; ∞ als Zeichen; −0 wird zu 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    text = f"{x:.{stellen}f}".replace(".", ",")
    return text[1:] if text.startswith("-0") and float(text.replace(",", ".")) == 0 else text.replace("-", "−")


def ganz(n) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(n):,}".replace(",", " ")


def exp_text(x: float, stellen: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise mit Dezimalkomma, z. B. 1,2e-12."""
    if x == 0:
        return "0"
    return f"{x:.{stellen}e}".replace(".", ",").replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e").replace("-", "−", 1) if x < 0 else f"{x:.{stellen}e}".replace(".", ",").replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e")


def prozent(x: float, stellen: int = 1) -> str:
    return zahl(100.0 * x, stellen) + " %"


def sd_prozentpunkte(p: float, n: int) -> float:
    """Standardabweichung eines Anteils p aus n unabhängigen Wiederholungen in Prozentpunkten: 100 · √(p (1 − p) / n)."""
    return 100.0 * math.sqrt(p * (1.0 - p) / n)


def glaettung_text(g: float) -> str:
    return "keine Glättung (g = 0)" if g == 0 else f"g = {exp_text(g, 0)}"


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Verschieben statt Überlappen
# ---------------------------------------------------------------------------------------------------------------------------------

def profil1(s: Settings, verschiebung: float) -> np.ndarray:
    kreis = s.achse1 == "Kreis"
    if s.profil1 == C.PROFIL1[0]:
        return PR.kasten(s.zellen1, C.KASTEN_START, C.KASTEN_BREITE, verschiebung, kreis)
    return PR.zwei_spitzen(s.zellen1, verschiebung, s.breite1, kreis, C.SPITZEN_LAGEN, C.SPITZEN_GEWICHTE)


def reihe1(s: Settings) -> dict:
    """Alle Abstände zwischen dem Ausgangsprofil und dem um s verschobenen Profil für s = 0 bis 8 Stunden (Schritt 0,5)."""
    h = C.TAG / s.zellen1
    a = profil1(s, 0.0)
    kreis = s.achse1 == "Kreis"
    zeilen = {"s": list(C.S1_WERTE), "w1": [], "tv": [], "ks": [], "js": [], "kl": [], "kl_roh": []}
    for sv in C.S1_WERTE:
        b = profil1(s, sv)
        zeilen["w1"].append((A.w1_kreis if kreis else A.w1)(a, b, h))
        zeilen["tv"].append(A.tv(a, b))
        zeilen["ks"].append(A.ks(a, b))
        zeilen["js"].append(A.js(a, b))
        zeilen["kl"].append(A.kl(b, a, s.glaettung1))          # KL(verschoben ‖ Ausgang)
        zeilen["kl_roh"].append(A.kl(b, a, 0.0))
    return zeilen


def zeile1(reihe: dict, sv: float) -> dict:
    i = reihe["s"].index(sv)
    return {k: reihe[k][i] for k in ("w1", "tv", "ks", "js", "kl", "kl_roh")}


def kl_gegen_glaettung(s: Settings, verschiebung: float) -> dict:
    """KL(verschoben ‖ Ausgang) bei den vier Glättungsstufen (Experiment 1, Tabelle zur Glättung)."""
    a, b = profil1(s, 0.0), profil1(s, verschiebung)
    return {g: A.kl(b, a, g) for g in C.GLAETTUNG1}


def verdikt1(s: Settings, reihe: dict) -> tuple[str, str]:
    """Verdikt zu Experiment 1 aus der Reihe (Zahlen und Bedingungen der Einstellung): beim Kasten über alle Verschiebungen, bei zwei Spitzen eine Zeile je Verschiebung der Tabelle."""
    sv = reihe["s"]
    tv = np.array(reihe["tv"])
    ks = np.array(reihe["ks"])
    js = np.array(reihe["js"])
    kl_roh = np.array(reihe["kl_roh"])
    ort = f"{s.profil1}, {s.zellen1} Zellen, Zeitachse {s.achse1}"
    if s.profil1 == C.PROFIL1[1]:
        zs = [zeile1(reihe, x) for x in C.S_ZEILEN]
        liste = " / ".join(zahl(x, 0) for x in C.S_ZEILEN)
        kl_text = ", ".join(zahl(z["kl"], 3) if math.isfinite(z["kl"]) else "∞" for z in zs)
        teile = [f"W1 = " + " / ".join(zahl(z["w1"], 3) for z in zs) + " Stunden", "TV = " + " / ".join(zahl(z["tv"], 3) for z in zs),
                 (f"KL ohne Glättung = {kl_text}" if s.glaettung1 == 0 else f"KL (mit {glaettung_text(s.glaettung1)}) = {kl_text}")]
        i_max = int(np.argmax(tv))
        if tv[-1] < tv[i_max] - 1e-9:
            teile.append(f"TV ist nicht monoton: Höchstwert {zahl(float(tv[i_max]), 3)} bei s = {zahl(sv[i_max], 1)}, bei s = {zahl(sv[-1], 1)} wieder {zahl(float(tv[-1]), 3)}")
        return "info", f"{ort}, σ = {zahl(s.breite1, 1)} Stunden, bei s = {liste} Stunden: " + "; ".join(teile) + ". In diesem Modell, eine Zeile je Verschiebung (Tabelle)."
    w1 = np.array(reihe["w1"])
    abw = float(np.max(np.abs(w1 - np.array(sv))))
    teile = [f"W1 = s Stunden in allen {len(sv)} Verschiebungen" if abw < 1e-9 else f"W1 weicht um höchstens {zahl(abw, 3)} Stunden von s ab"]
    teile.append("TV = KS in allen Verschiebungen" if np.max(np.abs(tv - ks)) < 1e-9 else f"TV und KS weichen um bis zu {zahl(float(np.max(np.abs(tv - ks))), 3)} voneinander ab")
    sat_tv = next((x for x, t in zip(sv, tv) if t >= 1 - 1e-9), None)
    sat_js = next((x for x, t in zip(sv, js) if t >= C.LN2 - 1e-9), None)
    unendlich = next((x for x, t in zip(sv, kl_roh) if math.isinf(t)), None)
    teile.append(f"TV sättigt bei 1 ab s = {zahl(sat_tv, 1)} Stunden" if sat_tv is not None else f"TV erreicht 1 nicht (Höchstwert {zahl(float(tv.max()), 3)})")
    teile.append(f"JS erreicht ln 2 = {zahl(C.LN2, 3)} ab s = {zahl(sat_js, 1)} Stunden" if sat_js is not None else f"JS bleibt unter ln 2 (Höchstwert {zahl(float(js.max()), 3)})")
    teile.append(f"KL ist ohne Glättung schon ab s = {zahl(unendlich, 1)} Stunden unendlich (jede Verschiebung füllt eine Zelle, die im Ausgangsprofil leer ist)" if unendlich is not None
                 else f"KL ist in keiner Verschiebung unendlich (Höchstwert {zahl(float(kl_roh[np.isfinite(kl_roh)].max()), 3)} nat)")
    return "info", f"{ort}: " + "; ".join(teile) + ". In diesem Modell, eine Zeile je Verschiebung."


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Das Raster macht die Zahl
# ---------------------------------------------------------------------------------------------------------------------------------

def treppe_zeile(zellen: int, abstand: float, lage: str = C.LAGE2) -> dict:
    """Zwei Spitzen (Dirac) bei `lage` und `lage + abstand` des Tages (Brüche, exakt): W1 im Raster, wahres W1, Fehler und Fehler in Zellenbreiten."""
    x1 = Fraction(lage)
    x2 = x1 + Fraction(str(abstand))
    z1, z2 = math.floor(x1 * zellen), math.floor(x2 * zellen)
    w1_raster = Fraction(abs(z2 - z1), zellen)
    wahr = x2 - x1
    fehler = w1_raster - wahr
    return {"zellen": zellen, "zelle1": z1, "zelle2": z2, "w1": float(w1_raster), "wahr": float(wahr), "fehler": float(fehler), "fehler_h": float(abs(fehler) * zellen)}


def glatt_zeile(zellen: int, abstand: float, sigma: float, lage: float = 0.3) -> dict:
    """Zwei Normalverteilungen gleicher Breite sigma (Anteile des Tages) im Abstand `abstand`; Zellenintegrale als Massen auf [0, 1]; wahres W1 = Abstand."""
    rand = np.linspace(0.0, 1.0, zellen + 1)
    a = PR.gauss_zellenmassen(rand, lage, sigma)
    b = PR.gauss_zellenmassen(rand, lage + abstand, sigma)
    w1_raster = A.w1(a, b, 1.0 / zellen)
    fehler = w1_raster - abstand
    return {"zellen": zellen, "w1": w1_raster, "wahr": abstand, "fehler": fehler, "fehler_h": abs(fehler) * zellen}


def reihe2(s: Settings) -> list[dict]:
    """Eine Zeile je Zellenzahl der Reihe (und die gewählte, falls sie nicht dazugehört)."""
    stufen = sorted(set(C.ZELLEN2_REIHE) | {s.zellen2})
    if s.profil2 == C.PROFIL2[0]:
        return [treppe_zeile(b, s.abstand2) for b in stufen]
    return [glatt_zeile(b, s.abstand2, s.breite2) for b in stufen]


def w2_raster_zeile(zellen: int, sigma2: float) -> dict:
    """W2² zwischen N(0, 1) und N(1,5; σ₂²) auf [−6, 8], Zellenintegrale als Massen (σ₂ = 0,7: ungleiche, σ₂ = 1: gleiche Streuung).

    `deckungsgleich`: die Verschiebung 1,5 ist ein Vielfaches der Zellenbreite h = 14/Zellen (dann ist bei gleicher Streuung das zweite Raster eine Verschiebung des ersten um ganze Zellen).
    Rückgabe: Fehler Raster − geschlossen und Fehler / h²."""
    rand = np.linspace(-6.0, 8.0, zellen + 1)
    mitte = 0.5 * (rand[1:] + rand[:-1])
    h = 14.0 / zellen
    a = PR.gauss_zellenmassen(rand, 0.0, 1.0)
    b = PR.gauss_zellenmassen(rand, 1.5, sigma2)
    wahr = 1.5 ** 2 + (1.0 - sigma2) ** 2
    fehler = A.w2_quadrat(mitte, a, b) - wahr
    in_h = 1.5 / h
    return {"zellen": zellen, "sigma2": sigma2, "deckungsgleich": abs(in_h - round(in_h)) < 1e-9, "h": h, "wahr": wahr, "fehler": fehler, "fehler_h2": fehler / h ** 2, "verschiebung_in_h": in_h}


def verdikt2(s: Settings, zeilen: list[dict]) -> tuple[str, str]:
    d = s.abstand2
    if s.profil2 == C.PROFIL2[0]:
        null = [z["zellen"] for z in zeilen if z["w1"] == 0.0]
        gross = max(zeilen, key=lambda z: z["fehler_h"])
        teile = [f"Zwei Spitzen im Abstand {zahl(d, 4)} des Tages: Das Raster meldet W1 = 0 bei " + (", ".join(str(b) for b in null) + " Zellen" if null else "keiner der gerechneten Zellenzahlen")
                 + f" (beide Spitzen in einer Zelle), der wahre Wert ist {zahl(d, 4)}",
                 f"der größte Fehler ist {zahl(gross['fehler_h'], 2)} Zellenbreiten bei {gross['zellen']} Zellen"]
        if all(abs(z["fehler"]) <= 1.0 / z["zellen"] + 1e-15 for z in zeilen):
            teile.append("der Fehler bleibt in jeder Zeile unter einer Zellenbreite")
        else:
            teile.append("in einer Zeile ist der Fehler größer als eine Zellenbreite")
        return "info", "; ".join(teile) + ". Die Fehler bilden eine Treppe, kein glattes Konvergieren."
    ab50 = [z for z in zeilen if z["zellen"] >= 50]
    groesster = max((abs(z["fehler"]) for z in ab50), default=float("nan"))
    unter10 = [z["zellen"] for z in zeilen if z["zellen"] < 50 and abs(z["fehler"]) > 1e-6]
    deutung = " (Rundung)" if groesster < C.RUNDUNG_GRENZE else ""            # Rundungsfehler liegen bei 1e-15 bis 1e-17; darüber steht der Wert ohne Deutung
    return "info", (f"Glatte Gauß-Profile gleicher Breite σ = {zahl(s.breite2, 2)} des Tages im Abstand {zahl(d, 4)}: ab 50 Zellen ist der größte Fehler {exp_text(groesster, 1)}{deutung}, "
                    + (f"bei {', '.join(str(b) for b in unter10)} Zellen größer als 1e-6" if unter10 else "auch unter 50 Zellen nirgends größer als 1e-6")
                    + ". Der Fehler hängt hier an der Glätte der Profile, nicht an der Zellenzahl allein.")


def verdikt2_w2(zeilen: list[dict]) -> tuple[str, str]:
    def teil(sigma_gleich: bool, deckungsgleich: bool) -> list[dict]:
        return [z for z in zeilen if (z["sigma2"] == 1.0) == sigma_gleich and z["deckungsgleich"] == deckungsgleich]

    def werte(zs, exp=False):
        return ", ".join(exp_text(abs(z["fehler_h2"]), 0) if exp else zahl(z["fehler_h2"], 4) for z in zs)

    def zellen(zs):
        return ", ".join(str(z["zellen"]) for z in zs)

    ung, ung_d, gl, gl_d = teil(False, False), teil(False, True), teil(True, False), teil(True, True)
    alle_ung = ung + ung_d
    lo, hi = min(z["fehler_h2"] for z in alle_ung), max(z["fehler_h2"] for z in alle_ung)
    positiv = "in allen " + str(len(alle_ung)) + " Zeilen positiv (das Raster überschätzt)" if all(z["fehler_h2"] > 0 for z in alle_ung) else "nicht in allen Zeilen positiv"
    return "info", (f"W2² zwischen N(0, 1) und N(1,5; σ₂²) auf [−6, 8]. Bei ungleicher Streuung (σ₂ = 0,7) ist der Rasterfehler geteilt durch h² {werte(ung)} bei {zellen(ung)} Zellen "
                    f"und {werte(ung_d)} bei {zellen(ung_d)} Zellen (dort ist die Verschiebung 1,5 ein Vielfaches von h): {positiv}, zwischen {zahl(lo, 4)} und {zahl(hi, 4)} (1/6 = {zahl(1 / 6, 4)}), "
                    f"auch bei deckungsgleichen Zellen. Bei gleicher Streuung (σ₂ = 1) ist er {werte(gl)} bei {zellen(gl)} Zellen und schwankt mit der Zellenzahl; liegen die Zellen deckungsgleich "
                    f"(Verschiebung 1,5 ein Vielfaches von h bei {zellen(gl_d)} Zellen), ist er nur {werte(gl_d, True)} (Abschneiden der Ränder), weil das zweite Raster dann das um ganze Zellen "
                    f"verschobene erste ist. Die Konstante nahe 1/6 gilt hier also bei ungleicher Streuung, unabhängig von der Lage der Zellen, und nicht bei gleicher Streuung.")


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Histogramm ohne Unterschied
# ---------------------------------------------------------------------------------------------------------------------------------

def kl_unendlich_exakt(zellen: int, n: int) -> Fraction:
    """P(KL = ∞) für zwei unabhängige Stichproben mit je n Beobachtungen aus derselben Gleichverteilung über `zellen` Zellen (exakt, Inklusion-Exklusion).

    Der Träger von p ist nicht im Träger von q enthalten. P(Träger von p = S) hängt nur von |S| = k ab: C(B, k) Mengen, je surj(n, k)/B^n; P(q besetzt jede Zelle von S) =
    Σ_j (−1)^j C(k, j) ((B − j)/B)^n."""
    def surj(n_, k_):
        return sum((-1) ** j * math.comb(k_, j) * (k_ - j) ** n_ for j in range(k_ + 1))

    gesamt = Fraction(0)
    for k in range(1, min(zellen, n) + 1):
        p_stuetze = Fraction(math.comb(zellen, k) * surj(n, k), zellen ** n)
        p_deckt = sum((-1) ** j * math.comb(k, j) * Fraction(zellen - j, zellen) ** n for j in range(k + 1))
        gesamt += p_stuetze * (1 - p_deckt)
    return gesamt


def histogramme(x: np.ndarray, zellen: int) -> np.ndarray:
    """Zellenzählungen je Zeile (Zufalls-Seed) für Werte in [0, 1): Form (Zeilen, Zellen)."""
    z = np.clip(np.floor(x * zellen).astype(np.int64), 0, zellen - 1)
    zeilen = np.arange(len(x))[:, None] * zellen + z
    return np.bincount(zeilen.ravel(), minlength=len(x) * zellen).reshape(len(x), zellen)


def stichproben(s_seed: int, n: int, d: float, seeds: int):
    """x ~ N(0,3; 0,04²), y ~ N(0,3 + d; 0,04²), je n Beobachtungen, `seeds` unabhängige Wiederholungen (Zufalls-Seed der Demo, Zweck HIST)."""
    zx = PR.zufall_normal(s_seed, C.ZWECK_HIST, seeds, 2 * n)
    return C.MU3 + C.SIGMA3 * zx[:, :n], C.MU3 + d + C.SIGMA3 * zx[:, n:]


def reihe3(s_seed: int, n: int, zellen: int, d: float, seeds: int) -> dict:
    """Für `seeds` Wiederholungen: Anteil mit KL = ∞ (beide Richtungen, ohne Glättung), W1 aus den Histogrammen und W1 über die sortierten Stichproben."""
    x, y = stichproben(s_seed, n, d, seeds)
    hx = histogramme(x, zellen) / n
    hy = histogramme(y, zellen) / n
    inf_xy = np.any((hx > 0) & (hy <= 0), axis=1)          # KL(x ‖ y): x hat etwas, wo y nichts hat
    inf_yx = np.any((hy > 0) & (hx <= 0), axis=1)
    w1_hist = np.sum(np.abs(np.cumsum(hx, axis=1) - np.cumsum(hy, axis=1)), axis=1) / zellen
    w1_sort = np.mean(np.abs(np.sort(x, axis=1) - np.sort(y, axis=1)), axis=1)   # gleiche Stichprobengröße: W1 = Mittel der Differenzen der Ordnungsstatistiken
    return {"zellen": zellen, "n": n, "d": d, "seeds": seeds, "anteil_xy": float(inf_xy.mean()), "anteil_yx": float(inf_yx.mean()),
            "w1_hist": w1_hist, "w1_sort": w1_sort,
            "w1_hist_median": float(np.median(w1_hist)), "w1_hist_q": (float(np.quantile(w1_hist, 0.05)), float(np.quantile(w1_hist, 0.95))),
            "w1_sort_median": float(np.median(w1_sort)), "w1_sort_q": (float(np.quantile(w1_sort, 0.05)), float(np.quantile(w1_sort, 0.95)))}


def verdikt3(s: Settings, zeilen: list[dict]) -> tuple[str, str]:
    """`zeilen`: die Reihe über B3_REIHE (gleiche n, d, Zufalls-Seeds)."""
    xy = ", ".join(f"{zahl(100 * z['anteil_xy'], 1)}" for z in zeilen)
    yx = ", ".join(f"{zahl(100 * z['anteil_yx'], 1)}" for z in zeilen)
    zellen = " / ".join(ganz(z["zellen"]) for z in zeilen)
    gleich = "derselben Verteilung" if s.d3 == 0 else f"um d = {zahl(s.d3, 2)} verschobenen Verteilungen"
    return "info", (f"n = {s.n3} aus N(0,3; 0,04²) und N(0,3 + d; 0,04²) mit d = {zahl(s.d3, 2)} (Stichproben aus {gleich}), ohne Glättung, {s.seeds3} Zufalls-Seeds: "
                    f"KL(x ‖ y) = ∞ in {xy} % der Zufalls-Seeds bei {zellen} Zellen; KL(y ‖ x) in {yx} %. W1 aus demselben Histogramm bleibt endlich "
                    f"(Median bei {zeilen[0]['zellen']} Zellen {zahl(zeilen[0]['w1_hist_median'], 4)}, über die sortierten Stichproben {zahl(zeilen[0]['w1_sort_median'], 4)}). "
                    f"Jeder Anteil ist aus {s.seeds3} Zufalls-Seeds gezählt und hat eine Standardabweichung von höchstens {zahl(sd_prozentpunkte(0.5, s.seeds3), 1)} Prozentpunkten "
                    f"(bei 8 % etwa {zahl(sd_prozentpunkte(0.08, s.seeds3), 1)}, bei 99 % etwa {zahl(sd_prozentpunkte(0.99, s.seeds3), 1)}); Unterschiede von wenigen Prozentpunkten zwischen zwei Anteilen sind Rauschen. "
                    f"Das ist ein Beispiel mit einer Dimension und einem n; wie der Rauschboden mit n und Dimension wächst, zeigt diese Demo nicht.")


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Distanz oder Divergenz?
# ---------------------------------------------------------------------------------------------------------------------------------

def menge4(menge: int):
    """(Profile 24 × 48, Zellenmitten x in Stunden, Kosten C mit max C = 1, exaktes W2² je Paar in der Normierung max C = 1, Paare)."""
    prof = PR.profil_menge(menge, C.ANZAHL4, C.ZELLEN4, C.ZWECK_MENGE)
    x = PR.zellenmitten(C.ZELLEN4)
    kosten = (x[:, None] - x[None, :]) ** 2
    kosten = kosten / kosten.max()
    skala = (x[-1] - x[0]) ** 2
    paare = S.paarliste(C.ANZAHL4)
    exakt = np.array([A.w2_quadrat(x, prof[i], prof[j]) for i, j in paare]) / skala
    return prof, x, kosten, exakt, paare


def pruefsumme_menge(prof: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.round(prof, 9).tobytes()).hexdigest()[:16]


def _paarvektor(mat: np.ndarray, paare) -> np.ndarray:
    return np.array([mat[i, j] for i, j in paare])


def kennzahlen4(mat: np.ndarray, exakt: np.ndarray, paare, nn_exakt: np.ndarray) -> dict:
    """Kennzahlen einer Verwendung: Selbstabstand (Mittel), Zahl der Verteilungen, für die eine andere näher liegt als sie selbst, Kendall τ gegen exaktes W2², nächster Nachbar wie
    beim exakten W2², Median Wert / W2², Dreiecksverletzungen."""
    anzahl = len(mat)
    diag = np.diag(mat)
    ohne = mat + np.diag(np.full(anzahl, np.inf))
    nicht_naechster = int(np.sum(diag > ohne.min(axis=1) + 1e-15))
    vec = _paarvektor(mat, paare)
    verletzt, tripel, negativ = S.dreiecksverletzungen(mat)
    return {"selbst_mittel": float(diag.mean()), "nicht_naechster": nicht_naechster, "kendall": A.kendall_tau(vec, exakt),
            "nn_wie_exakt": int(np.sum(ohne.argmin(axis=1) == nn_exakt)), "verhaeltnis": float(np.median(vec / exakt)),
            "verletzt": verletzt, "tripel": tripel, "negativ": negativ}


def analyse4(menge: int, c: float) -> dict:
    """Alle fünf Verwendungen für eine Verteilungsmenge und ε = c · D (D = Median von W2² über die 276 Paare, max C = 1)."""
    prof, x, kosten, exakt, paare = menge4(menge)
    anzahl = len(prof)
    d_med = float(np.median(exakt))
    eps = c * d_med
    w = S.werte_b_bis_e(prof, kosten, eps, C.TOL4, C.MAX_ITER4)
    kl_paare = _paarvektor(w.kl_plan, paare)
    alpha = float(np.median(kl_paare))
    m_a, unwirksam, nk_a, maske_a = S.wert_nebenbedingung(prof, kosten, alpha, C.TOL4_A, max_iter=C.MAX_ITER4, mit_maske=True)
    verh_unwirksam = (_paarvektor(m_a, paare) / exakt)[maske_a]       # Wert bei ε' = EPS_MIN_A gegen W2², nur Paare mit unwirksamer Nebenbedingung
    mats = {"V-a": m_a, "V-b": w.kosten_plan, "V-c": w.v_c, "V-d": w.v_d, "V-e": w.ot}
    exakt_mat = np.zeros((anzahl, anzahl))
    for (i, j), v in zip(paare, exakt):
        exakt_mat[i, j] = exakt_mat[j, i] = v
    nn_exakt = (exakt_mat + np.diag(np.full(anzahl, np.inf))).argmin(axis=1)
    kenn = {name: kennzahlen4(m, exakt, paare, nn_exakt) for name, m in mats.items()}
    produkte = _paarvektor(w.produkte, paare)
    return {"menge": menge, "c": c, "eps": eps, "d": d_med, "alpha": alpha, "unwirksam_a": unwirksam,
            "unwirksam_median": float(np.median(verh_unwirksam)) if len(verh_unwirksam) else None, "unwirksam_max": float(verh_unwirksam.max()) if len(verh_unwirksam) else None,
            "eps_min_durch_d": S.EPS_MIN_A / d_med, "nicht_konvergiert": w.nicht_konvergiert, "nicht_konvergiert_a": nk_a, "matrizen": mats, "kennzahlen": kenn,
            "exakt": exakt, "exakt_matrix": exakt_mat, "paare": paare, "produkte_median": float(np.median(produkte)), "produkte_max": float(produkte.max()), "pruefsumme": pruefsumme_menge(prof),
            "eps_durch_w2": None}


def unwirksam_satz(res: dict) -> str:
    """Satz zu V-a: bei wie vielen Paaren die Nebenbedingung nicht wirksam ist und was die Tabelle dann zeigt (der Wert beim kleinsten ε', eine obere Schranke)."""
    n = res["unwirksam_a"]
    satz = f"bei {n} von 276 Paaren ist die Nebenbedingung nicht wirksam"
    if n == 0:
        return satz
    return (satz + f" (dort zeigt die Tabelle den Wert beim kleinsten ε′ = {zahl(S.EPS_MIN_A, 3)}, das sind {zahl(res['eps_min_durch_d'], 2)} D: eine obere Schranke des Werts unter der Nebenbedingung, "
            f"im Median das {zahl(res['unwirksam_median'], 2)}-Fache, höchstens das {zahl(res['unwirksam_max'], 2)}-Fache des exakten W2²)")


def verhaeltnis_text(k: dict, name: str) -> str:
    return zahl(k[name]["verhaeltnis"], 2)


def verdikt4(res: dict) -> tuple[str, str]:
    k = res["kennzahlen"]
    c = res["c"]
    teile = [f"bei ε/D = {zahl(c, 1)} hat V-b für {k['V-b']['nicht_naechster']} der {len(res['matrizen']['V-b'])} Profile ein anderes Profil näher als sich selbst "
             f"(V-e: {k['V-e']['nicht_naechster']}, V-a: {k['V-a']['nicht_naechster']}); V-c und V-d haben den Selbstabstand 0 durch Konstruktion",
             f"der Median von Wert / W2² ist bei V-c {verhaeltnis_text(k, 'V-c')} und bei V-d {verhaeltnis_text(k, 'V-d')}",
             f"Kendall τ gegen das exakte W2²: V-b {zahl(k['V-b']['kendall'], 3)}, V-c {zahl(k['V-c']['kendall'], 3)}, V-d {zahl(k['V-d']['kendall'], 3)}"]
    teile.append(f"die Dreiecksungleichung der Wurzel ist bei V-c in {ganz(k['V-c']['verletzt'])}, bei V-d in {ganz(k['V-d']['verletzt'])}, bei V-a in {ganz(k['V-a']['verletzt'])} und bei V-b in "
                 f"{ganz(k['V-b']['verletzt'])} von {ganz(k['V-b']['tripel'])} Tripeln verletzt")
    return "info", (f"Verteilungsmenge {res['menge']}, 24 Verteilungen, 276 Paare: " + "; ".join(teile) + ". Gilt für diese Menge einer Familie (Mischungen von 1 bis 3 Gauß-Hügeln) und diese ε/D, "
                    "nicht als Aussage über alle Verteilungen.")


# ---------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: Prognosen bewerten
# ---------------------------------------------------------------------------------------------------------------------------------

def prognosen(s: Settings) -> list[tuple[str, float, float]]:
    """(Name, Lage, σ) der fünf Prognosen; die Wahrheit ist N(0, 1)."""
    return [("F0 wahr", 0.0, 1.0), (f"F1 Lage {zahl(s.f1_lage, 2)}", s.f1_lage, 1.0), (f"F2 σ = {zahl(s.f2_sigma, 2)}", 0.0, s.f2_sigma),
            (f"F3 σ = {zahl(s.f3_sigma, 2)}", 0.0, s.f3_sigma), (f"F4 Lage {zahl(s.f4_lage, 2)}, σ = {zahl(s.f4_sigma, 2)}", s.f4_lage, s.f4_sigma)]


def _simpson(y: np.ndarray, h: float) -> float:
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def tabelle5(s: Settings) -> list[dict]:
    """Je Prognose: W2² (Lage² + Streuung²), W1 und Cramér-Abstand² (Quadratur der Verteilungsfunktionsdifferenz), E[CRPS] und E|X − y| (geschlossene Formeln) gegen N(0, 1)."""
    x = np.linspace(-12.0, 12.0, 20001)
    h = x[1] - x[0]
    g = A.normal_cdf(x)
    zeilen = []
    for name, m, sg in prognosen(s):
        f = A.normal_cdf((x - m) / sg)
        sd = math.sqrt(sg * sg + 1.0)
        exy = sd * math.sqrt(2.0 / math.pi) * math.exp(-m * m / (2 * sd * sd)) + m * (1.0 - 2.0 * float(A.normal_cdf(-m / sd)))
        cramer2 = _simpson((f - g) ** 2, h)
        zeilen.append({"name": name, "lage": m, "sigma": sg, "lage2": m * m, "streu2": (sg - 1.0) ** 2, "w2": m * m + (sg - 1.0) ** 2, "w1": _simpson(np.abs(f - g), h),
                       "cramer2": cramer2, "ecrps": exy - sg / math.sqrt(math.pi), "exy": exy})
    return zeilen


def rangfolge(zeilen: list[dict], key: str) -> list[str]:
    """Namen (nur der Kürzel F0 bis F4) von klein nach groß."""
    return [z["name"].split()[0] for z in sorted(zeilen, key=lambda z: z[key])]


def _phi(z):
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


def _g_stamm(z):
    """G(z) = z Φ(z) + φ(z), Stammfunktion von Φ."""
    return z * A.normal_cdf(z) + _phi(z)


def w1_gegen_stichprobe(y_sortiert: np.ndarray, m: float, sg: float) -> np.ndarray:
    """W1 zwischen N(m, σ²) und der empirischen Verteilung der Zeilen von y_sortiert (Form (R, n), je Zeile aufsteigend), exakt: Integral von |F − F_n| stückweise geschlossen."""
    r, n = y_sortiert.shape
    z = (y_sortiert - m) / sg
    stamm = sg * _g_stamm(z)                                                   # ∫ F bis x
    links = stamm[:, 0]                                                        # (−∞, y_1): F_n = 0
    rechts = sg * _g_stamm(-z[:, -1])                                          # (y_n, ∞): 1 − F
    nd = NormalDist()
    c_werte = np.arange(1, n) / n
    xc = m + sg * np.array([nd.inv_cdf(float(c)) for c in c_werte])           # F(xc) = k/n
    lo, hi = y_sortiert[:, :-1], y_sortiert[:, 1:]
    xk = np.clip(xc[None, :], lo, hi)
    s_lo, s_hi = stamm[:, :-1], stamm[:, 1:]
    s_xk = sg * _g_stamm((xk - m) / sg)
    cc = c_werte[None, :]
    innen = (cc * (xk - lo) - (s_xk - s_lo)) + ((s_hi - s_xk) - cc * (hi - xk))
    return links + rechts + innen.sum(axis=1)


def w2_gegen_stichprobe(y_sortiert: np.ndarray, m: float, sg: float) -> np.ndarray:
    """W2² zwischen N(m, σ²) und der empirischen Verteilung (Zeilen von y_sortiert): Σ_i ∫ über das Quantilintervall ((i−1)/n, i/n) von (Q(u) − y_i)² du, geschlossen."""
    r, n = y_sortiert.shape
    nd = NormalDist()
    zg = np.array([-np.inf] + [nd.inv_cdf(i / n) for i in range(1, n)] + [np.inf])
    endlich = np.isfinite(zg)
    z0 = np.where(endlich, zg, 0.0)
    ph = np.where(endlich, _phi(z0), 0.0)
    zph = np.where(endlich, z0 * ph, 0.0)
    breite = 1.0 / n
    int_z = ph[:-1] - ph[1:]                                                  # ∫ z du über das Intervall
    int_z2 = breite - (zph[1:] - zph[:-1])                                    # ∫ z² du
    dm = m - y_sortiert
    return np.sum(dm * dm * breite + 2.0 * dm * sg * int_z[None, :] + sg * sg * int_z2[None, :], axis=1)


def crps_und_exy(y: np.ndarray, m: float, sg: float):
    """Mittlerer CRPS und mittleres E|X − y| der Prognose N(m, σ²) je Zeile der Beobachtungen y."""
    z = (y - m) / sg
    exy = sg * (z * (2.0 * A.normal_cdf(z) - 1.0) + 2.0 * _phi(z))
    return (exy - sg / math.sqrt(math.pi)).mean(axis=1), exy.mean(axis=1)


def mc5(s: Settings) -> dict:
    """Je Zufalls-Seed `nobs5` Beobachtungen aus N(0, 1); jede Prognose wird gegen diese Beobachtungen bewertet (W2², W1, mittlerer CRPS, mittleres E|X − y|).

    Rückgabe: Anteil der Zufalls-Seeds, in denen F0 (die wahre Prognose) je Regel auf Platz 1 liegt, und die Zahl der Zufalls-Seeds, in denen F3 vor F0 liegt."""
    y = np.sort(PR.zufall_normal(s.seed_input, C.ZWECK_PROGNOSE, s.seeds5, s.nobs5), axis=1)
    werte = {"W2²": [], "W1": [], "E[CRPS]": [], "E|X − y|": []}
    for _, m, sg in prognosen(s):
        crps, exy = crps_und_exy(y, m, sg)
        werte["W2²"].append(w2_gegen_stichprobe(y, m, sg))
        werte["W1"].append(w1_gegen_stichprobe(y, m, sg))
        werte["E[CRPS]"].append(crps)
        werte["E|X − y|"].append(exy)
    platz1 = {}
    f3_vor_f0 = {}
    for regel, v in werte.items():
        v = np.array(v)                                                       # (5, Seeds)
        platz1[regel] = float(np.mean(v.argmin(axis=0) == 0))
        f3_vor_f0[regel] = int(np.sum(v[3] < v[0]))
    return {"platz1": platz1, "f3_vor_f0": f3_vor_f0, "seeds": s.seeds5, "nobs": s.nobs5}


def verdikt5(s: Settings, tab: list[dict], mc: dict) -> tuple[str, str]:
    r1, r2 = tab[1], tab[2]
    namen = {1: "die verschobene Prognose F1", 2: "die zu breite Prognose F2"}

    def vor(regel, key):
        (n1, z1), (n2, z2) = sorted(((1, r1), (2, r2)), key=lambda t: t[1][key])
        return f"{regel} ordnet {namen[n1]} vor {namen[n2]} ({zahl(z1[key], 3)} gegen {zahl(z2[key], 3)})"

    exy = rangfolge(tab, "exy")
    f3_pos = exy.index("F3") < exy.index("F0")
    teile = [vor("W1", "w1") + ", " + vor("W2²", "w2")]
    teile.append("E|X − y| stellt die zu schmale Prognose F3 vor die wahre F0 " + (f"({zahl(tab[3]['exy'], 3)} gegen {zahl(tab[0]['exy'], 3)}); das gilt bei gleicher Lage für jede Stichprobe von Beobachtungen, "
                 "denn E|X − y| wächst mit σ; in den Zufalls-Seeds: " + f"{mc['f3_vor_f0']['E|X − y|']} von {mc['seeds']}" if f3_pos else "(in dieser Einstellung nicht)"))
    teile.append("die wahre Prognose F0 liegt bei " + str(mc["nobs"]) + " Beobachtungen auf Platz 1 in " + ", ".join(f"{prozent(mc['platz1'][r], 1)} ({r})" for r in ("W2²", "W1", "E[CRPS]", "E|X − y|")) + f" von {mc['seeds']} Zufalls-Seeds")
    return "info", ("; ".join(teile) + f". Die Prozentwerte sind aus {mc['seeds']} Zufalls-Seeds gezählt und haben eine Standardabweichung von höchstens {zahl(sd_prozentpunkte(0.5, mc['seeds']), 1)} Prozentpunkten; "
                     "Unterschiede von wenigen Prozentpunkten zwischen den Bewertungsregeln sind keine Rangordnung der Regeln. Nur Gauß-Prognosen, bekannte Wahrheit N(0, 1), unabhängige Beobachtungen.")
