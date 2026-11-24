"""Abstand zwischen Verteilungen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Abstand misst, wie weit zwei Verteilungen auseinanderliegen: Diese Annahme wird an Tagesprofilen (Nachfrage über 24 Stunden) geprüft. Fünf Experimente: Verschiebung (W1, TV, KS, JS, KL),
Raster (Treppe), Histogramm ohne Unterschied, Distanz oder Divergenz („Sinkhorn-Distanz“ in fünf Verwendungen) und Prognosebewertung.
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import vab_constants as C
import vab_evaluation as E
import vab_vorgerechnet as VG
from vab_layout import dunkles_design, kopf_css
from vab_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from vab_visualization import (
    build_anteil3,
    build_kurven1,
    build_platz1,
    build_profile,
    build_prognose5,
    build_selbst4,
    build_streu4,
    build_treppe,
    build_w1_hist,
)

st.set_page_config(page_title="Abstand zwischen Verteilungen – Sebastian Hanisch", layout="wide")
KONFIG = {"displayModeBar": False}


@st.cache_data(show_spinner=False)
def _reihe1(profil1, zellen1, achse1, breite1, glaettung1):
    return E.reihe1(E.Settings(profil1=profil1, zellen1=zellen1, achse1=achse1, breite1=breite1, glaettung1=glaettung1))


@st.cache_data(show_spinner=False)
def _reihe2(profil2, zellen2, abstand2, breite2):
    return E.reihe2(E.Settings(profil2=profil2, zellen2=zellen2, abstand2=abstand2, breite2=breite2))


@st.cache_data(show_spinner=False)
def _w2_zeilen():
    return [E.w2_raster_zeile(b, sg) for b, sg in C.W2_ZEILEN]


@st.cache_data(show_spinner=False)
def _reihe3(seed, n, d, seeds, b3):
    stufen = sorted(set(C.B3_REIHE) | {b3})
    return [E.reihe3(seed, n, b, d, seeds) for b in stufen]


@st.cache_data(show_spinner="Die Abstände werden gerechnet: 300 Sinkhorn-Läufe im Stapel, dazu die Nebenbedingungsform V-a (Bisektion) ...")
def _analyse4(menge, c):
    return E.analyse4(menge, c)


@st.cache_data(show_spinner=False)
def _tabelle5(f1_lage, f2_sigma, f3_sigma, f4_lage, f4_sigma):
    return E.tabelle5(E.Settings(f1_lage=f1_lage, f2_sigma=f2_sigma, f3_sigma=f3_sigma, f4_lage=f4_lage, f4_sigma=f4_sigma))


@st.cache_data(show_spinner="Die Prognosen werden an den Beobachtungen jedes Zufalls-Seeds bewertet ...")
def _mc5(seed, nobs, seeds, f1_lage, f2_sigma, f3_sigma, f4_lage, f4_sigma):
    return E.mc5(E.Settings(seed_input=seed, nobs5=nobs, seeds5=seeds, f1_lage=f1_lage, f2_sigma=f2_sigma, f3_sigma=f3_sigma, f4_lage=f4_lage, f4_sigma=f4_sigma))


def _tabelle_md(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in vab_layout.py).
st.html(kopf_css(dunkles_design()))
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')
with st.container(key="kopf"):
    st.title("📏 Abstand zwischen Verteilungen")
st.markdown(
    """
Ein **Abstand** (englisch distance) zwischen zwei Verteilungen soll sagen, wie weit sie auseinanderliegen. Diese Annahme prüft die Demo an **Tagesprofilen**: der Verteilung der Nachfrage über 24 Stunden,
auf einem Raster von Zellen. Verglichen werden der **Wasserstein-Abstand** (englisch Wasserstein distance, W1 und W2²), der **Totalvariationsabstand** (TV), der **Kolmogorov-Smirnov-Abstand** (KS),
die **Kullback-Leibler-Divergenz** (KL) und die **Jensen-Shannon-Divergenz** (JS), dazu das Wort **„Sinkhorn-Distanz“**, das fünf verschiedene Größen meint (V-a bis V-e). Die Annahme gilt für W1 bei einer
Verschiebung (der Wert wächst mit der Verschiebung: beim Kasten und bei zwei Spitzen auf der Geraden, auf dem Kreis bei zwei Spitzen nicht immer), nicht für TV, KS und JS (beim Kasten sättigen sie, bei zwei Spitzen
steigen sie und fallen wieder) und nicht für KL (beim Kasten wird sie unendlich, bei zwei Spitzen hängt sie an der Glättung), und sie gilt auf dem Raster nur mit der Rasterweite im Blick.

Weiter unten: **❓ Kernfragen**, die fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
Gezählt wird in Paaren, Verteilungen und Zufalls-Seeds; der Aufwand der Sinkhorn-Läufe erscheint als Zahl der **Matrix-Vektor-Produkte**, nicht in Sekunden.
"""
)

with st.expander("So rechnen die Abstände", expanded=True):
    st.markdown(
        """
1. **Ein Tagesprofil** ist eine Verteilung $p$ auf $B$ Zellen der Breite $h = 24/B$ Stunden (Summe 1); die Zellenmitte der Zelle $k$ liegt bei $(k + \\tfrac12)\\,h$. Der **Wasserstein-Abstand W1** ist der Aufwand,
   die eine Verteilung in die andere zu verschieben (Kosten = Stunden); auf einer Geraden ist er $h \\sum_k |F_k - G_k|$ aus den **Verteilungsfunktionen** (englisch cumulative distribution function) $F$, $G$.
   **W2²** verwendet die Kosten (Stunden)² und wird über die **Quantilfunktionen** gerechnet.
2. **TV**, **KS**, **JS** und **KL** vergleichen die Zellenwerte statt der Lage: TV $= \\tfrac12\\sum_k |p_k - q_k|$, KS $= \\max_k |F_k - G_k|$, KL ist unendlich, sobald eine Zelle in $p$ gefüllt und in $q$ leer ist.
   Die **Glättung** (englisch smoothing) $g$ addiert $g$ in Zähler und Nenner jeder Zelle; die Zahl hängt dann an $g$.
3. **„Sinkhorn-Distanz“** nennt in der Literatur fünf Größen: **V-a** die Kosten unter der Nebenbedingung $\\mathrm{KL}(P \\,\\|\\, a b^{\\top}) \\le \\alpha$ (Cuturi 2013, Definition 1), **V-b** die Kosten $\\langle C, P_\\varepsilon\\rangle$
   unter dem weichen Plan, **V-c** die zentrierte Größe $2W(a,b) - W(a,a) - W(b,b)$ mit $W$ = V-b (Genevay u. a. 2018), **V-d** die zentrierte Größe $S_\\varepsilon$ mit Faktor ½ und Entropieterm (Feydy u. a. 2019),
   **V-e** den Wert mit Entropieterm ohne Zentrieren. Jede Zahl der Demo nennt die Verwendung. Die Skala heißt $\\varepsilon/D$ mit $D$ = Median von W2² über die Paare der gezeigten Menge.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben

    st.header("🔬 Experiment 1: Verschieben")
    profil1 = st.radio("Profil", C.PROFIL1, key="profil1", help="Kasten: Gleichverteilung von 6 bis 9 Uhr. Zwei Spitzen: Normalverteilungen bei 6:30 und 14:30 Uhr (Gewichte 0,4 und 0,6).")
    s1 = st.slider("Verschiebung s (Stunden)", *bounds("s1"), step=C.S1_SCHRITT, key="s1", help="Um wie viele Stunden das zweite Profil gegen das erste verschoben ist (markiert in den Kurven; die Kurven zeigen alle Verschiebungen).")
    breite1 = st.slider("Breite σ der Spitzen (Stunden)", *bounds("breite1"), step=0.1, key="breite1", help="Nur für das Profil „Zwei Spitzen“: Standardabweichung jeder Spitze.", disabled=profil1 != C.PROFIL1[1])
    zellen1 = st.select_slider("Zellen", C.ZELLEN1, key="zellen1", help="Zahl der Zellen des Tages (24: eine Zelle je Stunde).")
    achse1 = st.radio("Zeitachse", C.ACHSE1, key="achse1", horizontal=True, help="Gerade: 0 Uhr und 24 Uhr sind die Enden. Kreis: Mitternacht schließt den Tag; nur W1 und die Profile rechnen anders (TV, KS, JS, KL sind Zellenwerte).")
    glaettung1 = st.select_slider("KL-Glättung g", C.GLAETTUNG1, key="glaettung1", format_func=lambda g: "aus" if g == 0 else f"{g:.0e}".replace("e-0", "e-"),
                                  help="Wird in Zähler und Nenner jeder Zelle addiert (ohne Renormierung). Bei „aus“ ist KL unendlich, sobald eine Zelle leer wird.")

    st.header("🔬 Experiment 2: Raster")
    profil2 = st.radio("Profile", C.PROFIL2, key="profil2", help="Zwei Spitzen: je die ganze Masse in einem Punkt. Glatte Profile: Normalverteilungen gleicher Breite, Zellenintegrale als Massen.")
    zellen2 = st.select_slider("Zellen B", C.ZELLEN2, key="zellen2", help="Zahl der Zellen von 0 bis 1 (des Tages); die Tabelle zeigt immer die ganze Reihe 10 bis 5 000 und diese Zahl dazu.")
    abstand2 = st.select_slider("Abstand der Spitzen (Anteil des Tages)", C.ABSTAND2, key="abstand2", format_func=lambda x: f"{x}".replace(".", ","), help="Die erste Spitze liegt bei 0,3 des Tages.")
    breite2 = st.select_slider("Breite σ (Anteil des Tages)", C.BREITE2, key="breite2", format_func=lambda x: f"{x}".replace(".", ","), help="Nur für die glatten Profile.", disabled=profil2 != C.PROFIL2[1])

    st.header("🔬 Experiment 3: Histogramm")
    n3 = st.select_slider("Beobachtungen n je Stichprobe", C.N3, key="n3", help="Zahl der Beobachtungen jeder der beiden Stichproben.")
    b3 = st.select_slider("Zellen B des Histogramms", C.B3, key="b3", help="Die Zellen des Histogramms von 0 bis 1; die Reihe zeigt zusätzlich B = 10, 50, 200 und 1 000.")
    d3 = st.select_slider("Verschiebung d (Anteil des Tages)", C.D3, key="d3", format_func=lambda x: f"{x}".replace(".", ","), help="0: beide Stichproben stammen aus derselben Verteilung N(0,3; 0,04²).")
    richtung3 = st.radio("Richtung der KL", C.RICHTUNG3, key="richtung3", horizontal=True, help="x ‖ y: KL(x ‖ y) ist unendlich, wenn x eine Zelle füllt, die in y leer ist. Beide Richtungen stehen im Diagramm.")
    seeds3 = st.select_slider("Zufalls-Seeds (Wiederholungen)", C.SEEDS3, key="seeds3", help="Zahl der unabhängigen Wiederholungen mit je zwei Stichproben.")

    st.header("🔬 Experiment 4: Distanz oder Divergenz")
    c4 = st.select_slider("ε/D", C.C4, key="c4", format_func=lambda x: f"{x}".replace(".", ","), help="Das Gewicht der Entropie, gemessen am Median D von W2² über die Paare. Klein heißt nah am exakten W2². Unter 0,3 wird nicht angeboten: einzelne Paare mit fast getrennten Trägern brauchen dann sehr viele Durchläufe, und die Rechnung dauert deutlich länger (gemessen bei 0,1 bis etwa eine Minute).")
    verw4 = st.radio("Verwendung (für die Diagramme)", C.VERW4, key="verw4", horizontal=True, help="Welche Verwendung die zwei Diagramme zeigen; die Tabelle zeigt immer alle fünf.")
    menge4 = st.slider("Verteilungsmenge (Zufalls-Seed der Menge)", *bounds("menge4"), key="menge4", help="Eine Menge sind 24 Tagesprofile aus Mischungen von 1 bis 3 Gauß-Hügeln; Menge 1 ist die festgeschriebene Standardmenge.")

    st.header("🔬 Experiment 5: Prognosen")
    f1_lage = st.select_slider("F1: Lage", C.F1_LAGE, key="f1_lage", format_func=lambda x: f"{x}".replace(".", ","), help="F1 ist N(Lage, 1): richtig breit, falsch verschoben. Die Wahrheit ist N(0, 1).")
    f2_sigma = st.select_slider("F2: σ (zu breit)", C.F2_SIGMA, key="f2_sigma", format_func=lambda x: f"{x}".replace(".", ","), help="F2 ist N(0, σ²) mit σ > 1.")
    f3_sigma = st.select_slider("F3: σ (zu schmal)", C.F3_SIGMA, key="f3_sigma", format_func=lambda x: f"{x}".replace(".", ","), help="F3 ist N(0, σ²) mit σ < 1.")
    f4_lage = st.select_slider("F4: Lage", C.F4_LAGE, key="f4_lage", format_func=lambda x: f"{x}".replace(".", ","), help="F4 ist N(Lage, σ²): zugleich verschoben und zu breit.")
    f4_sigma = st.select_slider("F4: σ", C.F4_SIGMA, key="f4_sigma", format_func=lambda x: f"{x}".replace(".", ","), help="Streuung von F4.")
    nobs5 = st.select_slider("Beobachtungen n_obs je Zufalls-Seed", C.NOBS5, key="nobs5", help="Wie viele Beobachtungen aus N(0, 1) jede Prognose je Zufalls-Seed bewerten.")
    seeds5 = st.select_slider("Zufalls-Seeds (Wiederholungen)", C.SEEDS5, key="seeds5", help="Zahl der unabhängigen Wiederholungen.")

    st.header("🎲 Zufall")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für die Stichproben in Experiment 3 und die Beobachtungen in Experiment 5 (Experiment 4 hat seine eigene Verteilungsmenge).")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

werte = {k: st.session_state[k] for k in (
    "seed_input", "profil1", "s1", "breite1", "zellen1", "achse1", "glaettung1", "profil2", "zellen2", "abstand2", "breite2", "n3", "b3", "d3", "richtung3", "seeds3", "c4", "verw4", "menge4",
    "f1_lage", "f2_sigma", "f3_sigma", "f4_lage", "f4_sigma", "nobs5", "seeds5")}
sync_query_params(werte)
s = E.settings_aus_zustand(werte)

# --- Rechnung ------------------------------------------------------------------------------------------------------------------------------

reihe1 = _reihe1(s.profil1, s.zellen1, s.achse1, s.breite1, s.glaettung1)
zeilen2 = _reihe2(s.profil2, s.zellen2, s.abstand2, s.breite2)
w2_zeilen = _w2_zeilen()
zeilen3 = _reihe3(s.seed_input, s.n3, s.d3, s.seeds3, s.b3)
try:
    res4 = _analyse4(s.menge4, s.c4)
    fehler4 = None
except ValueError as fehler:          # ε zu klein für float64 im Kern-Bereich
    res4, fehler4 = None, str(fehler)
tab5 = _tabelle5(s.f1_lage, s.f2_sigma, s.f3_sigma, s.f4_lage, s.f4_sigma)
mc5 = _mc5(s.seed_input, s.nobs5, s.seeds5, s.f1_lage, s.f2_sigma, s.f3_sigma, s.f4_lage, s.f4_sigma)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
kasten_r = _reihe1(C.PROFIL1[0], 24, "Gerade", 1.0, 0.0)
spitzen_r = _reihe1(C.PROFIL1[1], 24, "Gerade", 1.0, 1e-12)
k_s6 = E.zeile1(spitzen_r, 6.0)
i_tv = max(range(len(spitzen_r["s"])), key=lambda i: spitzen_r["tv"][i])          # Höchstwert von TV bei zwei Spitzen
st.markdown("**Wächst der Abstand mit der Verschiebung?**")
st.info(
    f"Beim Kasten (24 Zellen, Gerade) ist W1 bei jeder der {len(kasten_r['s'])} Verschiebungen gleich s; TV, KS und JS sättigen ab s = 3 Stunden (TV = KS = {E.zahl(E.zeile1(kasten_r, 3.0)['tv'], 0)}, "
    f"JS = ln 2 = {E.zahl(C.LN2, 3)}), KL ist ohne Glättung schon bei s = 0,5 Stunden unendlich. Beim Profil mit zwei Spitzen (σ = 1 Stunde, 24 Zellen, Gerade) steigt W1 weiter, TV aber ist nicht monoton: Höchstwert "
    f"{E.zahl(spitzen_r['tv'][i_tv], 3)} bei s = {E.zahl(spitzen_r['s'][i_tv], 1)}, danach {E.zahl(k_s6['tv'], 3)} bei s = 6 Stunden."
)
st.markdown("**Hängt die Zahl am Raster?**")
r_dirac = [E.treppe_zeile(b, 0.0137) for b in C.ZELLEN2_REIHE]
st.info(
    f"Zwei Spitzen im Abstand 0,0137 des Tages: Das Raster meldet W1 = 0 bei {', '.join(str(z['zellen']) for z in r_dirac if z['w1'] == 0)} Zellen und {E.zahl(r_dirac[3]['w1'], 3)} bei 100 Zellen; "
    f"der größte Fehler unter den gerechneten Zellenzahlen ist {E.zahl(max(z['fehler_h'] for z in r_dirac), 2)} Zellenbreiten. Glatte Profile gleicher Breite (σ = 0,04 des Tages) sind dagegen ab 50 Zellen auf Rundung genau (Experiment 2; bei kleinerem σ erst bei mehr Zellen)."
)
st.markdown("**Ist „Sinkhorn-Distanz“ (V-a bis V-e) ein Abstand?**")
if res4 is not None:
    k4 = res4["kennzahlen"]
    st.info(
        f"Bei ε/D = {E.zahl(res4['c'], 1)} (Menge {res4['menge']}, 24 Verteilungen) hat V-b für {k4['V-b']['nicht_naechster']} der 24 Profile ein anderes Profil näher als sich selbst; "
        f"V-c und V-d haben den Selbstabstand 0 durch Konstruktion und verletzen die Dreiecksungleichung der Wurzel in {E.ganz(k4['V-c']['verletzt'])} (V-c) und {E.ganz(k4['V-d']['verletzt'])} (V-d) von "
        f"{E.ganz(k4['V-c']['tripel'])} Tripeln. Welche Verwendung gemeint ist, ändert die Zahl (Experiment 4)."
    )
else:
    st.warning(fehler4)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Verschieben statt Überlappen")
st.caption(
    "Das Ausgangsprofil A und das um s Stunden verschobene Profil B; gezeigt wird, wie jeder Abstand mit s wächst. KL ist KL(B ‖ A) mit der gewählten Glättung. Die Kurven zeigen alle Verschiebungen von 0 bis 8 Stunden "
    "in Schritten von 0,5; die senkrechte Linie markiert die gewählte."
)
a1, b1 = E.profil1(s, 0.0), E.profil1(s, s.s1)
zeile = E.zeile1(reihe1, s.s1)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("W1 (Stunden)", E.zahl(zeile["w1"], 3))
m2.metric("TV", E.zahl(zeile["tv"], 3))
m3.metric("KS", E.zahl(zeile["ks"], 3))
m4.metric("JS (nat)", E.zahl(zeile["js"], 3))
m5.metric("KL (nat)", E.zahl(zeile["kl"], 3) if zeile["kl"] != float("inf") else "∞")
st.plotly_chart(build_profile(a1, b1, s.zellen1, s.s1), width="stretch", key="profile1", config=KONFIG)
st.plotly_chart(build_kurven1(reihe1, s.s1, s.glaettung1), width="stretch", key="kurven1", config=KONFIG)
tab1 = []
for sv in C.S_ZEILEN:
    z = E.zeile1(reihe1, sv)
    tab1.append([E.zahl(sv, 0), E.zahl(z["w1"], 3), E.zahl(z["tv"], 3), E.zahl(z["ks"], 3), E.zahl(z["js"], 3), E.zahl(z["kl"], 3), E.zahl(z["kl_roh"], 3)])
st.markdown(_tabelle_md(["s (Stunden)", "W1 (Stunden)", "TV", "KS", "JS", f"KL ({E.glaettung_text(s.glaettung1)})", "KL ohne Glättung"], tab1))
gl = E.kl_gegen_glaettung(s, s.s1)
st.markdown(f"**KL bei s = {E.zahl(s.s1, 1)} Stunden je Glättung:** " + "; ".join(f"{E.glaettung_text(g)}: {E.zahl(v, 3)}" for g, v in gl.items()) + ". Die Zahl hängt an der Glättung, sobald eine Zelle fast leer ist.")
kind, text = E.verdikt1(s, reihe1)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Das Raster macht die Zahl")
st.caption("W1 zwischen zwei Verteilungen auf [0, 1] (dem Tag) im Raster gegen den wahren Wert. Zwei Spitzen: ganze Masse in einem Punkt, die Zelle ist ⌊x · B⌋. Glatte Profile: Normalverteilungen gleicher Breite.")
st.plotly_chart(build_treppe(zeilen2, s.profil2 == C.PROFIL2[1]), width="stretch", key="treppe2", config=KONFIG)
tab2 = [[str(z["zellen"]), E.zahl(z["w1"], 4), E.zahl(z["wahr"], 4), E.exp_text(abs(z["fehler"]), 1), E.zahl(z["fehler_h"], 3) if z["fehler_h"] > 1e-3 else E.exp_text(z["fehler_h"], 1)] for z in zeilen2]
st.markdown(_tabelle_md(["Zellen B", "W1 im Raster", "W1 wahr", "Betrag des Fehlers", "Betrag des Fehlers / h"], tab2))
kind, text = E.verdikt2(s, zeilen2)
getattr(st, kind)(text)
st.markdown("**W2² zwischen zwei Normalverteilungen auf [−6, 8].** Als Massen der Zellen dienen die Zellenintegrale; gerechnet wird W2² im Raster über die Quantile.")
tab2w = [[str(z["zellen"]), E.zahl(z["sigma2"], 1), "ja" if z["deckungsgleich"] else "nein", E.zahl(z["h"], 3), E.zahl(z["wahr"], 4), E.exp_text(z["fehler"], 2), E.zahl(z["fehler_h2"], 4) if abs(z["fehler_h2"]) > 1e-3 else E.exp_text(z["fehler_h2"], 1)] for z in w2_zeilen]
st.markdown(_tabelle_md(["Zellen", "σ der zweiten Verteilung", "Zellen deckungsgleich", "h", "W2² geschlossen", "Fehler Raster − geschlossen", "Fehler / h²"], tab2w))
kind, text = E.verdikt2_w2(w2_zeilen)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Histogramm ohne Unterschied")
st.caption(
    "Zwei Stichproben mit je n Beobachtungen aus N(0,3; 0,04²) und N(0,3 + d; 0,04²) auf [0, 1], als Histogramm mit B Zellen. Bei d = 0 stammen beide aus **derselben** Verteilung. Zufalls-Seed der Demo und Zufalls-Seeds (Wiederholungen) wie in der Seitenleiste. "
    "Dies zeigt Stichprobenrauschen nur an einem Beispiel (eine Dimension, ein n)."
)
zg = next(z for z in zeilen3 if z["zellen"] == s.b3)
anteil = zg["anteil_xy"] if s.richtung3 == C.RICHTUNG3[0] else zg["anteil_yx"]
m1, m2, m3 = st.columns(3)
m1.metric(f"Anteil der Zufalls-Seeds mit KL({s.richtung3}) = ∞", E.prozent(anteil, 1), help="Ohne Glättung: eine Zelle ist in der ersten Stichprobe gefüllt und in der zweiten leer.")
m2.metric("W1 aus dem Histogramm (Median)", E.zahl(zg["w1_hist_median"], 4))
m3.metric("W1 über sortierte Stichproben (Median)", E.zahl(zg["w1_sort_median"], 4))
c31, c32 = st.columns(2)
c31.plotly_chart(build_anteil3(zeilen3), width="stretch", key="anteil3", config=KONFIG)
c32.plotly_chart(build_w1_hist(zg), width="stretch", key="w1hist3", config=KONFIG)
st.markdown(f"W1 aus dem Histogramm, 5 bis 95 % über die Zufalls-Seeds: {E.zahl(zg['w1_hist_q'][0], 4)} bis {E.zahl(zg['w1_hist_q'][1], 4)} (bei B = {zg['zellen']}); über die sortierten Stichproben: "
            f"{E.zahl(zg['w1_sort_q'][0], 4)} bis {E.zahl(zg['w1_sort_q'][1], 4)}.")
klein = [E.kl_unendlich_exakt(b, 3) for b in (4, 8, 16)]
st.markdown(f"**Klein gerechnet (exakt):** zwei Stichproben mit je 3 Beobachtungen aus derselben Gleichverteilung über B Zellen: P(KL = ∞) ist {klein[0].numerator} / {E.ganz(klein[0].denominator)} = {E.zahl(float(klein[0]), 4)} bei B = 4, "
            f"{E.zahl(float(klein[1]), 4)} bei B = 8 und {E.zahl(float(klein[2]), 4)} bei B = 16 (Aufzählung aller Paare von Stichproben und Inklusion-Exklusion geben dasselbe).")
kind, text = E.verdikt3(s, zeilen3)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Distanz oder Divergenz?")
st.caption(
    "24 Tagesprofile (Mischungen von 1 bis 3 Gauß-Hügeln) auf 48 Zellen à 30 Minuten, 276 Paare, Kosten (Δt)² mit größtem Wert 1. ε = ε/D · D mit D = Median von W2² über die Paare. "
    "„Selbst nicht am nächsten“ zählt die Verteilungen, für die eine andere Verteilung einen kleineren Wert hat als die Verteilung selbst."
)
if res4 is None:
    st.warning(fehler4)
else:
    k4 = res4["kennzahlen"]
    zeilen4 = []
    namen4 = {"V-a": "V-a Nebenbedingung KL ≤ α", "V-b": "V-b ⟨C, P_ε⟩", "V-c": "V-c Genevay (Faktor 2)", "V-d": "V-d Feydy (Faktor ½)", "V-e": "V-e OT_ε ohne Zentrieren"}
    for v in C.VERW4:
        z = k4[v]
        konstruktion = v in ("V-c", "V-d")
        zeilen4.append([namen4[v], E.zahl(z["selbst_mittel"], 4) + (" (Konstruktion)" if konstruktion else ""), f"{z['nicht_naechster']} von 24" + (" (Konstruktion)" if konstruktion else ""), E.zahl(z["kendall"], 3),
                        f"{z['nn_wie_exakt']} von 24", E.zahl(z["verhaeltnis"], 2), f"{E.ganz(z['verletzt'])} von {E.ganz(z['tripel'])}"])
    st.markdown(_tabelle_md(["Verwendung", "Selbstabstand (Mittel)", "selbst nicht am nächsten", "Kendall τ gegen W2²", "nächster Nachbar wie exakt", "Median Wert / W2²", "Dreiecksungleichung verletzt"], zeilen4))
    st.caption(
        f"D = {E.zahl(res4['d'], 4)}, ε = {E.zahl(res4['eps'], 4)}; V-a: α = {E.zahl(res4['alpha'], 4)} (Median der KL-Werte von V-b), {E.unwirksam_satz(res4)}. "
        f"Aufwand der Läufe für V-b bis V-e (Nebenanzeige, nie als Iterationen): im Median {E.ganz(res4['produkte_median'])} Matrix-Vektor-Produkte je Paar, am größten {E.ganz(res4['produkte_max'])}; "
        f"Läufe ohne Konvergenz innerhalb von {E.ganz(C.MAX_ITER4)} Durchläufen: {res4['nicht_konvergiert']} bei V-b bis V-e, {res4['nicht_konvergiert_a']} in den Bisektionsläufen von V-a. Prüfsumme der Verteilungsmenge: {res4['pruefsumme']}."
    )
    c41, c42 = st.columns(2)
    c41.plotly_chart(build_selbst4(res4, s.verw4), width="stretch", key="selbst4", config=KONFIG)
    c42.plotly_chart(build_streu4(res4, s.verw4), width="stretch", key="streu4", config=KONFIG)
    st.caption(f"Links: Wert von {s.verw4} je Verteilung gegen sich selbst (Balken, rot, wenn eine andere Verteilung näher ist) und der kleinste Wert zu einer anderen (Strich). Rechts: Wert gegen das exakte W2² je Paar.")
    streu = VG.streuung(s.c4)
    if streu is not None:
        kopf = ["Verwendung"] + [f"Menge {m}" for m in streu["mengen"]]
        zl = []
        for v in C.VERW4:
            zl.append([v] + [f"τ {E.zahl(streu['daten'][str(m)][v]['kendall'], 3)}, {streu['daten'][str(m)][v]['nicht_naechster']} von 24" for m in streu["mengen"]])
        st.markdown(f"**Dieselbe Zelle in fünf Verteilungsmengen (ε/D = {E.zahl(s.c4, 1)}):** Kendall τ und Zahl der Verteilungen, für die eine andere näher liegt als sie selbst.")
        st.markdown(_tabelle_md(kopf, zl))
    kind, text = E.verdikt4(res4)
    getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Prognosen bewerten")
st.caption("Die Wahrheit ist N(0, 1); fünf Gauß-Prognosen F0 bis F4 (F0 ist die wahre). Kleiner ist besser. Die Tabelle ist geschlossen gerechnet; die Zufalls-Seeds bewerten jede Prognose gegen die empirische Verteilung von n_obs Beobachtungen.")
tab5md = []
for z in tab5:
    tab5md.append([z["name"], E.zahl(z["lage2"], 4), E.zahl(z["streu2"], 4), E.zahl(z["w2"], 4), E.zahl(z["w1"], 4), E.zahl(z["ecrps"], 4), E.zahl(z["exy"], 4)])
st.markdown(_tabelle_md(["Prognose", "Lage²", "Streuung²", "W2²", "W1", "E[CRPS]", r"E\|X − y\|"], tab5md))
st.markdown("**Rangfolgen (klein bis groß):** " + "; ".join(f"{regel}: " + " < ".join(E.rangfolge(tab5, key)) for regel, key in (("W2²", "w2"), ("W1", "w1"), ("E[CRPS]", "ecrps"), ("E|X − y|", "exy"))))
c51, c52 = st.columns([3, 2])
c51.plotly_chart(build_prognose5(tab5), width="stretch", key="prognose5", config=KONFIG)
c52.plotly_chart(build_platz1(mc5), width="stretch", key="platz1_5", config=KONFIG)
kind, text = E.verdikt5(s, tab5, mc5)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme der Vorgänger | Was hier passiert, wenn sie nicht gilt | Wo es weitergeht |
|---|---|---|
| *Lager, Kunden und Kosten sind gegeben; der Optimalwert ist das Ende der Rechnung, ein Preis für einen Auftrag.* | Der Wert ist hier eine Abstandsgröße zwischen Verteilungen; Raster und Kostenwahl ändern ihn (Experiment 2). | – |
| *⟨C, P_ε⟩ ist eine Näherung des Transportwerts, die mit ε → 0 besser wird.* | Auch dann ist es kein Abstand (Selbstabstand ≠ 0, Experiment 4); die Definition der „Sinkhorn-Distanz“ muss gewählt werden (V-a bis V-e). | – |
| *Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle.* | Gilt in den Experimenten 1, 2, 4 und 5; Experiment 3 zeigt das Rauschen einer Stichprobe an einem Beispiel. | Erkennen einer Änderung mit Referenz, Fenster und Schwelle (spätere Demo der Reihe). |
| *Kontrast, keine behobene Schwäche:* Diese Demo misst den Abstand zwischen zwei Verteilungen. | Wie man von der einen zur anderen kommt (die Mischung als Gerade der Totalvariation gegen die Verschiebung als Gerade der W2-Geometrie), zeigt sie nicht. | Zwischenverteilungen (spätere Demo der Reihe). |
| *Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1.* | Experiment 3 zeigt das nur an einem Beispiel (n = 500 oder gewähltes n, eine Dimension); wie der Rauschboden mit n und Dimension wächst, zeigt diese Demo nicht. | Datenbedarf und Dimension (spätere Demo der Reihe). |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Tagesprofile liegen auf einem Raster von 24, 48 oder 96 Zellen; die Zellenmitten sind $(k + \\tfrac12)\\,h$. Die Zellenlage ist Teil der Konvention: Punktproben einer Dichte an den Zellenmitten geben andere TV-Werte als an den Zellgrenzen.
- Kosten in Experiment 4: $(\\Delta t)^2$ durch den größten Wert geteilt (max C = 1). W2² in Stunden² ist der Wert mal das Quadrat des größten Abstands der Zellenmitten.
- Eine **Familie** von Verteilungen: Mischungen von 1 bis 3 Gauß-Hügeln mit Breite 0,72 bis 2,4 Stunden. Kästen, Spitzen und schwere Ränder in Experiment 4 sind nicht gemessen.
- Zufall aus dem Hausgenerator SplitMix64; die Stichproben in Experiment 3 und 5 benutzen Box-Muller mit numpy-Funktionen (in der letzten Binärstelle plattformabhängig; ein Einfluss auf die Zählungen ist nur in Grenzfällen zu erwarten und auf einer zweiten Plattform nicht geprüft).

**Grenzen.**
- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Paare, Verteilungen, Tripel, Zufalls-Seeds). Eine Zahl über „die Familie“ ist ein Wert einer Verteilungsmenge; die Tabelle zu fünf Mengen zeigt die Streuung.
- Die Dreiecksungleichung wird für die Wurzel des Werts gezählt (die Kosten sind quadriert). „Keine Verletzung gezählt“ ist eine Zählung über die gezeigten Tripel, kein Beweis.
- Die Zeitachse „Kreis“ ändert nur W1 (und die Profile); TV, KS, JS und KL rechnen über die Zellenwerte der Geraden, KS ist auf dem Kreis nicht rotationsinvariant.
- Der Rasterfehler von W2² ist nahe $h^2/6$ bei ungleicher Streuung (gemessen an einem Paar von Normalverteilungen mit den Streuungen 1 und 0,7), auch bei deckungsgleichen Zellen; bei gleicher Streuung ist er null (deckungsgleiche Zellen) oder schwankt mit der Zellenzahl (Tabelle in Experiment 2), nicht allgemein.
- Experiment 5 bewertet nur Gauß-Prognosen gegen eine bekannte Wahrheit und unabhängige Beobachtungen; zeitlich abhängige Fehler, Mehrschrittprognosen und Ensembles sind nicht gemessen.
- Ein Nachweis, welcher Abstand für eine Entscheidung der richtige ist, steht hier nicht.

**Bekannte Probleme und Unsicherheiten.**
- Die Sinkhorn-Läufe haben bei kleinem ε/D wenige Paare mit fast getrennten Trägern, für die die Iteration sehr lange läuft; die Tabelle nennt, wie viele Läufe die Grenze von 200 000 Durchläufen erreichten.
- Unter ε/D = 0,3 bieten die Regler nichts an: einzelne Paare mit fast getrennten Trägern brauchen dort sehr viele Durchläufe; gemessen bei ε/D = 0,1 (Verteilungsmenge 1, ein Kern, bei gleichzeitiger anderer Last des Rechners) dauert die Rechnung bis etwa eine Minute, und mehrere Läufe erreichen die Grenze von 200 000 Durchläufen. Bei noch kleinerem ε (unter ε = 1/690 bei größtem Kostenwert 1) unterläuft float64 im Kern-Bereich (K = exp(−C/ε) wird 0), die Demo meldet das statt falsch zu rechnen.
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* zwei Verteilungen $p$, $q$ auf $B$ Zellen mit Zellenmitten $t_k$ und Breite $h$ (Summe 1). *Gesucht:* ein Abstand $d(p, q)$, der angibt, wie weit sie auseinanderliegen; hier werden W1, W2², TV, KS, KL, JS und
fünf Verwendungen von „Sinkhorn-Distanz“ verglichen.

| Symbol | Bedeutung |
|---|---|
| $p_k$, $q_k$ | Zellenwerte der beiden Tagesprofile, Summe 1 |
| $F_k$, $G_k$ | Verteilungsfunktionen $\sum_{i \le k} p_i$, $\sum_{i \le k} q_i$ |
| $h$, $B$ | Zellenbreite $24/B$ Stunden und Zahl der Zellen |
| $c(t, t')$ | Kosten (Bodenabstand), $\lvert t - t'\rvert^q$ mit $q = 1$ (W1) oder $q = 2$ (W2²) |
| $g$ | Glättung der KL je Zelle (Zähler und Nenner), ohne Renormierung |
| $\varepsilon$, $D$ | Gewicht der Entropie; $D$ = Median von W2² über die Paare, $\varepsilon = c \cdot D$ mit $c = \varepsilon/D$ |
| $a$, $b$, $P$ | zwei Verteilungen, weicher Plan $P_\varepsilon$ mit den Randverteilungen $a$, $b$ |
| $\alpha$ | Grenze der Entropie-Nebenbedingung von V-a |

**W1 und TV, KS, KL, JS.**
$$W_1(p, q) = h \sum_{k=1}^{B} |F_k - G_k|, \qquad \mathrm{TV} = \tfrac12 \sum_k |p_k - q_k|, \qquad \mathrm{KS} = \max_k |F_k - G_k|$$
$$\mathrm{KL}(p \,\|\, q) = \sum_{k:\, p_k > 0} p_k \ln \frac{p_k + g}{q_k + g}, \qquad \mathrm{JS}(p, q) = \tfrac12 \mathrm{KL}(p \,\|\, m) + \tfrac12 \mathrm{KL}(q \,\|\, m), \quad m = \tfrac{p + q}{2}$$
Auf dem Kreis (Mitternacht): $W_1 = h \sum_k |D_k - \operatorname{Median}(D)|$ mit $D = F - G$.

**W2² über die Quantilfunktionen.**
$$W_2^2(p, q) = \int_0^1 \big(F^{-1}(u) - G^{-1}(u)\big)^2\,\mathrm{d}u$$
**Handbeispiel Kasten** (24 Stundenzellen, Kasten 6 bis 9 Uhr, je Stunde $\tfrac13$, um $d$ Stunden verschoben, gerechnet für ganze und halbe Stunden): $W_1 = d$ Stunden, $\mathrm{TV} = \mathrm{KS} = \min(d, 3)/3$, $\mathrm{KL} = \infty$ für jedes $d > 0$ (jede Verschiebung füllt eine Zelle, die leer war), $\mathrm{JS} = \ln 2$ für $d \ge 3$ (getrennte Träger).
Bei $d = 1$ teilen beide Profile die Zellen 7 und 8, $\mathrm{TV} = \tfrac13$; die Verteilungsfunktionen unterscheiden sich an drei Zellgrenzen um je $\tfrac13$: $W_1 = 3 \cdot \tfrac13 \cdot 1 = 1$ Stunde.

**Handbeispiel Treppe** (zwei Spitzen bei $\tfrac{3}{10}$ und $\tfrac{3137}{10000}$ des Tages, Zelle $= \lfloor x B \rfloor$): bei $B = 10, 20, 50$ liegen beide in der Zelle 3, 6, 15 und $W_1 = 0$; bei $B = 100$ in den Zellen 30 und 31 mit
$W_1 = \tfrac{1}{100}$; bei $B = 500$ in den Zellen 150 und 156 mit $W_1 = \tfrac{6}{500}$, Fehler in Zellenbreiten $\tfrac{0{,}0137 - 0{,}012}{0{,}002} = \tfrac{17}{20}$.

**Fünf Verwendungen von „Sinkhorn-Distanz“.** Mit dem weichen Plan $P_\varepsilon$ (minimiert $\langle C, P\rangle + \varepsilon\,\mathrm{KL}(P \,\|\, a b^{\top})$ unter den Randverteilungen) und $\mathrm{OT}_\varepsilon(a, b) = \langle C, P_\varepsilon\rangle + \varepsilon\,\mathrm{KL}(P_\varepsilon \,\|\, a b^{\top})$:

$$\text{V-a: } \min_{P \in U(a, b),\ \mathrm{KL}(P \| a b^{\top}) \le \alpha} \langle C, P\rangle$$

$$\text{V-b: } \langle C, P_\varepsilon\rangle$$

$$\text{V-c: } 2\,W(a, b) - W(a, a) - W(b, b) \ \text{mit } W = \text{V-b}$$

$$\text{V-d: } S_\varepsilon(a, b) = \mathrm{OT}_\varepsilon(a, b) - \tfrac12 \mathrm{OT}_\varepsilon(a, a) - \tfrac12 \mathrm{OT}_\varepsilon(b, b)$$

$$\text{V-e: } \mathrm{OT}_\varepsilon(a, b)$$

V-a wird in der Lagrange-Form gerechnet: das $\varepsilon'$ je Paar, bei dem $\mathrm{KL}(P_{\varepsilon'} \,\|\, a b^{\top}) = \alpha$ gilt (Bisektion auf $\ln \varepsilon'$), der Wert ist $\langle C, P_{\varepsilon'}\rangle$.

**Handbeispiel Selbstabstand** ($2 \times 2$, $a = b = (\tfrac12, \tfrac12)$, $C = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$): $P_{11} = u^2$, $P_{12} = u^2 q$ mit $q = e^{-1/\varepsilon}$, $u^2 (1 + q) = \tfrac12$, also
$$\langle C, P_\varepsilon\rangle = 2 u^2 q = \frac{q}{1 + q}, \qquad \varepsilon = 1: \quad \frac{1}{1 + e} = 0{,}2689.$$
Drei Zellen mit den Kosten $(\Delta/2)^2$, $a = (\tfrac13, \tfrac13, \tfrac13)$, $b = (0{,}4;\ 0{,}2;\ 0{,}4)$: Überschuss $\tfrac{2}{15}$ in der Mitte, Fehlbedarf je $\tfrac1{15}$ außen, Kosten je $\tfrac14$, also
$$W_2^2(a, b) = \Big(\tfrac1{15} + \tfrac1{15}\Big) \cdot \tfrac14 = \tfrac1{30}.$$
Bei $\varepsilon = 0{,}25$ ist $\langle C, P_\varepsilon\rangle(a, b) = 0{,}0855 < 0{,}0869 = \langle C, P_\varepsilon\rangle(a, a)$: die andere Verteilung ist „näher“ als die Verteilung selbst.

**Gauß-Fall (Prognosen).** Für $N(m, s^2)$ gegen $N(0, 1)$: $W_2^2 = m^2 + (s - 1)^2$; mit gleicher Streuung ist $W_1 = |m|$, mit gleicher Lage $W_1 = |s - 1|\sqrt{2/\pi}$;
$$\mathbb{E}[\mathrm{CRPS}] = \underbrace{\int (F - G)^2\,\mathrm{d}x}_{\text{Cramér-Abstand}^2} + \frac{1}{\sqrt{\pi}}, \qquad \mathbb{E}|X - y| \text{ wächst mit } s: \quad \frac{\partial}{\partial s}\,\mathbb{E}|sZ - y| = 2\varphi(y/s) > 0.$$
Der erwartete CRPS ist das Quadrat der L2-Norm der Verteilungsfunktionsdifferenz $F - G$ plus eine Konstante ($1/\sqrt{\pi} = 0{,}5642$; deshalb ist der erwartete CRPS der wahren Prognose F0 nicht 0), W1 ist ihre L1-Norm.

**KL ohne Unterschied.** Zwei Stichproben aus derselben Gleichverteilung über 4 Zellen mit je 3 Beobachtungen: von den $4096$ Paaren von Folgen hat in $\tfrac{789}{1024}$ der Träger von $p$ eine Zelle, die nicht im Träger von $q$ liegt; $\mathrm{KL} = \infty$.

**Literatur** (nur so zitiert, wie gelesen): Cuturi (2013), *Sinkhorn distances: lightspeed computation of optimal transportation distances*, arXiv 1306.0895 (Fassung v1); Genevay, Peyré, Cuturi (2018), *Learning generative models with Sinkhorn
divergences*, arXiv 1706.00292; Feydy u. a. (2019), *Interpolating between optimal transport and MMD using Sinkhorn divergences*, arXiv 1810.08278; Arjovsky, Chintala, Bottou (2017), *Wasserstein GAN*, arXiv 1701.07875 (Beispiel 1);
Vershik (2005), *Kantorovich metric: initial history and little-known applications*, arXiv math/0503035; Kolouri u. a. (2019), *Generalized sliced Wasserstein distances*, arXiv 1902.00434 (Gleichung 3); Resin u. a., *Shift-dispersion
decompositions of Wasserstein and Cramér distances*, arXiv 2408.09770 v3 (Preprint, Auszug gelesen, nicht nachgerechnet). Peyré, Cuturi (2019), *Computational Optimal Transport*: ab Kapitel 4 nicht gelesen. Die Formeln für TV, KS, KL, JS und die Quantilformel sind Lehrstoff;
die Kreisformel für W1 ist aus dem Gedächtnis und an einem Linearprogramm geprüft, nicht aus einer gelesenen Quelle.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
