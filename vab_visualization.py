"""Plotly-Figuren: alle Achsen fest (fixedrange), Beschriftungen als Annotationen, ein Rahmen je Figur."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import vab_constants as C
import vab_profile as PR

SCHRIFT = dict(size=12)


def _fest(fig: go.Figure, hoehe: int, titel: str | None = None) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=hoehe, margin=dict(l=55, r=20, t=50 if titel else 30, b=80), title=dict(text=titel, x=0.0, font=dict(size=14)) if titel else None,
                      dragmode=False, legend=dict(orientation="h", y=-0.32, x=0.0), font=SCHRIFT)
    return fig


def build_profile(a: np.ndarray, b: np.ndarray, zellen: int, verschiebung: float) -> go.Figure:
    """Ausgangsprofil und verschobenes Profil als Stufen über den Tag (Anteil je Zelle)."""
    x = PR.zellenmitten(zellen)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=a, name="Ausgangsprofil A", marker_color=C.COLOR_A, opacity=0.6, width=C.TAG / zellen))
    fig.add_trace(go.Bar(x=x, y=b, name=f"verschoben um {verschiebung:g} h (B)".replace(".", ","), marker_color=C.COLOR_B, opacity=0.6, width=C.TAG / zellen))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Uhrzeit (Stunden)", range=[0, C.TAG], tickvals=list(range(0, 25, 4)))
    fig.update_yaxes(title="Anteil je Zelle", rangemode="tozero")
    return _fest(fig, 300)


def build_kurven1(reihe: dict, s_marker: float, glaettung: float) -> go.Figure:
    """Drei Felder: W1 in Stunden; TV, KS, JS (dimensionslos bzw. nat); KL (nat, ∞ als Marke am oberen Rand)."""
    sv = reihe["s"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("W1 (Stunden)", "TV, KS, JS", "KL (nat)"), horizontal_spacing=0.08)
    fig.add_trace(go.Scatter(x=sv, y=reihe["w1"], mode="lines+markers", name="W1", line=dict(color=C.COLOR_W1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sv, y=reihe["tv"], mode="lines+markers", name="TV", line=dict(color=C.COLOR_TV)), row=1, col=2)
    fig.add_trace(go.Scatter(x=sv, y=reihe["ks"], mode="lines+markers", name="KS", line=dict(color=C.COLOR_KS, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=sv, y=reihe["js"], mode="lines+markers", name="JS", line=dict(color=C.COLOR_JS)), row=1, col=2)
    kl = np.array(reihe["kl"], dtype=float)
    endlich = np.isfinite(kl)
    kmax = float(kl[endlich].max()) if endlich.any() else 1.0
    kmax = max(kmax, 1.0)                                 # kein Bereich um 0 (Zehnerpotenz-Beschriftung): mindestens bis 1 nat
    oben = kmax * 1.25
    fig.add_trace(go.Scatter(x=[x for x, e in zip(sv, endlich) if e], y=kl[endlich], mode="lines+markers", name="KL", line=dict(color=C.COLOR_KL)), row=1, col=3)
    unendlich = [x for x, e in zip(sv, endlich) if not e]
    if unendlich:
        fig.add_trace(go.Scatter(x=unendlich, y=[oben] * len(unendlich), mode="markers", name="KL = ∞", marker=dict(color=C.COLOR_KL, symbol="x", size=9)), row=1, col=3)
    for spalte in (1, 2, 3):
        fig.add_vline(x=s_marker, line=dict(color="rgba(120,120,120,0.7)", dash="dash"), row=1, col=spalte)
        fig.update_xaxes(title="Verschiebung s (Stunden)", range=[0, C.S1_MAX], row=1, col=spalte)
    fig.update_yaxes(range=[0, C.S1_MAX * 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    fig.update_yaxes(range=[0, oben * 1.08], row=1, col=3)
    if unendlich:
        fig.add_annotation(text="∞", x=unendlich[0], y=oben, xref="x3", yref="y3", showarrow=False, yshift=14, font=dict(color=C.COLOR_KL, size=14))
    fig.update_layout(showlegend=True)
    return _fest(fig, 360)


def build_treppe(zeilen: list[dict], glatt: bool) -> go.Figure:
    """Links W1 im Raster gegen die Zellenzahl (Log-Achse) und der wahre Wert; rechts der Fehler in Zellenbreiten."""
    b = [z["zellen"] for z in zeilen]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("W1 im Raster gegen wahres W1 (Anteil des Tages)", "Fehler in Zellenbreiten"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=b, y=[z["w1"] for z in zeilen], mode="lines+markers", name="Raster", line=dict(color=C.COLOR_W1, shape="hv")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[min(b), max(b)], y=[zeilen[0]["wahr"]] * 2, mode="lines", name="wahr", line=dict(color=C.COLOR_KL, dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=[str(x) for x in b], y=[max(z["fehler_h"], 1e-16) for z in zeilen], name="|Fehler| / h", marker_color=C.COLOR_A), row=1, col=2)
    fig.update_xaxes(type="log", title="Zellen", row=1, col=1)
    fig.update_xaxes(title="Zellen", type="category", row=1, col=2)
    ymax = max(max(z["w1"] for z in zeilen), zeilen[0]["wahr"]) * 1.2
    fig.update_yaxes(range=[0, ymax], row=1, col=1)
    if glatt:
        fig.update_yaxes(type="log", range=[-15, 0], row=1, col=2)
    else:
        fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    return _fest(fig, 360)


def build_anteil3(zeilen: list[dict]) -> go.Figure:
    """Anteil der Zufalls-Seeds mit KL = ∞ je Zellenzahl, beide Richtungen."""
    x = [str(z["zellen"]) for z in zeilen]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[100 * z["anteil_xy"] for z in zeilen], name="KL(x ‖ y) = ∞", marker_color=C.COLOR_KL))
    fig.add_trace(go.Bar(x=x, y=[100 * z["anteil_yx"] for z in zeilen], name="KL(y ‖ x) = ∞", marker_color=C.COLOR_A))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Zellen B", type="category")
    fig.update_yaxes(title="Anteil der Zufalls-Seeds (%)", range=[0, 105])
    return _fest(fig, 300)


def build_w1_hist(z: dict) -> go.Figure:
    """W1 aus dem Histogramm und aus den sortierten Stichproben, Verteilung über die Zufalls-Seeds (Kästen, 5 bis 95 %)."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=z["w1_hist"], name=f"aus Histogramm ({z['zellen']} Zellen)", marker_color=C.COLOR_W1, boxpoints=False, whiskerwidth=0.6, quartilemethod="linear"))
    fig.add_trace(go.Box(y=z["w1_sort"], name="aus sortierten Stichproben", marker_color=C.COLOR_B, boxpoints=False, whiskerwidth=0.6))
    fig.update_yaxes(title="W1 (Anteil des Tages)", rangemode="tozero")
    return _fest(fig, 300)


def build_selbst4(res: dict, verw: str) -> go.Figure:
    """Je Verteilung: Selbstabstand gegen den kleinsten Wert zu einer anderen Verteilung (rot, wenn eine andere näher ist als die Verteilung selbst)."""
    m = res["matrizen"][verw]
    n = len(m)
    diag = np.diag(m)
    ohne = (m + np.diag(np.full(n, np.inf))).min(axis=1)
    farben = [C.COLOR_KL if d > o + 1e-15 else C.COLOR_A for d, o in zip(diag, ohne)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, n + 1)), y=diag, marker_color=farben, name="Wert der Verteilung gegen sich selbst"))
    fig.add_trace(go.Scatter(x=list(range(1, n + 1)), y=ohne, mode="markers", marker=dict(color="black", symbol="line-ew", size=14, line=dict(width=2)), name="kleinster Wert zu einer anderen Verteilung"))
    fig.update_xaxes(title="Verteilung", dtick=1)
    fig.update_yaxes(title=f"{verw} (Kosten normiert, max C = 1)", rangemode="tozero")
    return _fest(fig, 320)


def build_streu4(res: dict, verw: str) -> go.Figure:
    """Wert der Verwendung gegen das exakte W2² je Paar (276 Punkte) mit den Geraden y = x und y = 2x."""
    m = res["matrizen"][verw]
    exakt = np.array(res["exakt"])
    wert = np.array([m[i, j] for i, j in res["paare"]])
    xmax = float(exakt.max()) * 1.05
    ymax = float(max(wert.max(), 2 * exakt.max())) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exakt, y=wert, mode="markers", marker=dict(color=C.COLOR_A, size=5, opacity=0.7), name=f"{verw} je Paar"))
    fig.add_trace(go.Scatter(x=[0, xmax], y=[0, xmax], mode="lines", line=dict(color="rgba(120,120,120,0.8)", dash="dash"), name="y = x"))
    fig.add_trace(go.Scatter(x=[0, xmax], y=[0, 2 * xmax], mode="lines", line=dict(color="rgba(200,120,30,0.8)", dash="dot"), name="y = 2x"))
    fig.update_xaxes(title="exaktes W2² (max C = 1)", range=[0, xmax])
    fig.update_yaxes(title=f"{verw}", range=[0, ymax])
    return _fest(fig, 320)


def build_prognose5(tab: list[dict]) -> go.Figure:
    """Vier Felder (W2², W1, E[CRPS], E|X − y|): je Prognose ein Balken, die Reihenfolge der Prognosen bleibt fest (F0 bis F4)."""
    namen = [z["name"].split()[0] for z in tab]
    fig = make_subplots(rows=1, cols=4, subplot_titles=("W2²", "W1", "E[CRPS]", "E|X − y|"), horizontal_spacing=0.06)
    for spalte, key in enumerate(("w2", "w1", "ecrps", "exy"), start=1):
        werte = [z[key] for z in tab]
        rang = {n: r + 1 for r, n in enumerate(sorted(namen, key=lambda n: werte[namen.index(n)]))}
        fig.add_trace(go.Bar(x=namen, y=werte, marker_color=[C.COLOR_W1 if n == "F0" else C.COLOR_A for n in namen], showlegend=False,
                             text=[f"Rang {rang[n]}" for n in namen], textposition="outside"), row=1, col=spalte)
        fig.update_yaxes(range=[0, max(werte) * 1.3], row=1, col=spalte)
    return _fest(fig, 320)


def build_platz1(mc: dict) -> go.Figure:
    regeln = list(mc["platz1"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=regeln, y=[100 * mc["platz1"][r] for r in regeln], marker_color=C.COLOR_W1, text=[f"{100 * mc['platz1'][r]:.1f}".replace(".", ",") + " %" for r in regeln], textposition="outside"))
    fig.update_yaxes(title="wahre Prognose auf Platz 1 (% der Zufalls-Seeds)", range=[0, 112])
    return _fest(fig, 300)
