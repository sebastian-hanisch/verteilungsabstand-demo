# Abstand zwischen Verteilungen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-verteilungsabstand-demo.streamlit.app/)**

Demo der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt an zwei Annahmen an, die die Demos „Vom Zuordnen zum Transportieren“ und „Sinkhorn: weich statt scharf“ stehen lassen: *Lager, Kunden und Kosten sind gegeben; der Optimalwert ist das Ende der Rechnung, ein Preis für einen Auftrag.* und *⟨C, P_ε⟩ ist eine Näherung des Transportwerts, die mit ε → 0 besser wird.* Hier ist der Wert eine Abstandsgröße zwischen Verteilungen: Raster und Kostenwahl ändern ihn, und auch bei kleinem ε ist ⟨C, P_ε⟩ kein Abstand (der Selbstabstand ist nicht 0). Fünf Experimente prüfen die Annahme „Ein Abstand misst, wie weit zwei Verteilungen auseinanderliegen“: Verschiebung (Experiment 1), Raster (Experiment 2), Histogramm ohne Unterschied (Experiment 3), Distanz oder Divergenz (Experiment 4) und Prognosebewertung (Experiment 5).

## Warum dieses Problem

„Der Abstand zwischen zwei Verteilungen“ steht in vielen Sätzen: beim Vergleich zweier Tagesprofile der Nachfrage, bei der Bewertung einer Prognose, bei der Frage, ob sich eine Verteilung geändert hat. Welche Zahl gemeint ist, ist eine Wahl mit zählbaren Folgen. Die Demo stellt dieselbe Verschiebung eines Tagesprofils den fünf gebräuchlichen Zahlen gegenüber (W1, TV, KS, JS, KL), zeigt, dass das Raster die Zahl mitbestimmt, und rechnet das Wort „Sinkhorn-Distanz“ in seinen fünf Verwendungen (V-a bis V-e) nebeneinander. Es geht um **Tagesprofile**: die Verteilung der Nachfrage über 24 Stunden auf einem Raster von Zellen (Objekt Zeit). „Tagesprofil“ heißt hier immer eine Verteilung über 24 Stunden; ein einzelner Tageswert ist etwas anderes.

## Modell

Ein **Tagesprofil** ist ein Vektor $p$ von $B$ Zellenwerten mit Summe 1 und Zellenbreite $h = 24/B$ Stunden; die Zellenmitte der Zelle $k$ liegt bei $(k + \tfrac12)\,h$. Formeln und Handrechnungen stehen im Expander „📐 Mathematische Formulierung“ der App. Eine ausführliche, verständlich geschriebene Erklärung mit Fachteil steht auf der Seite [Der Abstand zwischen Verteilungen verständlich erklärt](https://sebastianhanisch.net/abstand-zwischen-verteilungen-erklaert.html); die App fasst sie im Expander „💬 In einfachen Worten“ zusammen.

- **Abstände:** der **Wasserstein-Abstand** (englisch Wasserstein distance) W1 aus den Verteilungsfunktionen, $h \sum_k |F_k - G_k|$, und W2² über die Quantilfunktionen (beide 1D, exakt auf dem Raster); der **Totalvariationsabstand** TV, der **Kolmogorov-Smirnov-Abstand** KS, die **Kullback-Leibler-Divergenz** KL mit Glättung $g$ je Zelle (ohne Renormierung; $g = 0$ ist die reine KL) und die **Jensen-Shannon-Divergenz** JS. Alle Logarithmen sind natürliche (Einheit nat).
- **„Sinkhorn-Distanz“ in fünf Verwendungen:** **V-a** die Kosten unter der Nebenbedingung $\mathrm{KL}(P \,\|\, a b^{\top}) \le \alpha$ (Cuturi 2013, Definition 1), gerechnet in der Lagrange-Form (das $\varepsilon'$ je Paar, bei dem KL gleich $\alpha$ ist); **V-b** die Kosten $\langle C, P_\varepsilon\rangle$ unter dem weichen Plan ohne Entropieterm; **V-c** die zentrierte Größe $2W(a,b) - W(a,a) - W(b,b)$ mit $W$ = V-b (Genevay u. a. 2018, Faktor 2); **V-d** die zentrierte Größe $S_\varepsilon$ mit Entropieterm und Faktor ½ (Feydy u. a. 2019); **V-e** der Wert mit Entropieterm ohne Zentrieren. Jede Zahl der Demo nennt die Verwendung.
- **Instanzen:** Experiment 1: Kasten (Gleichverteilung von 6 bis 9 Uhr) oder zwei Spitzen (Normalverteilungen bei den Zellenmitten 6:30 und 14:30 Uhr, Gewichte 0,4 und 0,6, Standardabweichung 1 Stunde), 24, 48 oder 96 Zellen, Verschiebung 0 bis 8 Stunden in Schritten von 0,5. Experiment 2: zwei Verteilungen auf [0, 1] (dem Tag), erste Spitze bei 0,3. Experiment 3: Stichproben aus N(0,3; 0,04²) auf [0, 1]. Experiment 4: 24 Tagesprofile aus Mischungen von 1 bis 3 Gauß-Hügeln (Zentren 0,15 + 0,7 U des Tages, Breite 0,03 + 0,07 U des Tages, Gewichte Dirichlet(1), Boden 1e-12 auf der Dichte, normiert) auf 48 Zellen à 30 Minuten, 276 Paare, 2 024 Tripel; die **Verteilungsmenge** ist durch ihren Zufalls-Seed festgeschrieben (Menge 1, Prüfsumme der Profile im Test). Experiment 5: fünf Gauß-Prognosen gegen die Wahrheit N(0, 1).
- **Skala:** $\varepsilon = c \cdot D$ mit $c = \varepsilon/D$ und $D$ = Median von W2² über die Paare der gezeigten Menge, Kosten $(\Delta t)^2$ durch den größten Wert geteilt (max C = 1). Der Selbstabstand hat das Optimum 0, ε relativ zum Optimum (wie in den Demos zu Sinkhorn) wäre dort nicht definiert; die Zahlen aus jenen Demos sind nicht gegen die Zahlen hier zu lesen.
- **Zählweise:** Zähleinheiten sind Paare (276), Verteilungen (24), Tripel (2 024) und Zufalls-Seeds (Wiederholungen). Eine Sinkhorn-Iteration sind **zwei Matrix-Vektor-Produkte**; der Aufwand erscheint als Nebenanzeige in Produkten, nie in Sekunden. Abbruch je Paar bei einem Zeilenfehler unter 1e-10 (V-a: 1e-9), gerechnet naiv in float64 im Kern-Bereich, höchstens 200 000 Durchläufe; ein fertiges Paar wird eingefroren.
- **Zufall:** Hausgenerator SplitMix64 (`vab_rng.py`, auf Windows und Linux bitgleich) für die Verteilungsmenge; die Stichproben der Experimente 3 und 5 benutzen dieselben Zufallszahlen vektorisiert und Box-Muller mit numpy-Funktionen (in der letzten Binärstelle plattformabhängig; ein Einfluss auf die Zählungen ist nur in Grenzfällen zu erwarten und auf einer zweiten Plattform nicht geprüft). Die Reihen dieser README benutzen den Zufalls-Seed 2026, unabhängig vom Regler.

## Methodik

- **Abstände** (`vab_abstand.py`): W1, W1 auf dem Kreis (Formel $h \sum_k |D_k - \operatorname{Median}(D)|$, aus dem Gedächtnis und an einem Linearprogramm geprüft), W2², TV, KS, KL, JS, Kendall τ.
- **Profile und Zufall** (`vab_profile.py`): Kasten, zwei Spitzen, Gauß-Mischungen, Zellenintegrale, Verteilungsmenge, vektorisierter SplitMix64.
- **Sinkhorn im Stapel** (`vab_sinkhorn.py`): die Iteration für alle Paare gleichzeitig (gemeinsames Kostenmatrix, Start $u = v = 1$), Verwendungen V-a bis V-e, Zählung der Dreiecksverletzungen.
- **Auswertung** (`vab_evaluation.py`): Reihen der fünf Experimente, Kennzahlen, Verdikte als reine Funktionen, exakte Wahrscheinlichkeit von KL = ∞ für kleine Stichproben (Inklusion-Exklusion), Bewertung von Prognosen gegen empirische Verteilungen in geschlossener Form.
- **Zufall:** `vab_rng.py` ist eine Kopie der Vorlage des Portfolios (Test auf Gleichheit).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus dem Code der Demo; jede steht in `tests/test_claims.py` (Tabellen und Sätze wörtlich) oder in den dort genannten Handrechnungen. Die Reihe über fünf Verteilungsmengen steht in der gespeicherten Datei `tools/messreihen.json` (vorgerechnet mit `tools/messreihen.py`): der Test vergleicht die Spannen und Zählungen über die Mengen 1 bis 5 mit dieser Datei und rechnet nur die Menge 1 bei ε/D = 8 und 1 neu nach; die übrigen Zellen der Reihe (Mengen 2 bis 5 sowie ε/D = 3 und 0,3) sind im Test nur mit der Datei verglichen. Die Tabellen zur Menge 1 bei ε/D = 8, 1 und 0,3 werden davon unabhängig im Test jedes Mal neu gerechnet.

### Experiment 1: Verschieben statt Überlappen

Ausgangsprofil und um s Stunden verschobenes Profil, 24 Zellen, Zeitachse Gerade. **Kasten 3 h** (KL(B ‖ A) ohne Glättung):

| s (Stunden) | W1 (Stunden) | TV | KS | JS (nat) | KL ohne Glättung (nat) |
|---|---|---|---|---|---|
| 0 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |
| 1 | 1,000 | 0,333 | 0,333 | 0,231 | ∞ |
| 2 | 2,000 | 0,667 | 0,667 | 0,462 | ∞ |
| 3 | 3,000 | 1,000 | 1,000 | 0,693 | ∞ |
| 4 | 4,000 | 1,000 | 1,000 | 0,693 | ∞ |
| 6 | 6,000 | 1,000 | 1,000 | 0,693 | ∞ |

Kasten 3 h, 24 Zellen, Zeitachse Gerade: W1 = s Stunden in allen 17 Verschiebungen; TV = KS in allen Verschiebungen; TV sättigt bei 1 ab s = 3,0 Stunden; JS erreicht ln 2 = 0,693 ab s = 3,0 Stunden; KL ist ohne Glättung schon ab s = 0,5 Stunden unendlich (jede Verschiebung füllt eine Zelle, die im Ausgangsprofil leer ist). In diesem Modell, eine Zeile je Verschiebung.

**Zwei Spitzen** (6:30 und 14:30 Uhr, σ = 1 Stunde), KL(B ‖ A) mit Glättung g = 1e-12 und ohne:

| s (Stunden) | W1 (Stunden) | TV | KS | JS (nat) | KL mit g = 1e-12 (nat) | KL ohne Glättung (nat) |
|---|---|---|---|---|---|---|
| 1 | 1,000 | 0,399 | 0,239 | 0,111 | 0,498 | 0,498 |
| 2 | 2,000 | 0,639 | 0,385 | 0,335 | 1,964 | 1,964 |
| 3 | 3,000 | 0,878 | 0,530 | 0,519 | 4,199 | 4,199 |
| 4 | 4,000 | 0,911 | 0,562 | 0,603 | 6,640 | 6,641 |
| 6 | 5,999 | 0,840 | 0,597 | 0,516 | 11,160 | 11,410 |

Zwei Spitzen, 24 Zellen, Zeitachse Gerade, σ = 1,0 Stunden, bei s = 1 / 2 / 3 / 4 / 6 Stunden: W1 = 1,000 / 2,000 / 3,000 / 4,000 / 5,999 Stunden; TV = 0,399 / 0,639 / 0,878 / 0,911 / 0,840; KL (mit g = 1e-12) = 0,498, 1,964, 4,199, 6,640, 11,160; TV ist nicht monoton: Höchstwert 0,935 bei s = 4,5, bei s = 8,0 wieder 0,585. In diesem Modell, eine Zeile je Verschiebung (Tabelle).

W1 = 5,999 Stunden bei s = 6 statt 6,000: die Zellen sind bei 0 Uhr und 24 Uhr abgeschnitten.

**Die KL-Zahl hängt an der Glättung**, sobald eine Zelle fast leer ist (zwei Spitzen, KL in nat):

| s (Stunden) | g = 0 | g = 1e-12 | g = 1e-9 | g = 1e-6 |
|---|---|---|---|---|
| 4 | 6,641 | 6,640 | 6,618 | 6,312 |
| 6 | 11,410 | 11,160 | 10,127 | 7,486 |

**Gerade oder Kreis.** Ein Tagesprofil liegt auf dem Kreis (Mitternacht). Eine Spitze um 23:30 Uhr gegen eine Spitze um 1:30 Uhr (24 Zellen): W1 ist 22 Stunden auf der Geraden und 2 Stunden auf dem Kreis. Der Regler „Zeitachse“ schaltet W1 und die Profile um; TV, KS, JS und KL rechnen über die Zellenwerte.

*Ehrlich:* W1 ist hier der einzige gezeigte Abstand, der bei getrennten Trägern (Kasten) weder sättigt noch unendlich wird; auf dem Raster kann er eine kleine Verschiebung mit 0 melden (Experiment 2). Welcher Abstand für eine Entscheidung der richtige ist, zeigt die Demo nicht.

### Experiment 2: Das Raster macht die Zahl

Zwei Spitzen (ganze Masse in einem Punkt) bei 0,3 und 0,3137 des Tages, Zelle $= \lfloor x B \rfloor$ (exakt in Brüchen gerechnet):

| Zellen B | Zelle der ersten Spitze | Zelle der zweiten Spitze | W1 im Raster | W1 wahr | Betrag des Fehlers / h |
|---|---|---|---|---|---|
| 10 | 3 | 3 | 0,0000 | 0,0137 | 0,137 |
| 20 | 6 | 6 | 0,0000 | 0,0137 | 0,274 |
| 50 | 15 | 15 | 0,0000 | 0,0137 | 0,685 |
| 100 | 30 | 31 | 0,0100 | 0,0137 | 0,370 |
| 200 | 60 | 62 | 0,0100 | 0,0137 | 0,740 |
| 500 | 150 | 156 | 0,0120 | 0,0137 | 0,850 |
| 1 000 | 300 | 313 | 0,0130 | 0,0137 | 0,700 |
| 5 000 | 1500 | 1568 | 0,0136 | 0,0137 | 0,500 |

Zwei Spitzen im Abstand 0,0137 des Tages: Das Raster meldet W1 = 0 bei 10, 20, 50 Zellen (beide Spitzen in einer Zelle), der wahre Wert ist 0,0137; der größte Fehler ist 0,85 Zellenbreiten bei 500 Zellen; der Fehler bleibt in jeder Zeile unter einer Zellenbreite. Die Fehler bilden eine Treppe, kein glattes Konvergieren.

Glatte Profile gleicher Breite (Normalverteilungen mit σ = 0,04 des Tages, Zellenintegrale als Massen), gleicher Abstand:

| Zellen B | W1 im Raster | W1 wahr | Betrag des Fehlers |
|---|---|---|---|
| 10 | 0,014726 | 0,0137 | 1,0e-3 |
| 20 | 0,013700 | 0,0137 | 5,1e-8 |
| 50 | 0,013700 | 0,0137 | 9,1e-15 |
| 100 | 0,013700 | 0,0137 | 9,0e-15 |
| 200 | 0,013700 | 0,0137 | 9,0e-15 |
| 500 | 0,013700 | 0,0137 | 8,4e-15 |
| 1 000 | 0,013700 | 0,0137 | 8,0e-15 |
| 5 000 | 0,013700 | 0,0137 | 8,6e-15 |

Glatte Gauß-Profile gleicher Breite σ = 0,04 des Tages im Abstand 0,0137: ab 50 Zellen ist der größte Fehler 9,1e-15 (Rundung), bei 10 Zellen größer als 1e-6. Der Fehler hängt hier an der Glätte der Profile, nicht an der Zellenzahl allein.

**W2² im Raster.**

| Zellen | σ der zweiten Verteilung | Zellen deckungsgleich | h | Fehler Raster − geschlossen | Fehler / h² |
|---|---|---|---|---|---|
| 25 | 0,7 | nein | 0,560 | 5,20e-2 | 0,1658 |
| 100 | 0,7 | nein | 0,140 | 3,28e-3 | 0,1672 |
| 400 | 0,7 | nein | 0,035 | 2,03e-4 | 0,1660 |
| 28 | 0,7 | ja | 0,500 | 3,83e-2 | 0,1533 |
| 112 | 0,7 | ja | 0,125 | 2,57e-3 | 0,1643 |
| 25 | 1,0 | nein | 0,560 | 6,84e-2 | 0,2181 |
| 100 | 1,0 | nein | 0,140 | 4,00e-3 | 0,2041 |
| 400 | 1,0 | nein | 0,035 | 1,50e-4 | 0,1224 |
| 28 | 1,0 | ja | 0,500 | −1,61e-8 | −6,4e-8 |
| 112 | 1,0 | ja | 0,125 | −1,82e-8 | −1,2e-6 |

W2² zwischen N(0, 1) und N(1,5; σ₂²) auf [−6, 8]. Bei ungleicher Streuung (σ₂ = 0,7) ist der Rasterfehler geteilt durch h² 0,1658, 0,1672, 0,1660 bei 25, 100, 400 Zellen und 0,1533, 0,1643 bei 28, 112 Zellen (dort ist die Verschiebung 1,5 ein Vielfaches von h): in allen 5 Zeilen positiv (das Raster überschätzt), zwischen 0,1533 und 0,1672 (1/6 = 0,1667), auch bei deckungsgleichen Zellen. Bei gleicher Streuung (σ₂ = 1) ist er 0,2181, 0,2041, 0,1224 bei 25, 100, 400 Zellen und schwankt mit der Zellenzahl; liegen die Zellen deckungsgleich (Verschiebung 1,5 ein Vielfaches von h bei 28, 112 Zellen), ist er nur 6e-8, 1e-6 (Abschneiden der Ränder), weil das zweite Raster dann das um ganze Zellen verschobene erste ist. Die Konstante nahe 1/6 gilt hier also bei ungleicher Streuung, unabhängig von der Lage der Zellen, und nicht bei gleicher Streuung.

Die Konstante nahe 1/6 ist beobachtet (an einem Paar von Normalverteilungen mit den Streuungen 1 und 0,7; die Werte der Tabelle liegen bei 0,153 bis 0,167, nicht genau bei 1/6); eine Herleitung gibt es nicht. Vermutet (weder bewiesen noch in der Demo gemessen) ist, dass es am Rundungsfehler der Quantile liegt: Wäre er je Verteilung gleichverteilt über eine Zellenbreite, also mit Varianz $h^2/12$, und unabhängig, ergäben beide zusammen $2 \cdot h^2/12 = h^2/6$. Eine zusätzliche Varianz $h^2/12$ je Verteilung (Streuungen $\sigma_i^2 + h^2/12$) erklärt die Konstante dagegen nicht: Für Normalverteilungen mit den Streuungen 1 und 0,7 verkleinerte sie $W_2^2$ nach einer Handrechnung erster Ordnung, $2(\sigma_1 - \sigma_2)(1/\sigma_1 - 1/\sigma_2)\,h^2/24 \approx -0{,}011\,h^2$, statt es um etwa $h^2/6$ zu vergrößern. Für gleiche Streuung erklärt sie nichts: dort ist der Fehler null oder hängt an der Zellenzahl.

### Experiment 3: Histogramm ohne Unterschied

Zwei Stichproben mit je n = 500 Beobachtungen, 300 Zufalls-Seeds (Wiederholungen), Zufalls-Seed 2026, ohne Glättung. Bei d = 0 stammen beide Stichproben aus **derselben** Verteilung:

| Zellen B | KL(x ‖ y) = ∞ | KL(y ‖ x) = ∞ | W1 aus dem Histogramm (Median) | W1 aus sortierten Stichproben (Median) |
|---|---|---|---|---|
| 10 | 8,0 % | 8,3 % | 0,0030 | 0,0031 |
| 50 | 58,7 % | 60,7 % | 0,0031 | 0,0031 |
| 200 | 99,7 % | 98,7 % | 0,0031 | 0,0031 |
| 1 000 | 100,0 % | 100,0 % | 0,0031 | 0,0031 |

Bei einer Verschiebung d = 0,02 hängt der Anteil bei kleinem B an der Richtung der KL:

| Zellen B | KL(x ‖ y) = ∞ | KL(y ‖ x) = ∞ | W1 aus dem Histogramm (Median) | W1 aus sortierten Stichproben (Median) |
|---|---|---|---|---|
| 10 | 50,3 % | 6,7 % | 0,0218 | 0,0204 |
| 50 | 80,3 % | 82,3 % | 0,0204 | 0,0204 |
| 200 | 99,7 % | 100,0 % | 0,0204 | 0,0204 |
| 1 000 | 100,0 % | 100,0 % | 0,0204 | 0,0204 |

Bei d = 0,05:

| Zellen B | KL(x ‖ y) = ∞ | KL(y ‖ x) = ∞ | W1 aus dem Histogramm (Median) | W1 aus sortierten Stichproben (Median) |
|---|---|---|---|---|
| 10 | 90,0 % | 9,7 % | 0,0501 | 0,0504 |
| 50 | 99,3 % | 99,0 % | 0,0505 | 0,0504 |
| 200 | 100,0 % | 100,0 % | 0,0504 | 0,0504 |
| 1 000 | 100,0 % | 100,0 % | 0,0504 | 0,0504 |

n = 500 aus N(0,3; 0,04²) und N(0,3 + d; 0,04²) mit d = 0,00 (Stichproben aus derselben Verteilung), ohne Glättung, 300 Zufalls-Seeds: KL(x ‖ y) = ∞ in 8,0, 58,7, 99,7, 100,0 % der Zufalls-Seeds bei 10 / 50 / 200 / 1 000 Zellen; KL(y ‖ x) in 8,3, 60,7, 98,7, 100,0 %. W1 aus demselben Histogramm bleibt endlich (Median bei 10 Zellen 0,0030, über die sortierten Stichproben 0,0031). Jeder Anteil ist aus 300 Zufalls-Seeds gezählt und hat eine Standardabweichung von höchstens 2,9 Prozentpunkten (bei 8 % etwa 1,6, bei 99 % etwa 0,6); Unterschiede von wenigen Prozentpunkten zwischen zwei Anteilen sind Rauschen. Das ist ein Beispiel mit einer Dimension und einem n; wie der Rauschboden mit n und Dimension wächst, zeigt diese Demo nicht.

**Klein gerechnet (exakt):** zwei Stichproben mit je 3 Beobachtungen aus derselben Gleichverteilung über B Zellen: P(KL = ∞) ist 789 / 1 024 = 0,7705 bei B = 4, 0,9602 bei B = 8 und 0,9942 bei B = 16. Das ist der Mechanismus: je mehr Zellen, desto öfter ist in der zweiten Stichprobe eine Zelle leer, in der die erste etwas hat.

*Grenze:* Das Experiment zeigt Stichprobenrauschen an einem Beispiel (eine Dimension, n = 500 oder das gewählte n). Es sagt nicht, wie der Rauschboden von W1 mit n und der Dimension wächst.

### Experiment 4: Distanz oder Divergenz?

24 Tagesprofile auf 48 Zellen, 276 Paare, Verteilungsmenge 1. Ein Wert pro Verwendung; „selbst nicht am nächsten“ zählt die Verteilungen, für die eine andere einen kleineren Wert hat als die Verteilung selbst. Bei **ε/D = 8**:

| Verwendung | Selbstabstand (Mittel) | selbst nicht am nächsten | Kendall τ gegen W2² | nächster Nachbar wie exakt | Median Wert / W2² | Dreiecksungleichung verletzt |
|---|---|---|---|---|---|---|
| V-a Nebenbedingung KL ≤ α | 0,0196 | 17 von 24 | 0,809 | 8 von 24 | 1,49 | 0 von 2 024 |
| V-b ⟨C, P_ε⟩ | 0,0167 | 17 von 24 | 0,833 | 8 von 24 | 1,45 | 0 von 2 024 |
| V-c Genevay (Faktor 2) | 0,0000 (Konstruktion) | 0 von 24 (Konstruktion) | 0,930 | 22 von 24 | 1,83 | 8 von 2 024 |
| V-d Feydy (Faktor ½) | 0,0000 (Konstruktion) | 0 von 24 (Konstruktion) | 0,892 | 22 von 24 | 0,88 | 3 von 2 024 |
| V-e OT_ε ohne Zentrieren | 0,0188 | 17 von 24 | 0,816 | 8 von 24 | 1,49 | 0 von 2 024 |

D = 0,0268, ε = 0,2142; V-a: α = 0,0020, bei 0 von 276 Paaren ist die Nebenbedingung nicht wirksam; Läufe ohne Konvergenz innerhalb von 200 000 Durchläufen: 0 bei V-b bis V-e, 1 in den Bisektionsläufen von V-a; Aufwand der Läufe für V-b bis V-e (Nebenanzeige): im Median 12 Matrix-Vektor-Produkte je Paar, am größten 22.

Verteilungsmenge 1, 24 Verteilungen, 276 Paare: bei ε/D = 8,0 hat V-b für 17 der 24 Profile ein anderes Profil näher als sich selbst (V-e: 17, V-a: 17); V-c und V-d haben den Selbstabstand 0 durch Konstruktion; der Median von Wert / W2² ist bei V-c 1,83 und bei V-d 0,88; Kendall τ gegen das exakte W2²: V-b 0,833, V-c 0,930, V-d 0,892; die Dreiecksungleichung der Wurzel ist bei V-c in 8, bei V-d in 3, bei V-a in 0 und bei V-b in 0 von 2 024 Tripeln verletzt. Gilt für diese Menge einer Familie (Mischungen von 1 bis 3 Gauß-Hügeln) und diese ε/D, nicht als Aussage über alle Verteilungen.

Bei **ε/D = 1**:

| Verwendung | Selbstabstand (Mittel) | selbst nicht am nächsten | Kendall τ gegen W2² | nächster Nachbar wie exakt | Median Wert / W2² | Dreiecksungleichung verletzt |
|---|---|---|---|---|---|---|
| V-a Nebenbedingung KL ≤ α | 0,0118 | 16 von 24 | 0,879 | 10 von 24 | 1,27 | 0 von 2 024 |
| V-b ⟨C, P_ε⟩ | 0,0065 | 7 von 24 | 0,955 | 18 von 24 | 1,24 | 0 von 2 024 |
| V-c Genevay (Faktor 2) | 0,0000 (Konstruktion) | 0 von 24 (Konstruktion) | 0,987 | 22 von 24 | 2,00 | 14 von 2 024 |
| V-d Feydy (Faktor ½) | 0,0000 (Konstruktion) | 0 von 24 (Konstruktion) | 0,983 | 22 von 24 | 0,97 | 14 von 2 024 |
| V-e OT_ε ohne Zentrieren | 0,0109 | 13 von 24 | 0,899 | 12 von 24 | 1,35 | 0 von 2 024 |

D = 0,0268, ε = 0,0268; V-a: α = 0,0945, bei 0 von 276 Paaren ist die Nebenbedingung nicht wirksam; Läufe ohne Konvergenz innerhalb von 200 000 Durchläufen: 0 bei V-b bis V-e, 1 in den Bisektionsläufen von V-a; im Median 30 Matrix-Vektor-Produkte je Paar, am größten 246.

Verteilungsmenge 1, 24 Verteilungen, 276 Paare: bei ε/D = 1,0 hat V-b für 7 der 24 Profile ein anderes Profil näher als sich selbst (V-e: 13, V-a: 16); V-c und V-d haben den Selbstabstand 0 durch Konstruktion; der Median von Wert / W2² ist bei V-c 2,00 und bei V-d 0,97; Kendall τ gegen das exakte W2²: V-b 0,955, V-c 0,987, V-d 0,983; die Dreiecksungleichung der Wurzel ist bei V-c in 14, bei V-d in 14, bei V-a in 0 und bei V-b in 0 von 2 024 Tripeln verletzt. Gilt für diese Menge einer Familie (Mischungen von 1 bis 3 Gauß-Hügeln) und diese ε/D, nicht als Aussage über alle Verteilungen.

Bei **ε/D = 0,3**:

| Verwendung | Selbstabstand (Mittel) | selbst nicht am nächsten | Kendall τ gegen W2² | nächster Nachbar wie exakt | Median Wert / W2² | Dreiecksungleichung verletzt |
|---|---|---|---|---|---|---|
| V-a Nebenbedingung KL ≤ α | 0,0047 | 7 von 24 | 0,950 | 19 von 24 | 1,11 | 0 von 2 024 |
| V-b ⟨C, P_ε⟩ | 0,0029 | 1 von 24 | 0,991 | 23 von 24 | 1,11 | 0 von 2 024 |
| V-c Genevay (Faktor 2) | 0,0000 (Konstruktion) | 0 von 24 (Konstruktion) | 0,998 | 24 von 24 | 2,00 | 3 von 2 024 |
| V-d Feydy (Faktor ½) | 0,0000 (Konstruktion) | 0 von 24 (Konstruktion) | 0,997 | 24 von 24 | 0,99 | 7 von 2 024 |
| V-e OT_ε ohne Zentrieren | 0,0062 | 6 von 24 | 0,956 | 17 von 24 | 1,21 | 0 von 2 024 |

D = 0,0268, ε = 0,0080; V-a: α = 0,3774, bei 18 von 276 Paaren ist die Nebenbedingung nicht wirksam (dort zeigt die Tabelle den Wert beim kleinsten ε′ = 0,003, das sind 0,11 D: eine obere Schranke des Werts unter der Nebenbedingung, im Median das 1,04-Fache, höchstens das 10,97-Fache des exakten W2²); Läufe ohne Konvergenz innerhalb von 200 000 Durchläufen: 0 bei V-b bis V-e, 1 in den Bisektionsläufen von V-a.

**Dieselbe Zelle in fünf Verteilungsmengen** (Mengen 1 bis 5, je 24 Profile). Eine Zahl über „die Familie“ ist nur als Bereich über mehrere Mengen zu lesen: D liegt in den Mengen 1 bis 20 zwischen 0,0268 und 0,0601 (Faktor 2,2), in den hier gezeigten Mengen 1 bis 5 zwischen 0,0268 und 0,0601 (D ist je Menge der Median von W2² über die Paare); die Spannen der Tabellen stammen nur aus den Mengen 1 bis 5.

ε/D = 8:

| Verwendung | Kendall τ (Mengen 1 bis 5) | selbst nicht am nächsten (Mengen 1 bis 5) | Median Wert / W2² (Mengen 1 bis 5) |
|---|---|---|---|
| V-a | 0,695 bis 0,826 | 15 bis 20 von 24 | 1,47 bis 1,51 |
| V-b | 0,748 bis 0,847 | 15 bis 20 von 24 | 1,45 bis 1,49 |
| V-c | 0,883 bis 0,937 | 0 von 24 | 1,58 bis 1,83 |
| V-d | 0,837 bis 0,896 | 0 von 24 | 0,72 bis 0,88 |
| V-e | 0,712 bis 0,832 | 15 bis 20 von 24 | 1,48 bis 1,53 |

ε/D = 1:

| Verwendung | Kendall τ (Mengen 1 bis 5) | selbst nicht am nächsten (Mengen 1 bis 5) | Median Wert / W2² (Mengen 1 bis 5) |
|---|---|---|---|
| V-a | 0,810 bis 0,887 | 13 bis 18 von 24 | 1,27 bis 1,31 |
| V-b | 0,951 bis 0,959 | 4 bis 7 von 24 | 1,24 bis 1,26 |
| V-c | 0,975 bis 0,989 | 0 von 24 | 2,00 bis 2,02 |
| V-d | 0,970 bis 0,984 | 0 von 24 | 0,95 bis 0,97 |
| V-e | 0,865 bis 0,904 | 10 bis 13 von 24 | 1,35 bis 1,38 |

Verletzungen der Dreiecksungleichung der Wurzel (von 2 024 Tripeln), Spanne über die Mengen 1 bis 5:

| Verwendung | ε/D = 8 | ε/D = 3 | ε/D = 1 | ε/D = 0,3 |
|---|---|---|---|---|
| V-a | 0 | 0 | 0 | 0 |
| V-b | 0 | 0 | 0 | 0 |
| V-c | 1 bis 8 | 9 bis 18 | 7 bis 14 | 0 bis 6 |
| V-d | 0 bis 4 | 4 bis 12 | 8 bis 14 | 2 bis 7 |
| V-e | 0 | 0 | 0 | 0 |

V-c gegen V-d (Kendall-Rang, Mengen 1 bis 5): bei ε/D = 8 und 3 liegt V-c in 10 von 10 Zellen (Mengen 1 bis 5) im Kendall-Rang vor V-d, die Differenz V-c minus V-d ist 0,028 bis 0,064; bei ε/D = 1 und 0,3 liegt V-c in 10 von 10 Zellen (Mengen 1 bis 5) im Kendall-Rang vor V-d, die Differenz V-c minus V-d ist 0,001 bis 0,008. Die Differenzen bei ε/D ≤ 1 liegen unter der Streuung der Zellen zwischen den Mengen.

*Was die Zahlen zeigen:* Die nicht zentrierten Fassungen V-a, V-b und V-e haben einen Selbstabstand über 0, und für mehrere Verteilungen liegt eine andere näher als sie selbst (bei ε/D = 8 in der Menge 1 bei 17 von 24). V-c und V-d haben den Selbstabstand 0 durch Konstruktion (die Spalte ist dort keine Messung) und ordnen im Rang besser (Kendall τ gegen W2² in allen 20 Zellen der Reihe); zugleich ist V-d bei ε/D = 8 nicht bias-frei (Median von Wert / W2² unter 1 in allen fünf Mengen), und bei V-c und V-d ist die Dreiecksungleichung der Wurzel in einzelnen Tripeln verletzt; bei V-a, V-b und V-e wurde keine Verletzung gezählt (eine Zählung über die gezeigten Tripel, kein Beweis). Bei ε/D = 0,3 liegen V-c und V-d im Median bei 2,00 und 0,99 mal W2² (V-c: der Faktor 2 der Definition steht vor dem Kreuzterm; V-d: der Kreuzterm hat das Gewicht 1, die Hälften stehen nur vor den Selbsttermen, die für kleines ε gegen 0 gehen); bei ε/D = 8 liegen sie bei 1,83 und 0,88.

### Experiment 5: Prognosen bewerten

Wahrheit N(0, 1), fünf Gauß-Prognosen, geschlossen gerechnet (W2² = Lage² + Streuung²; W1 und E[CRPS] über Quadratur und Formel):

| Prognose | Lage² | Streuung² | W2² | W1 | E[CRPS] | E\|X − y\| |
|---|---|---|---|---|---|---|
| F0 wahr | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,5642 | 1,1284 |
| F1 Lage 0,50 | 0,2500 | 0,0000 | 0,2500 | 0,5000 | 0,6340 | 1,1982 |
| F2 σ = 1,60 | 0,0000 | 0,3600 | 0,3600 | 0,4787 | 0,6027 | 1,5054 |
| F3 σ = 0,55 | 0,0000 | 0,2025 | 0,2025 | 0,3590 | 0,6003 | 0,9106 |
| F4 Lage 0,30, σ = 1,30 | 0,0900 | 0,0900 | 0,1800 | 0,3500 | 0,5970 | 1,3305 |

Rangfolgen (klein ist gut): W2²: F0 < F4 < F3 < F1 < F2; W1: F0 < F4 < F3 < F2 < F1; E[CRPS]: F0 < F4 < F3 < F2 < F1; E\|X − y\|: F3 < F0 < F1 < F4 < F2.

W1 und W2² ordnen „Lage um 0,5 falsch“ (F1) und „zu breit“ (F2) entgegengesetzt. Der erwartete CRPS ist das Quadrat der L2-Norm der Verteilungsfunktionsdifferenz plus eine Konstante (1/√π = 0,5642; deshalb ist der E[CRPS] der wahren Prognose F0 nicht 0), W1 ist ihre L1-Norm; in diesen fünf Prognosen fällt ihre Rangfolge zusammen, allgemein nicht. **E|X − y| stellt die zu schmale Prognose vor die wahre**, bei gleicher Lage für jede Stichprobe von Beobachtungen (E|X − y| wächst mit σ; das ist ein Satz, keine Messung). Bewertet gegen die empirische Verteilung von n_obs Beobachtungen, 400 Zufalls-Seeds, Zufalls-Seed 2026:

| Beobachtungen n_obs | Platz 1: W2² | Platz 1: W1 | Platz 1: E[CRPS] | Platz 1: E\|X − y\| | F3 vor F0 bei E\|X − y\| |
|---|---|---|---|---|---|
| 25 | 77,8 % | 77,8 % | 77,0 % | 0,0 % | 400 von 400 |
| 100 | 99,0 % | 98,2 % | 97,5 % | 0,0 % | 400 von 400 |
| 400 | 100,0 % | 100,0 % | 100,0 % | 0,0 % | 400 von 400 |

W1 ordnet die zu breite Prognose F2 vor die verschobene Prognose F1 (0,479 gegen 0,500), W2² ordnet die verschobene Prognose F1 vor die zu breite Prognose F2 (0,250 gegen 0,360); E|X − y| stellt die zu schmale Prognose F3 vor die wahre F0 (0,911 gegen 1,128); das gilt bei gleicher Lage für jede Stichprobe von Beobachtungen, denn E|X − y| wächst mit σ; in den Zufalls-Seeds: 400 von 400; die wahre Prognose F0 liegt bei 100 Beobachtungen auf Platz 1 in 99,0 % (W2²), 98,2 % (W1), 97,5 % (E[CRPS]), 0,0 % (E|X − y|) von 400 Zufalls-Seeds. Die Prozentwerte sind aus 400 Zufalls-Seeds gezählt und haben eine Standardabweichung von höchstens 2,5 Prozentpunkten; Unterschiede von wenigen Prozentpunkten zwischen den Bewertungsregeln sind keine Rangordnung der Regeln. Nur Gauß-Prognosen, bekannte Wahrheit N(0, 1), unabhängige Beobachtungen.

Nicht gemessen: zeitlich abhängige Fehler, Mehrschrittprognosen, Ensembles.

### Handbeispiele von Hand gerechnet

- **Kasten (24 Stundenzellen, 6 bis 9 Uhr, je Stunde 1/3), um d Stunden verschoben (ganze und halbe Stunden):** W1 = d, TV = KS = min(d, 3)/3, KL = ∞ für jedes d > 0 (jede Verschiebung füllt eine Zelle, die leer war), JS = ln 2 für d ≥ 3; in Brüchen (`tests/test_abstand.py`).
- **Treppe:** zwei Spitzen bei 3/10 und 3 137/10 000: bei B = 10, 20, 50 dieselbe Zelle (3, 6, 15), bei B = 500 die Zellen 150 und 156, W1 = 6/500, Fehler 17/20 Zellenbreiten.
- **Selbstabstand:** 2 × 2 mit a = b = (1/2, 1/2): ⟨C, P_ε⟩ = q/(1 + q) mit q = e^(−1/ε), bei ε = 1 gleich 1/(1 + e) = 0,2689. Drei Zellen mit den Kosten (Δ/2)², a gleichverteilt, b = (0,4; 0,2; 0,4): W2² = 1/30; bei ε = 0,25 ist ⟨C, P_ε⟩(a, b) = 0,0855 kleiner als ⟨C, P_ε⟩(a, a) = 0,0869.
- **KL ohne Unterschied:** 4 Zellen, 3 Beobachtungen: 789/1 024 (Aufzählung aller 4 096 Paare und Inklusion-Exklusion).
- **E|X − y|:** ∂/∂σ E|σZ − y| = 2 φ(y/σ) > 0 für jedes y (gegen den Differenzenquotienten).

### Zahl → Test

| Zahl oder Satz | Test (Band oder exakt) |
|---|---|
| alle Tabellen und Messsätze dieser README | `tests/test_claims.py::test_tabellen_und_saetze_der_readme` (wörtlich aus dem Code gebildet; Zählungen auf anderen Plattformen im Band) |
| Streuung über die Mengen 1 bis 5 | `tests/test_claims.py` (Datei `tools/messreihen.json`; nur die Menge 1 bei ε/D = 8 und 1 wird neu gerechnet, im Band) |
| Kasten, Treppe, Zwei Spitzen, KL gegen Glättung | `tests/test_abstand.py` (exakt in Brüchen bzw. Toleranz 6e-4, 2e-3) |
| Rasterfehler von W2² bei ungleicher und gleicher Streuung | `tests/test_abstand.py` (Band 5e-4) |
| Standardabweichung der Anteile, V-a bei unwirksamer Nebenbedingung, Hinweise der Verdikte | `tests/test_evaluation.py`, `tests/test_sinkhorn.py`, `tests/test_claims.py` (Zählungen ± 2, Verhältnisse im Band) |
| Selbstabstand, 1/30, Faktor 2 gegen ½ | `tests/test_sinkhorn.py` |
| Prognosen (Tabelle, Rangfolgen, Satz E\|X − y\|) | `tests/test_oracle_verteilungsabstand.py` |
| 17 Verschiebungen, 276 Paare, 2 024 Tripel, Verteilungsmenge | `tests/test_claims.py::test_konstanten_der_readme`, `tests/test_profile.py` (Prüfsumme) |

## Ehrliche Grenzen

Die Annahmen der Vorgänger und was die Demo daran ändert:

- *Lager, Kunden und Kosten sind gegeben; der Optimalwert ist das Ende der Rechnung, ein Preis für einen Auftrag.* Hier ist der Wert eine Abstandsgröße zwischen Verteilungen; Raster und Kostenwahl ändern ihn.
- *⟨C, P_ε⟩ ist eine Näherung des Transportwerts, die mit ε → 0 besser wird.* Auch dann ist es kein Abstand (Selbstabstand ≠ 0); die Definition der „Sinkhorn-Distanz“ muss gewählt werden.

Und was die Demo für die weiteren Demos der Reihe stehen lässt:

- *Die Abstände werden an Verteilungen bewertet, die man genau kennt (Raster, bekannte Verschiebung), ohne Rauschen einer Stichprobe und ohne Schwelle.*
- *Die Verteilungen sind exakt gegeben (Raster, geschlossene Formeln); es gibt kein n und keine Dimension d > 1.* Experiment 3 zeigt Stichprobenrauschen nur an einem Beispiel und hebt diese Annahme nicht auf.
- Die Demo misst den Abstand; die zwei Wege von einer Verteilung zur anderen (die Mischung als Gerade der Totalvariation gegen die Verschiebung als Gerade der W2-Geometrie) zeigt sie nicht.

Weitere Grenzen:

- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit; eine Zahl über „die Familie“ ist die Zahl einer Verteilungsmenge. Gemessen ist nur eine Familie (Mischungen von 1 bis 3 Gauß-Hügeln, Breite 0,72 bis 2,4 Stunden); Kästen, Spitzen und schwere Ränder in Experiment 4 nicht.
- Die Zellenlage ist Teil der Konvention (Zellenmitten); Punktproben an den Zellgrenzen geben andere TV-Werte.
- Die Dreiecksungleichung wird für die Wurzel des Werts gezählt; eine Verletzung widerspricht keinem gelesenen Satz (Cuturis Satz gilt für die Nebenbedingungsform bei einer Metrik als Kosten, nicht für quadrierte Kosten), sie sagt, dass „Distanz“ für V-c und V-d keine selbstverständliche Vokabel ist.
- Die Kreisformel für W1 stammt aus dem Gedächtnis und ist an einem Linearprogramm geprüft, nicht aus einer gelesenen Quelle. KS ist auf dem Kreis nicht rotationsinvariant.
- V-a ist nur bei α = Median der KL-Werte von V-b gerechnet, nicht über eine Reihe von α.
- Gerechnet wird in float64 im Kern-Bereich; unter ε/D = 0,3 wird nicht angeboten (einzelne Paare mit fast getrennten Trägern brauchen dort sehr viele Durchläufe: gemessen bei ε/D = 0,1 mit der Menge 1, ein Kern, bei gleichzeitiger anderer Last des Rechners, rund eine Minute Rechenzeit und 1 Lauf bei V-b bis V-e sowie 21 Läufe der Bisektion von V-a ohne Konvergenz innerhalb von 200 000 Durchläufen; unter ε = 1/690 bei größtem Kostenwert 1 meldet die Demo den Unterlauf von K = exp(−C/ε) als Fehler, statt falsch zu rechnen).
- Literatur: Cuturi 2013 (arXiv-Fassung v1; die NeurIPS-Fassung kann abweichen), Genevay u. a. 2018, Feydy u. a. 2019, Arjovsky u. a. 2017 (Beispiel 1: parallele Linien) und Vershik 2005 sind im Wortlaut gelesen und so zitiert; Resin u. a. (arXiv 2408.09770 v3) ist ein Preprint, nur im Auszug gelesen und nicht nachgerechnet; Peyré, Cuturi (2019) ab Kapitel 4 nicht gelesen. TV, KL, KS, W1, die Quantilformel und die Zentrierung sind Lehrstoff oder Forschung; hier wird keine Neuheit behauptet.
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).

## Verifikation

- **Handwerte in Brüchen** (`tests/test_abstand.py`, `tests/test_sinkhorn.py`): Kasten für d = 0 bis 6 und für halbe Stunden (d = 0,5; 1,5; 2,5), Treppe (17/20, 6/500), Zwei-Spitzen-Zeilen und KL gegen Glättung, 2 × 2 geschlossen, 1/30, KL ohne Unterschied 789/1 024 (Aufzählung gegen Formel in 5 Instanzen).
- **Orakel** (`tests/test_oracle_verteilungsabstand.py`, `tests/test_sinkhorn.py`): W2² über Quantile gegen das Linearprogramm (HiGHS, Atome unter 1e-9 abgeschnitten, Massen mit 1e6 skaliert) auf 12 Paaren, W1 gegen das Linearprogramm und `scipy.stats.wasserstein_distance` auf 15 Zufallspaaren, Kreisformel gegen das Linearprogramm auf 20 Zufallspaaren (1e-8 mit skalierten Massen, 1e-6 ohne), Gauß: W2² = Lage² + Streuung² gegen die Quadratur der Quantile (5 Paare), Prognosetabelle gegen Quadratur, W1, W2², CRPS und E|X − y| gegen empirische Verteilungen gegen `scipy.integrate.quad` (je 3 Fälle), Sinkhorn-Stapel gegen das Dual per L-BFGS (5 Paare, 1e-8) und gegen die Log-Domänen-Iteration von „Sinkhorn: weich statt scharf“, V-a gegen SLSQP (6 Zufallsinstanzen mit 5 Zellen, α = 0,5 und erweitertem ε-Bereich von 0,002 bis 50, nicht in der Konfiguration der App; SLSQP liegt in diesen 6 Instanzen nie unter der Lagrange-Lösung, was aus der Konvexität des Problems folgt).
- **Eigenschaften** (`tests/test_eigenschaften.py`, je 10 bis 40 Zufallsinstanzen): Symmetrie, Selbstabstand 0, Pinsker-Ungleichung, gemeinsames Verschieben und Spiegeln ändert W1 nicht, W1 auf dem Kreis höchstens W1 auf der Geraden, W2 erfüllt die Dreiecksungleichung (0 von 2 024 in fünf Mengen), Kosten und ε mit demselben Faktor skalieren V-b bis V-e mit, V-c ≈ 2 · V-d bei kleinem ε.
- **Randfälle:** a = b, getrennte Träger, eine Zelle, ein Dirac, zwei Spitzen in derselben Zelle, ε zu klein (Fehlermeldung statt falscher Zahl), Läufe ohne Konvergenz werden gezählt.
- **Oberfläche:** AppTest (Standard, jedes Preset, jede Option der Auswahlregler an den Rändern, Permalink, alle Minima, alle Maxima), Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset, jede Reglerwirkung), Footer, Laufzeit-Importe.
- **Testzahl:** 277 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `vab_abstand.py` | Abstände und Divergenzen auf dem Raster |
| `vab_profile.py` | Tagesprofile, Verteilungsmengen, vektorisierter Zufall |
| `vab_sinkhorn.py` | Sinkhorn im Stapel, V-a bis V-e, Dreiecksverletzungen |
| `vab_evaluation.py` | Reihen der Experimente, Kennzahlen, Verdikte |
| `vab_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `vab_layout.py` | Fester Kopf (Stylesheet) |
| `vab_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `vab_rng.py` | Hausgenerator SplitMix64 (Kopie der Vorlage) |
| `vab_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `vab_vorgerechnet.py`, `tools/messreihen.py`, `tools/messreihen.json` | die Reihe „fünf Verteilungsmengen“ (vorgerechnet, Aufruf im Ordner der Demo) |
| `tests/` | Handwerte, Orakel, Eigenschaften, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- Realdaten (Bestellzeiten, Ankünfte), andere Familien von Verteilungen in Experiment 4, eine Reihe über α bei V-a, mehr als eine Dimension, der Rauschboden von W1 gegen n und Dimension.
- Eine Zwischenverteilung (Mittelung zweier Verteilungen) und das Erkennen einer Verteilungsänderung mit Schwelle: die Demos [Zwischen zwei Verteilungen – verschieben statt mischen](https://sebastianhanisch-profile-verschieben-demo.streamlit.app/) und [Verteilungsänderung erkennen – wer sieht was, und wann](https://sebastianhanisch-abstand-alarm-demo.streamlit.app/) der Reihe.
- PDF-Export.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die App ist für Python 3.14 gedacht (keine Pins in `requirements.txt`).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html).
