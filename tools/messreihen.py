"""Rechnet die Reihe „dieselbe Zelle in fünf Verteilungsmengen“ vor und schreibt `tools/messreihen.json` (siehe vab_vorgerechnet.py).

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/messreihen.py
Je ε/D in (8, 3, 1, 0,3) und je Verteilungsmenge 1 bis 5: die Kennzahlen der Verwendungen V-a bis V-e (Selbstabstand, „selbst nicht am nächsten“, Kendall τ, nächster Nachbar,
Median Wert / W2², Dreiecksverletzungen), dazu D der Mengen 1 bis 20. Dauer etwa 1 bis 2 Minuten; `OMP_NUM_THREADS=1` setzen.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import vab_evaluation as E  # noqa: E402
from vab_vorgerechnet import DATEI  # noqa: E402

MENGEN = (1, 2, 3, 4, 5)
C_WERTE = (8.0, 3.0, 1.0, 0.3)


def main():
    daten = {"streuung": {}, "d_mengen": {}}
    for m in range(1, 21):
        daten["d_mengen"][str(m)] = float(np.median(E.menge4(m)[3]))
    for c in C_WERTE:
        t = time.time()
        eintrag = {"mengen": list(MENGEN), "daten": {}}
        for m in MENGEN:
            res = E.analyse4(m, c)
            eintrag["daten"][str(m)] = {v: {k: float(x) if isinstance(x, float) else int(x) for k, x in res["kennzahlen"][v].items()} for v in res["kennzahlen"]}
            eintrag["daten"][str(m)]["d"] = res["d"]
            eintrag["daten"][str(m)]["nicht_konvergiert"] = int(res["nicht_konvergiert"])
            eintrag["daten"][str(m)]["nicht_konvergiert_a"] = int(res["nicht_konvergiert_a"])
            eintrag["daten"][str(m)]["unwirksam_a"] = int(res["unwirksam_a"])
        daten["streuung"][repr(c)] = eintrag
        print(f"eps/D = {c}: {time.time() - t:.1f} s", flush=True)
    DATEI.write_text(json.dumps(daten, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"geschrieben: {DATEI} ({DATEI.stat().st_size} Byte)")


if __name__ == "__main__":
    main()
