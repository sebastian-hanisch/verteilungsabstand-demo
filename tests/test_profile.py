"""Profile, Verteilungsmengen und Zufall: Form, Normierung, Festschreibung der Standardmenge, vektorisierter Zufall gleich dem Hausgenerator."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import vab_constants as C
import vab_evaluation as E
import vab_profile as PR
from vab_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
PRUEFSUMME_MENGE1 = "303ccccd8b59ce8c"          # Standardmenge: 24 Profile auf 48 Zellen, Menge 1; ein Test, der die Menge festschreibt


def test_kasten_hat_die_summe_1_und_liegt_von_6_bis_9_uhr():
    p = PR.kasten(24, 6.0, 3.0, 0.0)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(p[6:9], 1 / 3) and p[:6].sum() == 0 and p[9:].sum() == 0


def test_kasten_verschoben_um_einen_halben_tag_zerlegt_die_masse_auf_zwei_zellen():
    p = PR.kasten(24, 6.0, 3.0, 0.5)
    assert p[6] == pytest.approx(1 / 6) and p[7] == pytest.approx(1 / 3) and p[8] == pytest.approx(1 / 3) and p[9] == pytest.approx(1 / 6)


def test_kasten_laeuft_auf_dem_kreis_ueber_mitternacht_und_wird_auf_der_geraden_abgeschnitten():
    kreis = PR.kasten(24, 6.0, 3.0, 17.0, kreis=True)             # 23 bis 26 Uhr = 23 bis 2 Uhr
    assert kreis[23] == pytest.approx(1 / 3) and kreis[0] == pytest.approx(1 / 3) and kreis[1] == pytest.approx(1 / 3)
    gerade = PR.kasten(24, 6.0, 3.0, 17.0, kreis=False)           # nur die Zelle 23 liegt noch im Tag, neu normiert
    assert gerade[23] == pytest.approx(1.0) and gerade.sum() == pytest.approx(1.0)


def test_zwei_spitzen_haben_die_summe_1_und_sind_symmetrisch_um_ihre_zellen():
    p = PR.zwei_spitzen(24, 0.0, 1.0)
    assert p.sum() == pytest.approx(1.0, abs=1e-15) and p.argmax() == 14 and p[6] > p[5] and p[6] > p[7]
    assert p[6] / p[14] == pytest.approx(0.4 / 0.6, rel=1e-6)


def test_zwei_spitzen_gleichen_den_punktproben_an_ganzen_stunden():
    """Handbeispiel 3: Normalverteilungen an den Punkten 0 bis 23 mit Spitzen bei 6 und 14 sind unsere Spitzen bei den Zellenmitten 6,5 und 14,5."""
    stunden = np.arange(24, dtype=float)
    h = 0.4 * np.exp(-0.5 * (stunden - 6) ** 2) + 0.6 * np.exp(-0.5 * (stunden - 14) ** 2)
    assert np.allclose(PR.zwei_spitzen(24, 0.0, 1.0), h / h.sum(), atol=1e-15, rtol=0)


def test_profil_menge_ist_positiv_normiert_und_reproduzierbar():
    p = PR.profil_menge(1, 24, 48, C.ZWECK_MENGE)
    assert p.shape == (24, 48) and np.all(p > 0) and np.allclose(p.sum(axis=1), 1.0, atol=1e-14)
    assert np.array_equal(p, PR.profil_menge(1, 24, 48, C.ZWECK_MENGE)) and not np.array_equal(p, PR.profil_menge(2, 24, 48, C.ZWECK_MENGE))


def test_die_standardmenge_ist_festgeschrieben():
    prof = PR.profil_menge(C.DEFAULTS["menge4"], C.ANZAHL4, C.ZELLEN4, C.ZWECK_MENGE)
    assert E.pruefsumme_menge(prof) == PRUEFSUMME_MENGE1


def test_zweiter_prozess_mit_anderem_hashseed_gibt_dieselbe_menge():
    code = "import sys; sys.path.insert(0, sys.argv[1]); import vab_profile as P, vab_evaluation as E; print(E.pruefsumme_menge(P.profil_menge(1, 24, 48, 1)))"
    for hashseed in ("1", "2"):
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(ROOT)], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": hashseed}, timeout=120)
        assert r.stdout.strip() == PRUEFSUMME_MENGE1, r.stderr


def test_vektorisierter_zufall_gleicht_dem_hausgenerator_bitgleich():
    u = PR.zufall_gleich(7, 2, 3, 20)
    for j in range(3):
        g = PR.instanz_rng(7, 2, j)
        assert [g.random() for _ in range(20)] == u[j].tolist()


def test_vektorisierte_normalzahlen_gleichen_dem_hausgenerator_auf_1e_12():
    z = PR.zufall_normal(7, 2, 2, 10)
    for j in range(2):
        g = PR.instanz_rng(7, 2, j)
        assert np.allclose([g.normal() for _ in range(10)], z[j], rtol=1e-12, atol=1e-12)


def test_normalzahlen_haben_mittel_0_und_streuung_1():
    z = PR.zufall_normal(11, 3, 400, 400).ravel()
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01


def test_zellenmitten():
    assert np.allclose(PR.zellenmitten(24), np.arange(24) + 0.5) and PR.zellenmitten(48)[0] == 0.25


def test_rng_ist_kopie_der_vorlage():
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (CI)")
    assert (ROOT / "vab_rng.py").read_bytes() == vorlage.read_bytes()
    assert SplitMix64(1).next_u64() == SplitMix64(1).next_u64()
