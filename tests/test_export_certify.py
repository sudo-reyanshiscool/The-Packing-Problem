import copy
import subprocess
import sys

import mpmath
import pytest

from src.certify import certify
from src.export import export, roundtrip_error
from src.io_records import RECORDS, ROOT, VENDOR, load_packing


@pytest.mark.parametrize("n", [5, 11, 17])
def test_export_roundtrip_through_vendor_parser(n, tmp_path):
    p = load_packing(RECORDS / f"{n}.json")
    svg, js = export(p, tmp_path / f"n{n}")
    assert js.exists() and svg.exists()
    assert mpmath.mpf(roundtrip_error(p, svg)) < mpmath.mpf("1e-44")
    assert load_packing(js)["s"] == p["s"]


def test_export_passes_vendor_checker(tmp_path):
    # parse_svg_packing.py writes output.txt in the cwd; check_packing.py reads it.
    # (records/17.json is stored to 34 digits only, so it would fail the checker's
    # tolerance of 1e-47; n = 10 has closed-form s and exact coordinates.)
    p = load_packing(RECORDS / "10.json")
    svg, _ = export(p, tmp_path / "n17")
    py = sys.executable
    subprocess.run([py, str(VENDOR / "parse_svg_packing.py"), str(svg), "50"],
                   cwd=tmp_path, capture_output=True, check=True)
    r = subprocess.run([py, str(VENDOR / "check_packing.py"), "output.txt"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert "VALID" in r.stdout.splitlines()[-1] and "INVALID" not in r.stdout


@pytest.mark.parametrize("n", [5, 10])
def test_certify_records(n):
    # (17 is stored to 34 digits: its contacts overlap by 1e-33 and cannot be certified.)
    ok, s_cert = certify(load_packing(RECORDS / f"{n}.json"))
    assert ok
    with mpmath.workdps(60):
        rec = mpmath.mpf(load_packing(RECORDS / f"{n}.json")["s"])
        assert 0 < mpmath.mpf(s_cert) - rec < rec * mpmath.mpf("2e-40")


def test_certify_rejects_overlap():
    p = load_packing(RECORDS / "5.json")
    q = copy.deepcopy(p)
    with mpmath.workdps(60):
        q["s"] = mpmath.nstr(mpmath.mpf(p["s"]) - mpmath.mpf("1e-30"), 60)
    ok, msg = certify(q)
    assert not ok and "container" in msg
    q = copy.deepcopy(p)
    with mpmath.workdps(60):
        q["squares"][0]["cx"] = mpmath.nstr(mpmath.mpf(q["squares"][0]["cx"]) + mpmath.mpf("1e-20"), 60)
    ok, msg = certify(q)
    assert not ok


def test_certify_needs_inflation_for_touching():
    # With eps = 0 an exact contact cannot be proved separated.
    ok, msg = certify(load_packing(RECORDS / "5.json"), eps="0")
    assert not ok
