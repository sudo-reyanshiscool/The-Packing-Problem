import copy

import mpmath
import pytest

from src.io_records import RECORDS, load_packing, make_packing, svg_to_packing
from src.verify import verify_packing

RECORD_NS = [5, 10, 11, 17]


def rec(n):
    return load_packing(RECORDS / f"{n}.json")


def shifted(p, i, dx="0", dy="0", dtheta="0"):
    q = copy.deepcopy(p)
    with mpmath.workdps(60):
        sq = q["squares"][i]
        sq["cx"] = mpmath.nstr(mpmath.mpf(sq["cx"]) + mpmath.mpf(dx), 60)
        sq["cy"] = mpmath.nstr(mpmath.mpf(sq["cy"]) + mpmath.mpf(dy), 60)
        sq["theta"] = mpmath.nstr(mpmath.mpf(sq["theta"]) + mpmath.mpf(dtheta), 60)
    return q


@pytest.mark.parametrize("n", RECORD_NS + [103, 105, 126])
def test_records_pass(n):
    r = verify_packing(rec(n))
    assert r.ok, r.summary()


@pytest.mark.parametrize("n", RECORD_NS)
def test_shrunk_container_fails(n):
    p = rec(n)
    with mpmath.workdps(60):
        p["s"] = mpmath.nstr(mpmath.mpf(p["s"]) - mpmath.mpf("1e-25"), 60)
    r = verify_packing(p)
    assert not r.ok and "wall" in r.worst_where


@pytest.mark.parametrize("n", RECORD_NS)
def test_all_shifted_right_fails(n):
    p = rec(n)
    for i in range(n):
        p = shifted(p, i, dx="1e-25")
    r = verify_packing(p)
    assert not r.ok and "right" in r.worst_wall_where


@pytest.mark.parametrize("n", RECORD_NS)
def test_perturbed_square_fails(n):
    # Try each square with a small random-direction nudge; a non-rattler must fail.
    base = rec(n)
    moves = [("1e-12", "0", "0"), ("-1e-12", "0", "0"), ("0", "1e-12", "0"),
             ("0", "-1e-12", "0"), ("0", "0", "1e-12"), ("0", "0", "-1e-12")]
    failures = sum(not verify_packing(shifted(base, i, *m)).ok
                   for i in range(n) for m in moves)
    assert failures > 0


def test_rigid_five_any_move_fails():
    # n = 5 is rigid: every nudge of the tilted centre square must overlap.
    # Rotation about a vertex contact gives second-order overlap (~eps^2 / 4),
    # so rotations use 1e-12 rad (depth ~2.5e-25) rather than 1e-20.
    base = rec(5)
    i = next(k for k, q in enumerate(base["squares"]) if mpmath.mpf(q["theta"]) != 0)
    for m in [("1e-20", "0", "0"), ("-1e-20", "0", "0"), ("0", "1e-20", "0"),
              ("0", "-1e-20", "0"), ("0", "0", "1e-12"), ("0", "0", "-1e-12")]:
        r = verify_packing(shifted(base, i, *m))
        assert not r.ok and r.worst_where.startswith("pair"), m


def test_overlap_depth_matches_vendor_on_30_digit_svg():
    # The records-page SVG for n = 5 stores s to 30 digits only; the vendor
    # checker reports overlap depth 6.857577121094917e-30. We must agree.
    p = svg_to_packing(RECORDS / "svg" / "square-5.svg")
    r = verify_packing(p)
    assert not r.ok
    assert abs(r.worst_pair - mpmath.mpf("6.857577121094917e-30")) < mpmath.mpf("1e-40")


def test_touching_squares_pass():
    p = make_packing(2, "2", [("0.5", "0.5", "0"), ("1.5", "0.5", "0")], "test")
    assert verify_packing(p).ok


def test_touching_below_tolerance_passes_above_fails():
    ok = make_packing(2, "2", [("0.5", "0.5", "0"), ("1.5", "0.5", "0")], "t")
    ok["squares"][1]["cx"] = "1.4999999999999999999999999999999"  # 1e-31 overlap
    assert verify_packing(ok).ok
    bad = copy.deepcopy(ok)
    bad["squares"][1]["cx"] = "1.49999999999999999999999999999"  # 1e-29 overlap
    r = verify_packing(bad)
    assert not r.ok and r.worst_where == "pair 0,1"


def test_rotated_overlap_detected():
    # Two 45 degree squares whose centres are 1.2 apart overlap (half-diagonal 0.707).
    q = str(mpmath.pi / 4)
    p = make_packing(2, "4", [("1", "2", q), ("2.2", "2", q)], "t")
    r = verify_packing(p)
    assert not r.ok
    # Both edge normals are at 45 degrees: depth = 1 - 1.2 / sqrt(2).
    with mpmath.workdps(50):
        assert abs(r.worst_pair - (1 - mpmath.mpf("1.2") / mpmath.sqrt(2))) < mpmath.mpf("1e-40")


def test_wrong_n_rejected():
    p = rec(5)
    p["n"] = 6
    with pytest.raises(ValueError):
        verify_packing(p)
