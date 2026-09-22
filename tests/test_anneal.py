import math

import mpmath
import numpy as np
import pytest

from src.anneal import Config, anneal_fixed_s, _probs, _seed, run_start, save_candidate
from src.geometry import max_violation, pair_pen, total_energy
from src.io_records import RECORDS, as_float_arrays, load_packing
from src.verify import _Sq, _proj, pair_depth


def _exact_mtd(a, b):
    """Minimum translation distance (what the float search uses), exact."""
    best = None
    for ax in (a.u, a.v, b.u, b.v):
        lo1, hi1 = _proj(ax, a.pts)
        lo2, hi2 = _proj(ax, b.pts)
        d = min(hi1 - lo2, hi2 - lo1)
        best = d if best is None or d < best else best
    return best


def test_pair_pen_matches_exact_sat():
    # Float search energy = SAT minimum translation distance. The verifier (like
    # the vendor checker) measures interval intersection instead; both are
    # positive exactly when the squares overlap.
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(3000):
        xi, yi, xj, yj = rng.random(4) * 2
        ti, tj = rng.random(2) * math.pi / 2
        p = pair_pen(xi, yi, math.cos(ti), math.sin(ti), xj, yj, math.cos(tj), math.sin(tj))
        with mpmath.workdps(30):
            a = _Sq(mpmath.mpf(xi), mpmath.mpf(yi), mpmath.mpf(ti))
            b = _Sq(mpmath.mpf(xj), mpmath.mpf(yj), mpmath.mpf(tj))
            d, m = pair_depth(a, b), _exact_mtd(a, b)
        assert abs(p - max(float(m), 0.0)) < 1e-12
        assert (p > 0) == (d > 0)
        checked += p > 0
    assert checked > 500


@pytest.mark.parametrize("n", [5, 10, 11, 17])
def test_records_have_zero_float_energy(n):
    s, x, y, t = as_float_arrays(load_packing(RECORDS / f"{n}.json"))
    assert total_energy(x, y, np.cos(t), np.sin(t), s) < 1e-12
    assert max_violation(x, y, t, s) < 1e-12
    # and a shrunk container does not
    assert total_energy(x, y, np.cos(t), np.sin(t), s - 1e-3) > 1e-4


def test_kernel_energy_bookkeeping():
    # Tracked energy must agree with a fresh recomputation after many moves.
    n, s = 12, 3.6
    rng = np.random.default_rng(1)
    _seed(1)
    x = 0.5 + rng.random(n) * (s - 1)
    y = 0.5 + rng.random(n) * (s - 1)
    t = rng.random(n) * math.pi / 2
    steps = np.array([0.2, 0.3, 0.0])
    stats = np.zeros(2)
    E = anneal_fixed_s(x, y, t, s, 50000, 1e-2, 1e-2, steps, _probs(Config(n=n, s0=s)), 0.5, -1.0, stats)
    assert abs(E - total_energy(x, y, np.cos(t), np.sin(t), s)) < 1e-12
    assert np.all((t >= 0) & (t < math.pi / 2))


def test_n4_finds_s2():
    r = run_start(Config(n=4, s0=2.3, max_sweeps=200000), seed=3)
    assert r["s"] is not None and 2.0 - 1e-9 < r["s"] < 2.0 + 1e-4
    assert r["max_pen"] < 1e-9


def test_n5_close_to_record():
    ss = [run_start(Config(n=5, s0=2.85, max_sweeps=400000), seed=k)["s"] for k in range(6)]
    best = min(v for v in ss if v is not None)
    assert abs(best - (2 + math.sqrt(2) / 2)) < 1e-3


def test_seed_from_record_stays_near_record():
    rec = float(load_packing(RECORDS / "10.json")["s"])
    r = run_start(Config(n=10, s0=0.0, seed_from=str(RECORDS / "10.json"), perturb=0.02,
                         max_sweeps=20000, shrink=1e-3), seed=0)
    assert r["s"] is not None and r["s"] < rec + 0.05


def test_candidate_roundtrip(tmp_path, monkeypatch):
    import src.anneal as A
    monkeypatch.setattr(A, "ROOT", tmp_path)
    r = run_start(Config(n=4, s0=2.3, max_sweeps=5000), seed=0)
    path = save_candidate(r, 4, "test")
    p = load_packing(path)
    assert p["n"] == 4 and not p["verified"] and abs(float(p["s"]) - r["s"]) < 1e-15
