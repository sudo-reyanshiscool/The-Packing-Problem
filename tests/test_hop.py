import math

import numpy as np

from src.anneal import Config
from src.hop import STARTS, float_polish, kick, run_chain
from src.io_records import RECORDS, as_float_arrays, load_packing


def test_starts_have_right_shape_and_range():
    rng = np.random.default_rng(0)
    for kind, f in STARTS.items():
        x, y, t = f(rng, 17, 4.9)
        assert len(x) == len(y) == len(t) == 17, kind
        assert (x > 0).all() and (x < 4.9).all() and (t >= 0).all() and (t < math.pi / 2 + 1e-12).all()


def test_kick_changes_something():
    rng = np.random.default_rng(1)
    s, x, y, t = as_float_arrays(load_packing(RECORDS / "17.json"))
    kinds = set()
    for _ in range(30):
        xk, yk, tk, how = kick(rng, x, y, t, s)
        kinds.add(how)
        assert not (np.array_equal(xk, x) and np.array_equal(yk, y) and np.array_equal(tk, t))
    assert {"rerandomise", "retilt"} <= kinds


def test_float_polish_recovers_record_from_perturbation():
    s, x, y, t = as_float_arrays(load_packing(RECORDS / "10.json"))
    rng = np.random.default_rng(2)
    sp, *_ , viol = float_polish(x + 1e-4 * rng.standard_normal(10), y + 1e-4 * rng.standard_normal(10),
                                 t, s + 5e-4)
    assert abs(sp - (3 + 1 / math.sqrt(2))) < 1e-9 and viol < 1e-9


def test_run_chain_returns_basins():
    cfg = Config(n=5, s0=2.9, max_sweeps=40000)
    r = run_chain((cfg, 3, "random", 3, 2.9, 0.02, 2e-3, float("inf")))
    assert r["hops"] == 3 and r["best"] is not None
    assert min(r["basins"]) < 2.75
