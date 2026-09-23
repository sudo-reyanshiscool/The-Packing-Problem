import math

import mpmath
import numpy as np
import pytest

from src.io_records import RECORDS, as_float_arrays, load_packing, make_packing
from src.polish import (Con, Problem, active_set, choose_constraints, con_value, inflate,
                        newton_exact, polish_packing, rigidity, slsqp_polish)
from src.verify import verify_packing


def _z(packing):
    s, x, y, t = as_float_arrays(packing)
    return np.concatenate([x, y, t, [s]])


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    n = 6
    x, y = rng.random(n) * 3, rng.random(n) * 3
    t = rng.random(n) * math.pi / 2
    z = np.concatenate([x, y, t, [3.2]])
    cons = choose_constraints(x, y, t, close2=100.0)  # every pair
    prob = Problem(n, cons)
    g, J = prob.eval(z)
    h = 1e-6
    for col in range(3 * n + 1):
        zp, zm = z.copy(), z.copy()
        zp[col] += h
        zm[col] -= h
        fd = (prob.eval(zp)[0] - prob.eval(zm)[0]) / (2 * h)
        assert np.allclose(J[:, col], fd, atol=1e-8)


def test_scalar_and_vector_constraints_agree():
    rng = np.random.default_rng(1)
    n = 5
    x, y = rng.random(n) * 3, rng.random(n) * 3
    t = rng.random(n) * math.pi / 2
    z = np.concatenate([x, y, t, [3.0]])
    cons = choose_constraints(x, y, t, close2=100.0)
    g, J = Problem(n, cons).eval(z)
    for r, c in enumerate(cons):
        v, grad = con_value(c, x, y, t, 3.0)
        assert abs(v - g[r]) < 1e-14
        for col, gv in grad.items():
            assert abs(J[r, col] - gv) < 1e-14


def test_feasible_constraints_imply_no_overlap():
    # Any point satisfying the chosen separating-edge constraints is a packing.
    p = load_packing(RECORDS / "11.json")
    s, x, y, t = as_float_arrays(p)
    cons = choose_constraints(x, y, t)
    g, _ = Problem(11, cons).eval(_z(p))
    assert g.min() > -1e-12


def test_active_set_of_record_5():
    # Rigid, but not first order: the tilted square touches the corner squares at
    # its edge midpoints, so its rotation is blocked only at second order.
    p = load_packing(RECORDS / "5.json")
    z = _z(p)
    cons = choose_constraints(*as_float_arrays(p)[1:])
    act = active_set(z, cons, 1e-8)
    flex, rattlers = rigidity(z, act)
    assert flex == 1 and rattlers == []


@pytest.mark.parametrize("n", [10, 11])
def test_polish_perturbed_record_recovers_it(n):
    p = load_packing(RECORDS / f"{n}.json")
    s, x, y, t = as_float_arrays(p)
    rng = np.random.default_rng(n)
    q = make_packing(n, s * 1.002, zip(x + 1e-3 * rng.standard_normal(n) + 0.001 * s,
                                    y + 1e-3 * rng.standard_normal(n) + 0.001 * s,
                                    t + 1e-3 * rng.standard_normal(n)), "test")
    out, info = polish_packing(q, log=lambda m: None)
    assert info["ok"]
    with mpmath.workdps(50):
        assert abs(mpmath.mpf(out["s"]) - mpmath.mpf(p["s"])) < mpmath.mpf("1e-30")


def test_polish_reaches_closed_form_for_n10():
    p = load_packing(RECORDS / "10.json")
    s, x, y, t = as_float_arrays(p)
    q = make_packing(10, s + 1e-4, zip(x + 5e-5, y + 5e-5, t), "test")
    out, info = polish_packing(q, log=lambda m: None)
    assert info["ok"]
    with mpmath.workdps(50):
        assert abs(mpmath.mpf(out["s"]) - (3 + 1 / mpmath.sqrt(2))) < mpmath.mpf("1e-30")
    assert 3 in info["rattlers"] or len(info["rattlers"]) >= 1


def test_inflate_repairs_violation():
    p = load_packing(RECORDS / "5.json")
    with mpmath.workdps(60):
        p["s"] = mpmath.nstr(mpmath.mpf(p["s"]) - mpmath.mpf("1e-20"), 60)
    r = verify_packing(p)
    assert not r.ok
    q = inflate(p, 4 * float(r.worst))
    assert verify_packing(q).ok


def test_newton_on_two_touching_squares():
    # Two unit squares side by side in a 2 x 1 strip, slightly apart: Newton closes the gap.
    n = 2
    z = np.array([0.5, 1.5 + 1e-6, 0.5, 0.5, 0.0, 0.0, 2.0 + 2e-6])
    cons = [Con(1, 0, 0, 0, q) for q in range(4)] + [Con(1, 1, 1, 1, q) for q in range(4)]
    cons += [Con(0, 0, 1, 0, q) for q in range(4)]
    act = active_set(z, cons, 1e-3)
    assert len(act) == 6
    zm, resid = newton_exact(z, act, log=lambda m: None)
    assert resid < mpmath.mpf("1e-40")
    assert abs(zm[-1] - 2) < mpmath.mpf("1e-40")
