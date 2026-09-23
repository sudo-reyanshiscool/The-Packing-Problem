"""Contact-constrained polish: float64 candidate -> exact (50-digit) packing.

Stage A (float64, scipy SLSQP). Variables z = (x, y, theta for each square, s).
Minimise s subject to smooth non-overlap constraints:

  * for each close pair, one separating edge is chosen (the edge of either
    square whose outward half-plane best separates the other square); the four
    corners of the other square must lie beyond that edge line (4 constraints);
  * every corner of every square must lie inside [0, s]^2 (16 per square).

Corner-beyond-edge constraints are smooth in all variables (no |sin| kinks at
parallel contacts) and any feasible point is a genuine packing for the included
pairs. The separating edges are re-chosen and the problem re-solved until the
choice is stable and the full SAT check (geometry.max_violation) is clean.

Stage B (mpmath, 60 digits). The active constraints (gap below --act-tol) are
solved as equations g_A(z) = 0 by Gauss-Newton with an exact (SVD) pseudo-
inverse; active sets are rank deficient as a rule (flexes, rattlers, dependent
contacts), and a float64 pseudo-inverse stalls at the square of the float
error (1e-32). Above 160 unknowns the float64 pseudo-inverse is used anyway
(mpmath SVD cost) and the inflation step below covers the remainder. If the
exact verifier finds a violation d, the packing is scaled by 1 + 4 d about
its centre, which restores every gap (pair gaps grow by at least 4 d, wall
gaps by at least 2 d).

The result is a packing at which the identified contacts are exactly closed.
Flexes are only solved to first order, so s can exceed the true local
minimum by second-order amounts (n = 10: 3e-34).

Output: verify.py result, rigidity report (first-order flexes and rattlers),
and, on PASS, a JSON in verified/. A verified s below the record triggers the
record protocol of CLAUDE.md (RECORD_CANDIDATE files + fresh-process re-check).

CLI:
  python -m src.polish candidates/<file>.json [...] [--act-tol 1e-8] [--quiet]
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mpmath
import numpy as np
from mpmath import mpf
from scipy.optimize import minimize

from .geometry import max_violation
from .io_records import ROOT, as_float_arrays, load_packing, make_packing, record_s, save_packing
from .export import write_svg
from .verify import verify_packing

HALF = 0.5
CORNERS = [(-HALF, -HALF), (-HALF, HALF), (HALF, -HALF), (HALF, HALF)]  # (a, b) offsets
CLOSE2 = 2.0 + 0.6  # pairs with centre distance^2 below this get a constraint


# ------------------------------------------------------------- constraints
# A constraint is (kind, owner, other, k, corner):
#   kind 0: corner `corner` of square `other` beyond edge k of square `owner`
#           (k: 0 = +u, 1 = +v, 2 = -u, 3 = -v, u = (cos t, sin t), v = (-sin t, cos t))
#   kind 1: corner `corner` of square `other` inside wall k (0 left, 1 right, 2 bottom, 3 top)

@dataclass(frozen=True)
class Con:
    kind: int
    owner: int
    other: int
    k: int
    corner: int

    def label(self):
        if self.kind == 0:
            return f"pair {self.owner},{self.other} edge {self.k} corner {self.corner}"
        return f"wall {('left', 'right', 'bottom', 'top')[self.k]} square {self.other} corner {self.corner}"


def split(z, n):
    return z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n]


def _normal(c, s, k):
    """Edge normal k and its derivative with respect to theta."""
    if k == 0:
        return (c, s), (-s, c)
    if k == 1:
        return (-s, c), (-c, -s)
    if k == 2:
        return (-c, -s), (s, -c)
    return (s, -c), (c, s)


def con_value(con: Con, x, y, t, s, cos=math.cos, sin=math.sin):
    """Constraint value (>= 0 feasible) and gradient as {index: value}; scalar, any arithmetic."""
    n = len(x)
    j = con.other
    a, b = CORNERS[con.corner]
    cj, sj = cos(t[j]), sin(t[j])
    px = x[j] + a * cj - b * sj
    py = y[j] + a * sj + b * cj
    dpx = -a * sj - b * cj   # d corner / d theta_j
    dpy = a * cj - b * sj
    g = {}
    if con.kind == 1:
        k = con.k
        if k == 0:
            val = px
            g[j], g[n + j], g[2 * n + j] = 1, 0, dpx
        elif k == 1:
            val = s - px
            g[j], g[n + j], g[2 * n + j], g[3 * n] = -1, 0, -dpx, 1
        elif k == 2:
            val = py
            g[j], g[n + j], g[2 * n + j] = 0, 1, dpy
        else:
            val = s - py
            g[j], g[n + j], g[2 * n + j], g[3 * n] = 0, -1, -dpy, 1
        return val, g
    i = con.owner
    ci, si = cos(t[i]), sin(t[i])
    (nx, ny), (dnx, dny) = _normal(ci, si, con.k)
    rx, ry = px - x[i], py - y[i]
    val = rx * nx + ry * ny - HALF
    g[j] = nx
    g[n + j] = ny
    g[2 * n + j] = dpx * nx + dpy * ny
    g[i] = -nx
    g[n + i] = -ny
    g[2 * n + i] = rx * dnx + ry * dny
    return val, g


def _corners_np(x, y, t):
    """Corners of all squares: arrays (n, 4) for x and y, plus d/dtheta."""
    c, s = np.cos(t), np.sin(t)
    A = np.array([a for a, b in CORNERS])
    B = np.array([b for a, b in CORNERS])
    px = x[:, None] + A[None, :] * c[:, None] - B[None, :] * s[:, None]
    py = y[:, None] + A[None, :] * s[:, None] + B[None, :] * c[:, None]
    dpx = -A[None, :] * s[:, None] - B[None, :] * c[:, None]
    dpy = A[None, :] * c[:, None] - B[None, :] * s[:, None]
    return px, py, dpx, dpy


class Problem:
    """Vectorised constraint set for SLSQP."""

    def __init__(self, n, cons):
        self.n = n
        self.cons = cons
        self.m = len(cons)
        self.kind = np.array([c.kind for c in cons])
        self.own = np.array([c.owner for c in cons])
        self.oth = np.array([c.other for c in cons])
        self.k = np.array([c.k for c in cons])
        self.cr = np.array([c.corner for c in cons])

    def eval(self, z):
        n, m = self.n, self.m
        x, y, t, s = split(z, n)
        px, py, dpx, dpy = _corners_np(x, y, t)
        c, sn = np.cos(t), np.sin(t)
        val = np.empty(m)
        J = np.zeros((m, 3 * n + 1))
        rows = np.arange(m)
        j, cr = self.oth, self.cr
        qx, qy, dqx, dqy = px[j, cr], py[j, cr], dpx[j, cr], dpy[j, cr]

        w = self.kind == 1
        if w.any():
            k = self.k[w]
            r, jj = rows[w], j[w]
            sign = np.where((k == 1) | (k == 3), -1.0, 1.0)
            horiz = (k == 0) | (k == 1)
            val[w] = np.where(horiz, qx[w], qy[w]) * sign + np.where(sign < 0, s, 0.0)
            J[r, jj] = np.where(horiz, sign, 0.0)
            J[r, n + jj] = np.where(horiz, 0.0, sign)
            J[r, 2 * n + jj] = np.where(horiz, dqx[w], dqy[w]) * sign
            J[r[sign < 0], 3 * n] = 1.0

        p = ~w
        if p.any():
            i, jj, k, r = self.own[p], j[p], self.k[p], rows[p]
            ci, si = c[i], sn[i]
            nx = np.select([k == 0, k == 1, k == 2], [ci, -si, -ci], si)
            ny = np.select([k == 0, k == 1, k == 2], [si, ci, -si], -ci)
            dnx = np.select([k == 0, k == 1, k == 2], [-si, -ci, si], ci)
            dny = np.select([k == 0, k == 1, k == 2], [ci, -si, -ci], si)
            rx, ry = qx[p] - x[i], qy[p] - y[i]
            val[p] = rx * nx + ry * ny - HALF
            J[r, jj] = nx
            J[r, n + jj] = ny
            J[r, 2 * n + jj] = dqx[p] * nx + dqy[p] * ny
            J[r, i] = -nx
            J[r, n + i] = -ny
            J[r, 2 * n + i] = rx * dnx + ry * dny
        return val, J


def choose_constraints(x, y, t, close2=CLOSE2):
    """Wall constraints for every square, and the best separating edge for each close pair."""
    n = len(x)
    cons = [Con(1, j, j, k, q) for j in range(n) for k in range(4) for q in range(4)]
    px, py, _, _ = _corners_np(x, y, t)
    c, sn = np.cos(t), np.sin(t)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[j] - x[i], y[j] - y[i]
            if dx * dx + dy * dy >= close2:
                continue
            best, arg = -np.inf, None
            for owner, other in ((i, j), (j, i)):
                for k in range(4):
                    (nx, ny), _ = _normal(c[owner], sn[owner], k)
                    gap = np.min((px[other] - x[owner]) * nx + (py[other] - y[owner]) * ny) - HALF
                    if gap > best:
                        best, arg = gap, (owner, other, k)
            owner, other, k = arg
            cons += [Con(0, owner, other, k, q) for q in range(4)]
    return cons


def _edge_key(cons):
    return {(c.owner, c.other, c.k) for c in cons if c.kind == 0}


# ------------------------------------------------------------ stage A: SLSQP

def slsqp_polish(x, y, t, s, rounds=12, maxiter=1000, ftol=1e-16, log=print):
    """Minimise s over separating-edge choices until stable. Returns z (float64) and constraints."""
    n = len(x)
    z = np.concatenate([x, y, t, [s]])
    cons = choose_constraints(x, y, t)
    prev = None
    for r in range(rounds):
        prob = Problem(n, cons)
        cache = {}

        def f_con(z, prob=prob, cache=cache):
            key = z.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = prob.eval(z)
            return cache[key][0]

        def j_con(z, prob=prob, cache=cache):
            f_con(z)
            return cache[z.tobytes()][1]

        obj = np.zeros(3 * n + 1)
        obj[-1] = 1.0
        res = minimize(lambda z: z[-1], z, jac=lambda z: obj, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": f_con, "jac": j_con}],
                       options={"maxiter": maxiter, "ftol": ftol})
        z_new = res.x
        x, y, t, s = split(z_new, n)
        viol = max_violation(x, y, t % (math.pi / 2), s)
        g = prob.eval(z_new)[0]
        log(f"  slsqp round {r + 1}: s={s:.12f} iters={res.nit} min_g={g.min():.2e} "
            f"sat_viol={viol:.2e} cons={len(cons)} ({res.message})")
        z = z_new
        cons_new = choose_constraints(x, y, t)
        same = _edge_key(cons_new) == _edge_key(cons)
        cons = cons_new
        if viol < 1e-10 and g.min() > -1e-10 and prev is not None and abs(prev - s) < 1e-12:
            break  # no progress: the SLP stage handles any remaining edge re-choice
        prev = s
    return z, cons


# HiGHS defaults to a 1e-7 feasibility tolerance, far too loose here: use its
# minimum and scale the constraint rows so the effective tolerance is ~1e-13.
LP_OPTS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
           "time_limit": 120.0}  # a degenerate simplex can otherwise run for hours (seen at n = 304)
LP_SCALE = 1e3


def _restore(z, iters=4, window=1e-4, tol=1e-13):
    """Second-order correction after an LP step: the smallest (L-infinity) pose
    change, with s fixed, that makes every constraint within `window` of active
    feasible again at the new linearisation point. An LP: min r subject to
    J dz >= -g, |dz| <= r, ds = 0. Iterated because the constraints are nonlinear."""
    from scipy.optimize import linprog
    n = (len(z) - 1) // 3
    N = 3 * n + 1
    for _ in range(iters):
        cons = choose_constraints(*split(z, n)[:3])
        g, J = Problem(n, cons).eval(z)
        if g.min() > -tol:
            break
        rows = g < window
        A = J[rows]
        m = A.shape[0]
        # variables: dz (N), r (1)
        A_ub = np.zeros((m + 2 * N, N + 1))
        b_ub = np.zeros(m + 2 * N)
        A_ub[:m, :N] = -A
        b_ub[:m] = g[rows]
        A_ub[m:m + N, :N] = np.eye(N)          # dz - r <= 0
        A_ub[m:m + N, N] = -1.0
        A_ub[m + N:, :N] = -np.eye(N)          # -dz - r <= 0
        A_ub[m + N:, N] = -1.0
        obj = np.zeros(N + 1)
        obj[N] = 1.0
        bounds = [(None, None)] * N + [(0, None)]
        bounds[N - 1] = (0, 0)                  # s fixed
        res = linprog(obj, A_ub=A_ub * LP_SCALE, b_ub=b_ub * LP_SCALE, bounds=bounds,
                      method="highs", options=LP_OPTS)
        if res.status != 0:
            break
        z = z + res.x[:N]
    return z


def slp_polish(z, cons, rounds=400, radius=1e-3, log=print):
    """Sequential linear programming: minimise ds subject to J dz >= -g, |dz| <= radius.

    Robust where SLSQP stalls on degenerate active sets (large n). Each LP
    step is followed by a second-order correction (_restore); a step is
    accepted if the corrected point is feasible to 1e-11 and s decreased.
    The trust radius doubles after a step that used it, halves after a
    rejection, and widens when the LP is infeasible inside it.
    """
    from scipy.optimize import linprog
    n = (len(z) - 1) // 3
    N = 3 * n + 1
    obj = np.zeros(N)
    obj[-1] = 1.0
    g, J = Problem(n, cons).eval(z)
    s_start = z[-1]
    accepted = rejected = 0
    for r in range(rounds):
        res = linprog(obj, A_ub=-J * LP_SCALE, b_ub=g * LP_SCALE, bounds=[(-radius, radius)] * N,
                      method="highs", options=LP_OPTS)
        if res.status == 2:  # infeasible within the trust region: widen it
            radius *= 4.0
            rejected += 1
            if radius > 1.0:
                log("  slp: LP infeasible; stopping")
                break
            continue
        if res.status != 0:
            log(f"  slp: LP status {res.status} ({res.message}); treating as a rejected step")
            radius *= 0.5
            rejected += 1
            if radius < 1e-10:
                break
            continue
        dz = res.x
        if -dz[-1] < 1e-15:
            break
        z_new = _restore(z + dz)
        cons_new = choose_constraints(*split(z_new, n)[:3])
        g_new, J_new = Problem(n, cons_new).eval(z_new)
        viol_new = max(0.0, -g_new.min())
        if viol_new > 1e-11 or z_new[-1] >= z[-1]:
            radius *= 0.5
            rejected += 1
            if radius < 1e-10:
                break
            continue
        z, cons, g, J = z_new, cons_new, g_new, J_new
        accepted += 1
        if abs(dz).max() > 0.9 * radius:
            radius = min(radius * 2.0, 1e-2)
    x, y, t, s = split(z, n)
    log(f"  slp: s={s:.12f} ({s - s_start:+.3e}) steps accepted={accepted} rejected={rejected} "
        f"min_g={g.min():.2e} sat_viol={max_violation(x, y, t % (math.pi / 2), s):.2e} radius={radius:.1e}")
    return z, cons


# ------------------------------------------------------ stage B: exact Newton

def active_set(z, cons, act_tol):
    n = (len(z) - 1) // 3
    x, y, t, s = split(z, n)
    g, _ = Problem(n, cons).eval(z)
    return [c for c, v in zip(cons, g) if v < act_tol]


def newton_exact(z, active, dps=60, iters=8, exact_max=160, log=print):
    """Chord Newton on g_A(z) = 0 in mpmath. Returns (z_mp list, residual)."""
    n = (len(z) - 1) // 3
    N = 3 * n + 1
    with mpmath.workdps(dps):
        zm = [mpf(v) for v in z]
        m = len(active)
        resid = None
        for it in range(iters):
            x, y, t, s = zm[:n], zm[n:2 * n], zm[2 * n:3 * n], zm[3 * n]
            vals = []
            for c in active:
                v, _ = con_value(c, x, y, t, s, mpmath.cos, mpmath.sin)
                vals.append(v)
            resid = max(abs(v) for v in vals) if vals else mpf(0)
            log(f"  newton {it}: max |g_A| = {mpmath.nstr(resid, 3)}")
            if resid < mpf(10) ** (-(dps - 12)) or m == 0:
                break
            gm = mpmath.matrix(vals)
            if N <= exact_max:
                # exact pseudo-inverse (rank-deficient active sets are the norm:
                # a float64 pinv stalls at the square of the float error)
                Jm = mpmath.matrix(m, N)
                for r, c in enumerate(active):
                    _, g = con_value(c, x, y, t, s, mpmath.cos, mpmath.sin)
                    for col, v in g.items():
                        Jm[r, col] = v
                U, S, V = mpmath.svd_r(Jm, full_matrices=False)
                cut = S[0] * mpf(10) ** (-(dps // 2))
                Ut_g = U.T * gm
                w = mpmath.matrix([Ut_g[q] / S[q] if S[q] > cut else mpf(0) for q in range(len(S))])
                step = V.T * w
            else:
                # large n: float64 pseudo-inverse (residual bottoms out near 1e-32; inflation covers it)
                zf = np.array([float(v) for v in zm])
                _, J = Problem(n, active).eval(zf)
                step = mpmath.matrix(np.linalg.pinv(J, rcond=1e-9).tolist()) * gm
            zm = [zm[i] - step[i] for i in range(N)]
        return zm, resid


def rigidity(z, active):
    """First-order rigidity from the active constraint Jacobian (s fixed).

    Returns (flex_count, rattlers): flexes are nullspace directions of the pose
    Jacobian; rattlers are squares with no active contact at all.
    """
    n = (len(z) - 1) // 3
    if not active:
        return 3 * n, list(range(n))
    _, J = Problem(n, active).eval(np.asarray(z, dtype=float))
    Jp = J[:, :3 * n]
    sv = np.linalg.svd(Jp, compute_uv=False)
    rank = int((sv > 1e-9 * max(sv[0], 1.0)).sum())
    touched = {c.other for c in active} | {c.owner for c in active if c.kind == 0}
    rattlers = [i for i in range(n) if i not in touched]
    return 3 * n - rank, rattlers


def inflate(packing: dict, eps):
    """Scale the packing about its centre by 1 + eps (restores gaps of size <= eps / 2)."""
    with mpmath.workdps(60):
        s = mpf(packing["s"])
        f = 1 + mpf(eps)
        s2 = s * f
        sq = [((mpf(q["cx"]) - s / 2) * f + s2 / 2, (mpf(q["cy"]) - s / 2) * f + s2 / 2, q["theta"])
              for q in packing["squares"]]
        return make_packing(packing["n"], s2, sq, packing["source"] + f" inflated {mpmath.nstr(mpf(eps), 3)}")


# ---------------------------------------------------------------- pipeline

def polish_packing(packing: dict, act_tol=1e-8, dps=60, skip_slsqp=False, log=print) -> tuple[dict, dict]:
    """Full polish of a float candidate. Returns (exact packing dict, info dict)."""
    n = packing["n"]
    s0, x, y, t = as_float_arrays(packing)
    log(f"polish n={n} s0={s0:.12f} float64 violation {max_violation(x, y, t, s0):.2e}")
    t_a = time.time()
    if skip_slsqp:  # SLSQP costs 30 min per round at n = 300 and rarely progresses there
        z, cons = np.concatenate([x, y, t, [s0]]), choose_constraints(x, y, t)
    else:
        z, cons = slsqp_polish(x, y, t, s0, log=log)
    z, cons = slp_polish(z, cons, log=log)
    log(f"  stage A {time.time() - t_a:.1f}s")
    active = active_set(z, cons, act_tol)
    pairs = {(c.owner, c.other) for c in active if c.kind == 0}
    walls = {(c.other, c.k) for c in active if c.kind == 1}
    log(f"  active: {len(active)} constraints ({len(pairs)} pair contacts, {len(walls)} wall contacts)")
    zm, resid = newton_exact(z, active, dps=dps, log=log)
    src = f"polish of {packing.get('source', '?')}"

    def finish(vals, tag):
        with mpmath.workdps(dps):
            sq = [(vals[i], vals[n + i], vals[2 * n + i]) for i in range(n)]
            p = make_packing(n, vals[3 * n], sq, src)
        r = verify_packing(p)
        log(f"  verify {tag}: {r.summary()}")
        if not r.ok:
            eps = 4 * float(r.worst)
            log(f"  inflating {tag} by {eps:.3e}")
            p = inflate(p, eps)
            r = verify_packing(p)
            log(f"  verify {tag}: {r.summary()}")
        return p, r

    # Take the smaller verified s of the exact solution and the (inflated) float point:
    # with a float64 pseudo-inverse a degenerate active set can send Newton astray.
    out, r = finish(zm, "exact")
    if not r.ok or float(out["s"]) > z[-1] + 1e-12:
        out2, r2 = finish([mpf(v) for v in z], "float")
        if r2.ok and (not r.ok or mpf(out2["s"]) < mpf(out["s"])):
            out, r = out2, r2
    flex, rattlers = rigidity(z, active)
    out["verified"] = bool(r.ok)
    out["contacts"] = {"pairs": sorted(f"{i},{j}" for i, j in pairs),
                       "walls": sorted(f"{i} {('left', 'right', 'bottom', 'top')[k]}" for i, k in walls)}
    out["rigidity"] = {"first_order_flexes": int(flex), "rattlers": rattlers}
    info = {"ok": r.ok, "s": out["s"], "s_float": float(z[-1]), "result": r, "active": active,
            "flex": flex, "rattlers": rattlers, "newton_resid": resid}
    return out, info


RECORD_MARGIN = mpf("1e-10")  # below this a lower s is a rediscovery (records are stored to 14 to 34 digits)


def record_protocol(out: dict, n: int, log=print) -> bool:
    """CLAUDE.md record handling. Returns True only if the fresh-process re-verification passes."""
    rec = record_s(n)
    if rec is None:
        return False
    with mpmath.workdps(60):
        margin = mpf(rec) - mpf(out["s"])
        if margin <= 0:
            return False
        if margin < RECORD_MARGIN:
            log(f"  s below stored record by {mpmath.nstr(margin, 3)}: within the record's stored "
                f"precision, treated as a rediscovery (no record claim)")
            return False
    jpath = ROOT / f"RECORD_CANDIDATE_n{n}.json"
    spath = ROOT / f"RECORD_CANDIDATE_n{n}.svg"
    save_packing(jpath, out)
    write_svg(out, spath, f"n = {n}, s = {out['s'][:24]}, record candidate; {out['source']}")
    bar = "#" * 78
    log(f"\n{bar}\n{bar}\n  POSSIBLE NEW RECORD  n = {n}\n  old record s = {rec}\n  new s        = {out['s']}\n"
        f"  margin       = {mpmath.nstr(margin, 12)}\n  written: {jpath.name}, {spath.name}\n{bar}\n{bar}")
    log("  re-verifying in a fresh process ...")
    p = subprocess.run([sys.executable, "-m", "src.verify", str(jpath)], cwd=ROOT,
                       capture_output=True, text=True)
    log("  " + p.stdout.strip())
    ok = p.returncode == 0
    log("  FRESH-PROCESS VERIFICATION " + ("PASSED: record candidate stands" if ok else "FAILED: no claim"))
    return ok


def polish_file(path, act_tol=1e-8, skip_slsqp=False, log=print):
    packing = load_packing(path)
    n = packing["n"]
    out, info = polish_packing(packing, act_tol=act_tol, skip_slsqp=skip_slsqp, log=log)
    rec = record_s(n)
    if info["ok"]:
        vdir = ROOT / "verified"
        vdir.mkdir(exist_ok=True)
        dest = vdir / f"n{n}_s{float(out['s']):.12f}.json"
        save_packing(dest, out)
        log(f"  verified -> {dest.relative_to(ROOT)}")
        gap = f"{float(out['s']) - float(rec):+.3e}" if rec else "n/a"
        rig = "rigid (first order)" if info["flex"] == 0 else f"{info['flex']} first-order flex(es)"
        log(f"RESULT {Path(path).name}: PASS s={out['s'][:22]} gap_to_record={gap} "
            f"{rig}; rattlers: {info['rattlers'] or 'none'}")
        info["record"] = record_protocol(out, n, log=log)
    else:
        log(f"RESULT {Path(path).name}: FAIL {info['result'].summary()}")
    return out, info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.polish")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--act-tol", type=float, default=1e-8, help="active contact tolerance (float gap)")
    ap.add_argument("--quiet", action="store_true", help="print RESULT lines only")
    ap.add_argument("--skip-slsqp", action="store_true", help="go straight to the LP stage (large n)")
    args = ap.parse_args(argv)

    def log(msg):
        if not args.quiet or msg.startswith("RESULT") or "RECORD" in msg:
            print(msg, flush=True)

    ok = True
    for f in args.files:
        _, info = polish_file(f, act_tol=args.act_tol, skip_slsqp=args.skip_slsqp, log=log)
        ok &= info["ok"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
