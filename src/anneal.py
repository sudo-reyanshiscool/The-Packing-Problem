"""Simulated annealing search for small containers (float64; results are candidates only).

Energy = sum over pairs of SAT penetration depth + sum of wall penetration
depths (linear: a squared penalty rewards spreading overlap over many contacts
and traps the search in squeezed grid rows). Each start:

  1. random poses in a container s0 (~5% above the record), or a perturbed record;
  2. anneal at fixed s until energy < --e-tol ("feasible");
  3. on success shrink s by a relative step (positions scaled about the centre);
     on failure restore the last feasible state and halve the step;
  4. stop when the step drops below --min-shrink or the move budget is spent.

The best feasible s per start is reported; anything below --save-below is
written to candidates/. Nothing here is final: candidates go to polish + verify.

CLI:
  python -m src.anneal --n 10 --starts 200
  python -m src.anneal --n 17 --starts 1000 --save-below 4.690 --workers 0
  python -m src.anneal --n 17 --seed-from records/17.json --perturb 0.1
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from numba import njit

from .geometry import local_energy, max_violation, pair_pen, total_energy, wall_pen
from .io_records import ROOT, as_float_arrays, load_packing, make_packing, record_s, save_packing

HALF_PI = 0.5 * math.pi

# move types
M_TRANS, M_ROT, M_BOTH, M_SWAP, M_RAND = 0, 1, 2, 3, 4


# ------------------------------------------------------------------ kernel

@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _recompute_local(x, y, c, sn, s, e_loc):
    n = x.shape[0]
    for i in range(n):
        e_loc[i] = wall_pen(x[i], y[i], c[i], sn[i], s)
    for i in range(n):
        for j in range(i + 1, n):
            p = pair_pen(x[i], y[i], c[i], sn[i], x[j], y[j], c[j], sn[j])
            e_loc[i] += p
            e_loc[j] += p


@njit(cache=True)
def anneal_kernel(x, y, t, sbox, n_moves, T0, T1, steps, probs, p_bias, e_tol, stats, mu, p_s):
    """Anneal poses (and, if p_s > 0, the container). Modifies x, y, t, sbox, steps in place.

    Minimises H = E + mu * s, where E is the overlap energy in container sbox[0].
    With p_s = 0 the container is fixed (fixed-s mode).
    steps: [translate, rotate, log-scale of s] (adapted to ~40% acceptance).
    probs: cumulative probabilities of move types (translate, rotate, both, swap, rerandomise).
    stats: [moves done, accepted] (accumulated).
    Returns final energy E. Exits early once E < e_tol (pass e_tol < 0 to disable).
    """
    s = sbox[0]
    n = x.shape[0]
    c = np.cos(t)
    sn = np.sin(t)
    e_loc = np.zeros(n)
    _recompute_local(x, y, c, sn, s, e_loc)
    E = total_energy(x, y, c, sn, s)
    pold = np.zeros(n)
    pnew = np.zeros(n)

    acc_w = np.zeros(3)   # accepted in window, per step kind (trans, rot, s)
    pro_w = np.zeros(3)
    xs = np.empty(n)
    ys = np.empty(n)
    window = 200
    logr = math.log(T1 / T0) / max(n_moves - 1, 1)

    for k in range(n_moves):
        if E < e_tol:
            E = total_energy(x, y, c, sn, s)  # tracked E drifts; confirm exactly
            if E < e_tol:
                stats[0] += k
                sbox[0] = s
                return E
        T = T0 * math.exp(logr * k)

        if p_s > 0.0 and np.random.random() < p_s:
            # container move: scale positions about the centre
            s_new = s * math.exp(steps[2] * (2.0 * np.random.random() - 1.0))
            f = s_new / s
            for q in range(n):
                xs[q] = 0.5 * s_new + (x[q] - 0.5 * s) * f
                ys[q] = 0.5 * s_new + (y[q] - 0.5 * s) * f
            E_new = total_energy(xs, ys, c, sn, s_new)
            dH = E_new - E + mu * (s_new - s)
            pro_w[2] += 1
            if dH <= 0.0 or np.random.random() < math.exp(-dH / T):
                x[:] = xs
                y[:] = ys
                s = s_new
                E = E_new
                _recompute_local(x, y, c, sn, s, e_loc)
                acc_w[2] += 1
                stats[1] += 1
            continue

        u = np.random.random()
        mt = 0
        while mt < 4 and u >= probs[mt]:
            mt += 1

        if mt == M_SWAP:
            i = np.random.randint(n)
            j = np.random.randint(n - 1)
            if j >= i:
                j += 1
            # exchange positions, keep angles
            xi, yi, xj, yj = x[j], y[j], x[i], y[i]
            old = (local_energy(i, x[i], y[i], c[i], sn[i], x, y, c, sn, s, j)
                   + local_energy(j, x[j], y[j], c[j], sn[j], x, y, c, sn, s, i))
            pij = pair_pen(x[i], y[i], c[i], sn[i], x[j], y[j], c[j], sn[j])
            old += pij
            # evaluate with the other square temporarily moved out of the way
            new = (local_energy(i, xi, yi, c[i], sn[i], x, y, c, sn, s, j)
                   + local_energy(j, xj, yj, c[j], sn[j], x, y, c, sn, s, i))
            pij = pair_pen(xi, yi, c[i], sn[i], xj, yj, c[j], sn[j])
            new += pij
            dE = new - old
            if dE <= 0.0 or np.random.random() < math.exp(-dE / T):
                x[i], y[i], x[j], y[j] = xi, yi, xj, yj
                E += dE
                _recompute_local(x, y, c, sn, s, e_loc)
                stats[1] += 1
            continue

        # choose a square, biased towards ones with energy
        i = np.random.randint(n)
        if np.random.random() < p_bias:
            for _ in range(n):
                if e_loc[i] > 0.0:
                    break
                i = np.random.randint(n)

        xi, yi, ti = x[i], y[i], t[i]
        if mt == M_RAND:
            xi = 0.5 + np.random.random() * (s - 1.0)
            yi = 0.5 + np.random.random() * (s - 1.0)
            ti = np.random.random() * HALF_PI
        else:
            if mt == M_TRANS or mt == M_BOTH:
                xi += steps[0] * (2.0 * np.random.random() - 1.0)
                yi += steps[0] * (2.0 * np.random.random() - 1.0)
            if mt == M_ROT or mt == M_BOTH:
                ti += steps[1] * (2.0 * np.random.random() - 1.0)
                ti = ti % HALF_PI
        ci = math.cos(ti)
        si = math.sin(ti)

        dE = wall_pen(xi, yi, ci, si, s) - wall_pen(x[i], y[i], c[i], sn[i], s)
        for j in range(n):
            if j == i:
                pold[j] = 0.0
                pnew[j] = 0.0
                continue
            a = pair_pen(x[i], y[i], c[i], sn[i], x[j], y[j], c[j], sn[j])
            b = pair_pen(xi, yi, ci, si, x[j], y[j], c[j], sn[j])
            pold[j] = a
            pnew[j] = b
            dE += pnew[j] - pold[j]

        accept = dE <= 0.0 or np.random.random() < math.exp(-dE / T)
        if mt != M_RAND:
            if mt != M_ROT:
                pro_w[0] += 1
            if mt != M_TRANS:
                pro_w[1] += 1
        if accept:
            x[i], y[i], t[i], c[i], sn[i] = xi, yi, ti, ci, si
            E += dE
            e_loc[i] = wall_pen(xi, yi, ci, si, s)
            for j in range(n):
                if j != i:
                    e_loc[j] += pnew[j] - pold[j]
                    e_loc[i] += pnew[j]
            stats[1] += 1
            if mt != M_RAND:
                if mt != M_ROT:
                    acc_w[0] += 1
                if mt != M_TRANS:
                    acc_w[1] += 1

        # adapt step sizes
        for q in range(3):
            if pro_w[q] >= window:
                r = acc_w[q] / pro_w[q]
                if r > 0.5:
                    steps[q] *= 1.25
                elif r < 0.3:
                    steps[q] *= 0.8
                acc_w[q] = 0.0
                pro_w[q] = 0.0
        if steps[0] > 0.25 * s:
            steps[0] = 0.25 * s
        if steps[0] < 1e-13:
            steps[0] = 1e-13
        if steps[1] > HALF_PI:
            steps[1] = HALF_PI
        if steps[1] < 1e-13:
            steps[1] = 1e-13
        if steps[2] > 0.05:
            steps[2] = 0.05
        if steps[2] < 1e-14:
            steps[2] = 1e-14

        if (k & 4095) == 4095:  # guard against drift
            E = total_energy(x, y, c, sn, s)
            if E < 0.0:
                E = 0.0

    stats[0] += n_moves
    sbox[0] = s
    return total_energy(x, y, c, sn, s)


@njit(cache=True)
def anneal_fixed_s(x, y, t, s, n_moves, T0, T1, steps, probs, p_bias, e_tol, stats):
    """Anneal at fixed container size s (see anneal_kernel)."""
    sbox = np.array([s])
    return anneal_kernel(x, y, t, sbox, n_moves, T0, T1, steps, probs, p_bias, e_tol, stats, 0.0, 0.0)


# ---------------------------------------------------------------- one start

@dataclass
class Config:
    n: int
    s0: float
    t0: float = 1e-2         # stage start temperature (energy units: depth)
    t_ratio: float = 1e-8    # stage end temperature = t0 * t_ratio
    stage_sweeps: int = 2000  # moves per stage = stage_sweeps * n
    max_sweeps: int = 1000000  # budget per start (sweeps; includes pressure phase)
    retries: int = 4          # failed stages at one shrink step before halving it
    reset_ladder: int = 1     # 1: when the step bottoms out, restart it from best until budget ends
    e_tol: float = 1e-10     # "feasible" threshold on energy (sum of depths)
    shrink: float = 0.01     # initial relative shrink step
    min_shrink: float = 1e-7
    grow_limit: float = 1.15  # give up if s must exceed s0 * this to find a feasible start
    p_trans: float = 0.35
    p_rot: float = 0.2
    p_both: float = 0.35
    p_swap: float = 0.09
    p_rand: float = 0.01
    p_bias: float = 0.5
    mode: str = "ladder"      # "ladder": fixed-s shrink ladder; "pressure": anneal s first, then finish
    mu: float = 0.5           # pressure: H = E + mu * s (mu < 1 keeps the T = 0 optimum overlap-free)
    press_sweeps: int = 200000  # pressure phase length (sweeps)
    press_t0: float = 0.05
    press_t1: float = 1e-8
    p_s: float = 0.0          # container move probability in pressure phase (0: use 1 / n)
    fin_t0: float = 1e-6      # finishing ladder start temperature
    fin_shrink: float = 1e-4  # finishing ladder initial relative step
    seed_from: str | None = None
    perturb: float = 0.0


TUNABLE = ("mode", "mu", "press_sweeps", "press_t0", "press_t1", "p_s", "fin_t0", "fin_shrink",
           "t0", "t_ratio", "stage_sweeps", "max_sweeps", "retries", "reset_ladder", "e_tol",
           "shrink", "min_shrink", "grow_limit", "p_trans", "p_rot", "p_both", "p_swap",
           "p_rand", "p_bias")


def _probs(cfg: Config):
    p = np.array([cfg.p_trans, cfg.p_rot, cfg.p_both, cfg.p_swap, cfg.p_rand], dtype=np.float64)
    p = np.cumsum(p / p.sum())
    p[-1] = 1.0
    return p


def _scale(x, y, s_old, s_new):
    f = s_new / s_old
    x[:] = 0.5 * s_new + (x - 0.5 * s_old) * f
    y[:] = 0.5 * s_new + (y - 0.5 * s_old) * f


def _energy(x, y, t, s):
    return total_energy(x, y, np.cos(t), np.sin(t), s)


def _ladder(cfg, x, y, t, s, s_ref, t0, shrink, steps, probs, stats):
    """Fixed-s annealing with a shrinking container: shrink on success, retry then halve on failure."""
    n = x.shape[0]
    stage_moves = cfg.stage_sweeps * n
    budget = cfg.max_sweeps * n
    delta = shrink
    best = None  # (s, x, y, t)
    stages = 0
    fails = 0
    while stats[0] < budget:
        if delta < cfg.min_shrink:
            if not cfg.reset_ladder or best is None:
                break
            delta = shrink  # iterate: restart the shrink ladder from the best state
        stages += 1
        E = anneal_fixed_s(x, y, t, s, stage_moves, t0, t0 * cfg.t_ratio,
                           steps, probs, cfg.p_bias, cfg.e_tol, stats)
        if E < cfg.e_tol:
            best = (s, x.copy(), y.copy(), t.copy())
            fails = 0
            s_new = s * (1.0 - delta)
            _scale(x, y, s, s_new)
            s = s_new
            continue
        fails += 1
        if best is None:
            # no feasible state yet: retry, then grow the container
            if fails > cfg.retries:
                fails = 0
                if s > s_ref * cfg.grow_limit:
                    break
                s_new = s * (1.0 + delta)
                _scale(x, y, s, s_new)
                s = s_new
            continue
        # retry from the last feasible state; after `retries` failures halve the step
        if fails > cfg.retries:
            fails = 0
            delta *= 0.5
        s_b, xb, yb, tb = best
        x, y, t = xb.copy(), yb.copy(), tb.copy()
        s_new = s_b * (1.0 - delta)
        _scale(x, y, s_b, s_new)
        s = s_new
    return best, stages


def run_start(cfg: Config, seed: int) -> dict:
    t_start = time.time()
    rng = np.random.default_rng(seed)
    _seed(seed % (2**32))
    n = cfg.n

    if cfg.seed_from:
        s_rec, x, y, t = as_float_arrays(load_packing(cfg.seed_from))
        s = cfg.s0 if cfg.s0 else s_rec
        _scale(x, y, s_rec, s)
        x += cfg.perturb * rng.standard_normal(n)
        y += cfg.perturb * rng.standard_normal(n)
        t = (t + cfg.perturb * rng.standard_normal(n)) % HALF_PI
    else:
        s = cfg.s0
        x = 0.5 + rng.random(n) * (s - 1.0)
        y = 0.5 + rng.random(n) * (s - 1.0)
        t = rng.random(n) * HALF_PI

    probs = _probs(cfg)
    steps = np.array([0.1 * s, 0.3, 0.01])
    stats = np.zeros(2)
    s_ref = s
    t0, shrink = cfg.t0, cfg.shrink
    if cfg.mode == "pressure":
        sbox = np.array([s])
        p_s = cfg.p_s if cfg.p_s > 0 else 1.0 / n
        anneal_kernel(x, y, t, sbox, cfg.press_sweeps * n, cfg.press_t0, cfg.press_t1,
                      steps, probs, cfg.p_bias, -1.0, stats, cfg.mu, p_s)
        s = s_ref = float(sbox[0])
        t0, shrink = cfg.fin_t0, cfg.fin_shrink
    best, stages = _ladder(cfg, x, y, t, s, s_ref, t0, shrink, steps, probs, stats)

    out = {"seed": seed, "moves": int(stats[0]), "accepted": int(stats[1]), "s_press": s_ref,
           "stages": stages, "seconds": time.time() - t_start, "s": None}
    if best is not None:
        s_b, xb, yb, tb = best
        out.update(s=float(s_b), x=xb.tolist(), y=yb.tolist(), t=tb.tolist(),
                   max_pen=float(max_violation(xb, yb, tb, s_b)))
    return out


# ------------------------------------------------------------------ driver

def _worker(args):
    cfg, seed = args
    return run_start(cfg, seed)


def _warmup():
    cfg = Config(n=3, s0=2.2, stage_sweeps=10, max_sweeps=30, press_sweeps=10)
    run_start(cfg, 0)


def save_candidate(res: dict, n: int, tag: str) -> Path:
    s = res["s"]
    p = make_packing(n, s, zip(res["x"], res["y"], res["t"]),
                     f"anneal {tag} seed {res['seed']}")
    p["max_pen_float"] = repr(res["max_pen"])
    path = ROOT / "candidates" / f"n{n}_s{s:.9f}_seed{res['seed']}.json"
    save_packing(path, p)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.anneal")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--starts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    ap.add_argument("--seed", type=int, default=None, help="base seed (default: time)")
    ap.add_argument("--save-below", type=float, default=None,
                    help="save candidates with s below this (default: record * 1.003)")
    ap.add_argument("--s0", type=float, default=None,
                    help="initial container (default: record * 1.05, or record s when seeded)")
    ap.add_argument("--seed-from", default=None, help="start from a perturbed packing JSON")
    ap.add_argument("--perturb", type=float, default=0.1)
    d = Config(n=0, s0=0)
    for f in TUNABLE:
        v = getattr(d, f)
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args(argv)

    n = args.n
    rec = record_s(n)
    rec_f = float(rec) if rec else None
    if args.s0:
        s0 = args.s0
    elif args.seed_from:
        s0 = 0.0  # use seed packing's s
    elif rec_f:
        s0 = 1.05 * rec_f
    else:
        s0 = 1.05 * (math.sqrt(n) + 0.5)
    save_below = args.save_below if args.save_below else (rec_f * 1.003 if rec_f else float("inf"))
    base = args.seed if args.seed is not None else int(time.time()) % 1_000_000_000
    workers = args.workers or os.cpu_count()

    cfg = Config(n=n, s0=s0, seed_from=args.seed_from,
                 perturb=args.perturb if args.seed_from else 0.0,
                 **{f: getattr(args, f) for f in TUNABLE})

    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = f"run {stamp}"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logf = open(logs / f"anneal_n{n}_{stamp}.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"anneal n={n} starts={args.starts} workers={workers} base_seed={base} "
        f"record={rec} s0={'seed file' if not s0 else f'{s0:.6f}'} save_below={save_below:.6f}")
    log("config " + str({k: v for k, v in asdict(cfg).items()}))

    _warmup()
    t0 = time.time()
    results = []
    best = None
    saved = 0
    jobs = [(cfg, base + k) for k in range(args.starts)]
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
            results.append(r)
            s = r["s"]
            mark = ""
            if s is not None and (best is None or s < best["s"]):
                best = r
                mark = " *best"
            if s is not None and rec_f and s < rec_f:
                mark += " BELOW RECORD IN FLOAT64 (unverified: polish + verify before any claim)"
            if s is not None and s < save_below:
                path = save_candidate(r, n, tag)
                saved += 1
                mark += f" saved {path.name}"
            s_txt = f"{s:.9f}" if s is not None else "none"
            log(f"[{k}/{args.starts}] seed {r['seed']} s={s_txt} pen={r.get('max_pen', 0):.1e} "
                f"moves={r['moves']:.2e} {r['seconds']:.1f}s{mark}")

    wall = time.time() - t0
    ss = np.array([r["s"] for r in results if r["s"] is not None])
    log("-" * 60)
    if len(ss):
        log(f"best s = {ss.min():.9f} (seed {best['seed']})")
        if rec_f:
            log(f"record = {rec_f:.9f}; gap = {ss.min() - rec_f:+.3e}")
        qs = np.percentile(ss, [0, 10, 50, 90])
        log(f"s percentiles 0/10/50/90: " + " ".join(f"{q:.6f}" for q in qs))
        if rec_f:
            for tol in (1e-6, 1e-4, 1e-3):
                log(f"within {tol:.0e} of record: {(ss < rec_f + tol).sum()}/{len(ss)}")
    log(f"saved {saved} candidates; wall {wall:.1f}s; "
        f"{wall / args.starts * 1000 / 3600:.2f} h per 1000 starts at {workers} workers")
    logf.close()


if __name__ == "__main__":
    main()
