"""Basin hopping over packing arrangements, with polishing built in.

anneal.py restarts from scratch and keeps landing in the same few local
optima. This driver walks between optima instead. Each worker runs chains:

  1. start: a random anneal start, a structured start (axis-aligned frame
     plus free interior), or a neighbouring record (n + 1 with one square
     removed, n - 1 with one added), annealed with the shrink ladder;
  2. float polish (polish.slsqp_polish) to the exact local optimum of the
     arrangement's contact structure; the rounded s identifies the basin;
  3. kick: re-randomise 2 or 3 squares, or swap two squares with different
     tilts, or straighten a tilted square; grow s by --grow, re-anneal with a
     short ladder, polish again;
  4. Metropolis acceptance on s at temperature --t-hop; occasional restart
     from the best basin seen by this chain.

Distinct basins (s rounded to 1e-7) are counted in logs/hop_n{n}_{stamp}.json.
Any basin below the record by more than 1e-9 is polished exactly and put
through the record protocol. The run stops after --hours.

CLI: python -m src.hop --n 17 --hours 12 --workers 14
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace

import numpy as np

from .anneal import HALF_PI, Config, _ladder, _probs, _scale, _seed, _warmup, save_candidate
from .geometry import max_violation
from .io_records import RECORDS, ROOT, as_float_arrays, load_packing, make_packing, record_s
from .polish import choose_constraints, polish_file, slsqp_polish, split


# ----------------------------------------------------------------- starts

def random_start(rng, n, s):
    return (0.5 + rng.random(n) * (s - 1.0), 0.5 + rng.random(n) * (s - 1.0), rng.random(n) * HALF_PI)


def structured_start(rng, n, s):
    """Axis-aligned squares along two or three walls, the rest random and tilted."""
    k = int(math.floor(s))
    frame = []
    kind = rng.integers(4)
    if kind == 0:      # bottom row + left column (L)
        frame = [(0.5 + i, 0.5) for i in range(k)] + [(0.5, 0.5 + j) for j in range(1, k)]
    elif kind == 1:    # bottom and top rows
        frame = [(0.5 + i, 0.5) for i in range(k)] + [(0.5 + i, s - 0.5) for i in range(k)]
    elif kind == 2:    # U shape
        frame = [(0.5 + i, 0.5) for i in range(k)] + [(0.5, 0.5 + j) for j in range(1, k)] \
            + [(s - 0.5, 0.5 + j) for j in range(1, k)]
    else:              # bottom row only
        frame = [(0.5 + i, 0.5) for i in range(k)]
    m = min(len(frame), n - 2)
    rng.shuffle(frame)
    frame = frame[:m]
    x = np.array([p[0] for p in frame] + list(1.0 + rng.random(n - m) * (s - 2.0)))
    y = np.array([p[1] for p in frame] + list(1.0 + rng.random(n - m) * (s - 2.0)))
    t = np.concatenate([np.zeros(m), rng.random(n - m) * HALF_PI])
    return x, y, t


def neighbour_start(rng, n, s):
    """Record for n + 1 with one square removed, or n - 1 with one square added, scaled to s."""
    for m in ([n + 1, n - 1] if rng.random() < 0.5 else [n - 1, n + 1]):
        p = RECORDS / f"{m}.json"
        if p.exists():
            s_rec, x, y, t = as_float_arrays(load_packing(p))
            if m == n + 1:
                i = rng.integers(m)
                x, y, t = np.delete(x, i), np.delete(y, i), np.delete(t, i)
            else:
                x = np.append(x, 0.5 + rng.random() * (s_rec - 1.0))
                y = np.append(y, 0.5 + rng.random() * (s_rec - 1.0))
                t = np.append(t, rng.random() * HALF_PI)
            _scale(x, y, s_rec, s)
            return x, y, t
    return random_start(rng, n, s)


STARTS = {"random": random_start, "structured": structured_start, "neighbour": neighbour_start}


# ------------------------------------------------------------------ kicks

def kick(rng, x, y, t, s):
    x, y, t = x.copy(), y.copy(), t.copy()
    n = len(x)
    u = rng.random()
    if u < 0.5:
        for i in rng.choice(n, size=rng.integers(2, 4), replace=False):
            x[i] = 0.5 + rng.random() * (s - 1.0)
            y[i] = 0.5 + rng.random() * (s - 1.0)
            t[i] = rng.random() * HALF_PI
        return x, y, t, "rerandomise"
    if u < 0.75:
        tilted = [i for i in range(n) if min(t[i], HALF_PI - t[i]) > 0.05]
        flat = [i for i in range(n) if i not in tilted]
        if tilted and flat:
            i, j = rng.choice(tilted), rng.choice(flat)
            x[i], x[j], y[i], y[j] = x[j], x[i], y[j], y[i]
            return x, y, t, "swap tilt"
    i = rng.integers(n)
    t[i] = 0.0 if min(t[i], HALF_PI - t[i]) > 0.05 else rng.random() * HALF_PI
    return x, y, t, "retilt"


# ------------------------------------------------------------------ chain

def anneal_from(cfg, x, y, t, s, stats):
    """Shrink ladder from a given state. Returns (s, x, y, t) or None."""
    probs = _probs(cfg)
    steps = np.array([0.1 * s, 0.3, 0.01])
    best, _ = _ladder(cfg, x.copy(), y.copy(), t.copy(), s, s, cfg.t0, cfg.shrink, steps, probs, stats)
    return best


def float_polish(x, y, t, s):
    """Exact local optimum of the arrangement (float64). Returns (s, x, y, t, viol)."""
    z, cons = slsqp_polish(x, y, t, s, rounds=6, log=lambda m: None)
    n = len(x)
    xp, yp, tp, sp = split(z, n)
    tp = tp % HALF_PI
    return float(sp), xp, yp, tp, float(max_violation(xp, yp, tp, sp))


def run_chain(args):
    cfg, seed, start_kind, hops, s0, grow, t_hop, deadline = args
    rng = np.random.default_rng(seed)
    _seed(seed % (2**32))
    n = cfg.n
    stats = np.zeros(2)
    basins = {}
    out = {"seed": seed, "start": start_kind, "hops": 0, "basins": basins, "best": None, "moves": 0}

    def visit(res, how):
        if res is None:
            return None
        s_b, xb, yb, tb = res
        try:
            sp, xp, yp, tp, viol = float_polish(xb, yb, tb, s_b)
        except Exception:
            return None
        if viol > 1e-9 or not np.isfinite(sp):
            return None
        key = round(sp, 7)
        basins[key] = basins.get(key, 0) + 1
        if out["best"] is None or sp < out["best"]["s"]:
            out["best"] = {"s": sp, "x": xp.tolist(), "y": yp.tolist(), "t": tp.tolist(), "how": how}
        return sp, xp, yp, tp

    x, y, t = STARTS[start_kind](rng, n, s0)
    cur = visit(anneal_from(cfg, x, y, t, s0, stats), start_kind)
    if cur is None:
        out["moves"] = int(stats[0])
        return out
    best_local = cur
    short = replace(cfg, max_sweeps=cfg.max_sweeps // 4)
    for h in range(hops):
        if time.time() > deadline:
            break
        s_c, xc, yc, tc = cur
        xk, yk, tk, how = kick(rng, xc, yc, tc, s_c)
        s_k = s_c * (1.0 + grow)
        _scale(xk, yk, s_c, s_k)
        stats_h = np.zeros(2)
        new = visit(anneal_from(short, xk, yk, tk, s_k, stats_h), how)
        stats[0] += stats_h[0]
        out["hops"] += 1
        if new is None:
            continue
        ds = new[0] - s_c
        if ds <= 0 or rng.random() < math.exp(-ds / t_hop):
            cur = new
            if new[0] < best_local[0]:
                best_local = new
        if rng.random() < 0.05:
            cur = best_local
    out["moves"] = int(stats[0])
    return out


# ----------------------------------------------------------------- driver

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.hop")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hops", type=int, default=40, help="kicks per chain")
    ap.add_argument("--grow", type=float, default=0.015, help="relative s growth before a re-anneal")
    ap.add_argument("--t-hop", type=float, default=2e-3, help="Metropolis temperature on s")
    ap.add_argument("--start-sweeps", type=int, default=200000, help="ladder budget for the chain start")
    ap.add_argument("--s0", type=float, default=None)
    args = ap.parse_args(argv)

    n = args.n
    rec = record_s(n)
    rec_f = float(rec) if rec else None
    s0 = args.s0 or (1.05 * rec_f if rec_f else 1.05 * (math.sqrt(n) + 0.5))
    workers = args.workers or max(1, os.cpu_count() - 1)
    base = args.seed if args.seed is not None else int(time.time()) % 1_000_000_000
    cfg = Config(n=n, s0=s0, max_sweeps=args.start_sweeps)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logf = open(logs / f"hop_n{n}_{stamp}.log", "w")
    state_path = logs / f"hop_n{n}_{stamp}.json"

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    deadline = time.time() + args.hours * 3600
    log(f"hop n={n} hours={args.hours} workers={workers} base_seed={base} record={rec} s0={s0:.6f} "
        f"hops={args.hops} grow={args.grow} t_hop={args.t_hop} start_sweeps={args.start_sweeps}")
    _warmup()

    kinds = ["random", "structured", "neighbour", "random"]

    def jobs():
        k = 0
        while time.time() < deadline:
            yield (cfg, base + k, kinds[k % len(kinds)], args.hops, s0, args.grow, args.t_hop, deadline)
            k += 1

    basins = {}
    best = None
    chains = hops = 0
    t_start = time.time()
    last_save = t_start
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    pool = ctx.Pool(workers)
    try:
        for r in pool.imap_unordered(run_chain, jobs()):
            chains += 1
            hops += r["hops"]
            for k, v in r["basins"].items():
                basins[k] = basins.get(k, 0) + v
            b = r["best"]
            mark = ""
            if b is not None and (best is None or b["s"] < best["s"]):
                best = dict(b, seed=r["seed"])
                mark = " *best"
                if rec_f and b["s"] < rec_f - 1e-9:
                    mark += " BELOW RECORD IN FLOAT64: exact polish follows"
                    res = dict(seed=r["seed"], s=b["s"], x=b["x"], y=b["y"], t=b["t"],
                               max_pen=float(max_violation(np.array(b["x"]), np.array(b["y"]),
                                                           np.array(b["t"]), b["s"])))
                    path = save_candidate(res, n, f"hop {stamp}")
                    log(f"  saved {path.name}; polishing exactly")
                    try:
                        _, info = polish_file(path, log=log)
                        log(f"  exact s = {info['s'][:30]} record protocol passed = {info.get('record')}")
                    except Exception as e:  # keep the run alive
                        log(f"  exact polish failed: {e!r}")
            bs = f"{b['s']:.9f}" if b else "none"
            log(f"[{chains}] seed {r['seed']} {r['start']:<10} hops={r['hops']} best={bs} "
                f"basins={len(r['basins'])} {(time.time() - t_start) / 3600:.2f}h{mark}")
            if time.time() - last_save > 300:
                last_save = time.time()
                top = sorted(basins.items())[:40]
                with open(state_path, "w") as f:
                    json.dump({"chains": chains, "hops": hops, "distinct_basins": len(basins),
                               "best": best, "lowest_basins": top}, f, indent=1)
            if time.time() > deadline:
                break
    finally:
        pool.terminate()
        pool.join()
    top = sorted(basins.items())[:40]
    with open(state_path, "w") as f:
        json.dump({"chains": chains, "hops": hops, "distinct_basins": len(basins),
                   "best": best, "lowest_basins": top}, f, indent=1)
    wall = time.time() - t_start
    log("-" * 60)
    log(f"chains {chains}, hops {hops}, distinct basins {len(basins)}, wall {wall / 3600:.2f} h")
    if best:
        log(f"best s = {best['s']:.9f} (seed {best['seed']}, via {best['how']})")
        if rec_f:
            log(f"record = {rec_f:.9f}; gap = {best['s'] - rec_f:+.3e}")
    log("lowest basins (s, visits): " + " ".join(f"{k:.7f}:{v}" for k, v in top[:12]))
    logf.close()


if __name__ == "__main__":
    main()
