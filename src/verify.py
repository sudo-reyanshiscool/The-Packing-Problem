"""Exact judge for unit-square packings (mpmath, 50 digits).

A packing passes iff every corner of every square lies in [0, s]^2 and no two
squares overlap with positive area. Violations up to TOL (1e-30) count as
touching. Overlap depth is the separating-axis penetration depth: the minimum,
over the 4 edge normals of the pair, of the projected interval overlap.

CLI: python -m src.verify records/17.json [more.json ...] [--quiet]
Exit code 0 if all pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import mpmath
from mpmath import mpf

DPS = 50
TOL = mpf("1e-30")  # do not weaken (CLAUDE.md)


@dataclass
class Result:
    ok: bool
    n: int
    s: mpf
    worst: mpf                 # worst violation found (0 if none)
    worst_where: str           # "pair i,j", "wall i <side>" or ""
    worst_wall: mpf
    worst_wall_where: str
    worst_pair: mpf
    worst_pair_where: str
    failures: int              # number of violations above TOL

    def summary(self) -> str:
        v = "PASS" if self.ok else "FAIL"
        where = f" at {self.worst_where}" if self.worst_where else ""
        return (f"{v}  n={self.n}  s={mpmath.nstr(self.s, 20)}  "
                f"worst={mpmath.nstr(self.worst, 5)}{where}  "
                f"(wall {mpmath.nstr(self.worst_wall, 5)} {self.worst_wall_where or '-'}; "
                f"pair {mpmath.nstr(self.worst_pair, 5)} {self.worst_pair_where or '-'}; "
                f"violations>{mpmath.nstr(TOL, 1)}: {self.failures})")


class _Sq:
    __slots__ = ("x", "y", "u", "v", "pts")

    def __init__(self, cx, cy, th):
        c, s = mpmath.cos(th), mpmath.sin(th)
        h = mpf(1) / 2
        self.x, self.y = cx, cy
        self.u = (c, s)       # edge normals (unit)
        self.v = (-s, c)
        self.pts = [(cx + a * c - b * s, cy + a * s + b * c)
                    for a in (-h, h) for b in (-h, h)]


def _proj(ax, pts):
    ps = [x * ax[0] + y * ax[1] for x, y in pts]
    return min(ps), max(ps)


def pair_depth(a: _Sq, b: _Sq):
    """SAT penetration depth; <= 0 means separated (value is minus the gap on the best axis)."""
    best = None
    for ax in (a.u, a.v, b.u, b.v):
        lo1, hi1 = _proj(ax, a.pts)
        lo2, hi2 = _proj(ax, b.pts)
        ov = min(hi1, hi2) - max(lo1, lo2)
        if best is None or ov < best:
            best = ov
    return best


def verify_packing(packing: dict, verbose: bool = False) -> Result:
    with mpmath.workdps(DPS):
        n = int(packing["n"])
        s = mpf(packing["s"])
        sq = [_Sq(mpf(q["cx"]), mpf(q["cy"]), mpf(q["theta"])) for q in packing["squares"]]
        if len(sq) != n:
            raise ValueError(f"n = {n} but {len(sq)} squares")
        zero = mpf(0)
        fails = 0

        ww, www = zero, ""
        for i, q in enumerate(sq):
            for x, y in q.pts:
                for d, side in ((-x, "left"), (x - s, "right"), (-y, "bottom"), (y - s, "top")):
                    if d > TOL:
                        fails += 1
                        if verbose:
                            print(f"  wall violation: square {i} {side} {mpmath.nstr(d, 5)}")
                    if d > ww:
                        ww, www = d, f"wall {i} {side}"

        pw, pww = zero, ""
        r2 = mpf(2)  # centres further apart than sqrt(2) cannot overlap
        for i in range(n):
            a = sq[i]
            for j in range(i + 1, n):
                b = sq[j]
                dx, dy = a.x - b.x, a.y - b.y
                if dx * dx + dy * dy >= r2:
                    continue
                d = pair_depth(a, b)
                if d > TOL:
                    fails += 1
                    if verbose:
                        print(f"  overlap: squares {i},{j} depth {mpmath.nstr(d, 5)}")
                if d > pw:
                    pw, pww = d, f"pair {i},{j}"

        worst, where = (ww, www) if ww >= pw else (pw, pww)
        return Result(ok=fails == 0, n=n, s=s, worst=worst, worst_where=where,
                      worst_wall=ww, worst_wall_where=www,
                      worst_pair=pw, worst_pair_where=pww, failures=fails)


def verify_file(path, verbose=False) -> Result:
    from .io_records import load_packing
    return verify_packing(load_packing(path), verbose=verbose)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.verify")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every violation")
    args = ap.parse_args(argv)
    all_ok = True
    for f in args.files:
        r = verify_file(f, verbose=args.verbose)
        print(f"{f}: {r.summary()}")
        all_ok &= r.ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
