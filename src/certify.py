"""Rigorous certificate of a packing by interval arithmetic (mpmath.iv).

verify.py is the project judge, but it works in ordinary 50-digit floating
point and accepts violations up to 1e-30 as touching. This module gives a
mathematical proof: every quantity is computed as an interval that provably
encloses the true value (rounding outward), and the packing is certified only
if every separating gap is provably positive and every corner is provably
inside the container. Contacts that touch exactly cannot be certified (their
gap encloses 0), so the packing is first inflated by a factor 1 + eps about
its centre (default eps = 1e-40), which turns every touching contact into a
gap of at least eps / 2 while raising s by s * eps. The certificate therefore
proves

    s(n) <= s_certified = s * (1 + eps),

which for eps = 1e-40 is the stated s to 40 digits.

Separation is proved with the separating axis theorem: for each pair, some
edge normal of one square (interval computed) separates the projections of
the two squares' corners with a provably positive gap.

CLI: python -m src.certify verified/<file>.json [--eps 1e-40] [--dps 80]
Exit code 0 if certified.
"""
from __future__ import annotations

import argparse
import sys

import mpmath
from mpmath import iv

DPS = 80


def _corners(cx, cy, th):
    c, s = iv.cos(th), iv.sin(th)
    h = iv.mpf(1) / 2
    pts = [(cx + a * c - b * s, cy + a * s + b * c) for a in (-h, h) for b in (-h, h)]
    return pts, [(c, s), (-s, c)]


def _proj(ax, pts):
    ps = [x * ax[0] + y * ax[1] for x, y in pts]
    lo = ps[0]
    hi = ps[0]
    for p in ps[1:]:
        lo = iv.mpf([min(lo.a, p.a), min(lo.b, p.b)])
        hi = iv.mpf([max(hi.a, p.a), max(hi.b, p.b)])
    return lo, hi


def certify(packing: dict, eps="1e-40", dps=DPS, log=print) -> tuple[bool, str]:
    """Return (certified, s_certified as a string) using interval arithmetic."""
    iv.dps = dps
    n = int(packing["n"])
    s0 = iv.mpf(packing["s"])
    f = 1 + iv.mpf(eps)
    s = s0 * f
    sq = []
    for q in packing["squares"]:
        cx = (iv.mpf(q["cx"]) - s0 / 2) * f + s / 2
        cy = (iv.mpf(q["cy"]) - s0 / 2) * f + s / 2
        pts, axes = _corners(cx, cy, iv.mpf(q["theta"]))
        sq.append((cx, cy, pts, axes))
    if len(sq) != n:
        raise ValueError("n mismatch")

    # containment: every corner interval provably inside [0, s]
    for i, (_, _, pts, _) in enumerate(sq):
        for x, y in pts:
            if not (x.a > 0 and y.a > 0 and x.b < s.a and y.b < s.a):
                return False, f"square {i}: corner not provably inside the container"

    # separation: some axis provably separates each close pair
    two = iv.mpf(2)
    for i in range(n):
        cxi, cyi, pi, ai = sq[i]
        for j in range(i + 1, n):
            cxj, cyj, pj, aj = sq[j]
            dx, dy = cxi - cxj, cyi - cyj
            d2 = dx * dx + dy * dy
            if d2.a >= two.b:
                continue  # provably further apart than sqrt(2): cannot overlap
            ok = False
            for ax in ai + aj:
                lo1, hi1 = _proj(ax, pi)
                lo2, hi2 = _proj(ax, pj)
                if lo2.a > hi1.b or lo1.a > hi2.b:
                    ok = True
                    break
            if not ok:
                return False, f"pair {i},{j}: no provably separating axis"
    with mpmath.workdps(dps):
        return True, mpmath.nstr(mpmath.mpf(s.b), 45, strip_zeros=False)


def main(argv=None):
    from .io_records import load_packing
    ap = argparse.ArgumentParser(prog="python -m src.certify")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--eps", default="1e-40", help="inflation factor minus 1")
    ap.add_argument("--dps", type=int, default=DPS)
    args = ap.parse_args(argv)
    all_ok = True
    for f in args.files:
        p = load_packing(f)
        ok, msg = certify(p, eps=args.eps, dps=args.dps)
        if ok:
            print(f"{f}: CERTIFIED n={p['n']} s(n) <= {msg} (interval arithmetic, {args.dps} digits, "
                  f"inflation {args.eps})")
        else:
            print(f"{f}: NOT CERTIFIED: {msg}")
        all_ok &= ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
