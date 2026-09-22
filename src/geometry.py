"""Float64 geometry for the search hot loop (numba). Not a judge: see verify.py.

Square i: centre (x[i], y[i]), angle t[i], side 1. Container [0, s]^2.
Penetration depth between two squares is the separating-axis minimum overlap
over the 4 edge normals (0 if separated).
"""
import math

import numpy as np
from numba import njit

SQRT2 = math.sqrt(2.0)


@njit(cache=True, fastmath=False)
def pair_pen(xi, yi, ci, si, xj, yj, cj, sj):
    """Penetration depth of squares i, j given centres and (cos, sin) of angles."""
    dx = xj - xi
    dy = yj - yi
    d2 = dx * dx + dy * dy
    if d2 >= 2.0:
        return 0.0
    # |cos| and |sin| of relative angle give the projected half-width of the
    # other square on each normal: 0.5 * (|a.u| + |a.v|).
    cr = abs(ci * cj + si * sj)
    sr = abs(ci * sj - si * cj)
    r = 0.5 + 0.5 * (cr + sr)
    best = r - abs(dx * ci + dy * si)
    if best <= 0.0:
        return 0.0
    o = r - abs(-dx * si + dy * ci)
    if o <= 0.0:
        return 0.0
    if o < best:
        best = o
    o = r - abs(dx * cj + dy * sj)
    if o <= 0.0:
        return 0.0
    if o < best:
        best = o
    o = r - abs(-dx * sj + dy * cj)
    if o <= 0.0:
        return 0.0
    if o < best:
        best = o
    return best


@njit(cache=True)
def wall_pen(x, y, c, s_, s):
    """Sum of wall penetration depths for a square (x, y, cos, sin) in [0, s]^2."""
    h = 0.5 * (abs(c) + abs(s_))
    e = 0.0
    d = h - x
    if d > 0.0:
        e += d
    d = x + h - s
    if d > 0.0:
        e += d
    d = h - y
    if d > 0.0:
        e += d
    d = y + h - s
    if d > 0.0:
        e += d
    return e


@njit(cache=True)
def local_energy(i, xi, yi, ci, si, x, y, c, sn, s, skip):
    """Energy of square i at a given pose against all others (except skip) plus walls."""
    e = wall_pen(xi, yi, ci, si, s)
    n = x.shape[0]
    for j in range(n):
        if j == i or j == skip:
            continue
        e += pair_pen(xi, yi, ci, si, x[j], y[j], c[j], sn[j])
    return e


@njit(cache=True)
def total_energy(x, y, c, sn, s):
    n = x.shape[0]
    e = 0.0
    for i in range(n):
        e += wall_pen(x[i], y[i], c[i], sn[i], s)
        for j in range(i + 1, n):
            e += pair_pen(x[i], y[i], c[i], sn[i], x[j], y[j], c[j], sn[j])
    return e


@njit(cache=True)
def max_violation(x, y, t, s):
    """Largest single pair penetration or wall penetration (float64 diagnostic)."""
    n = x.shape[0]
    c = np.cos(t)
    sn = np.sin(t)
    m = 0.0
    for i in range(n):
        h = 0.5 * (abs(c[i]) + abs(sn[i]))
        m = max(m, h - x[i], x[i] + h - s, h - y[i], y[i] + h - s)
        for j in range(i + 1, n):
            m = max(m, pair_pen(x[i], y[i], c[i], sn[i], x[j], y[j], c[j], sn[j]))
    return m
