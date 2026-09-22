# Packing unit squares in squares

Search for packings of n unit squares in a smaller square than the current
records (https://kingbird.myphotos.cc/packing/squares_in_squares.html).
See CLAUDE.md for rules and the pipeline. Only `src/verify.py` counts as evidence.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install numpy numba scipy mpmath pytest
git submodule update --init          # vendor/packing_tools (Ellsworth tools)
```

## Status

| Stage | Module | State |
|---|---|---|
| 1 | `src/verify.py` | done, tested |
| 2 | `src/anneal.py`, `src/geometry.py` | done, tested |
| 3 | `src/polish.py` | not started |
| 4 | `src/export.py` | not started |

## Records

`records/index.json`: all 194 records from the records page (s to 14 to 16
digits, SVG name, `analytic` flag; 37 are not analytically optimised).
`records/<n>.json`: full-precision packings for n = 5, 10, 11, 17, 103, 105, 126,
converted from `records/svg/` with the vendor parser:

```
python -m src.io_records index page.html -o records/index.json
python -m src.io_records svg records/svg/square-17.svg -o records/17.json
python -m src.io_records svg records/svg/square-5.svg  -o records/5.json  --entity "s=2+sqrt(2)/2"
python -m src.io_records svg records/svg/square-10.svg -o records/10.json --entity "s=3+sqrt(2)/2"
```

The n = 5 and 10 SVGs store s to 30 digits only, which leaves overlaps of
6.86e-30 (above the 1e-30 tolerance; the vendor checker agrees). Their JSON uses
the closed-form s instead.

## Verifier

mpmath at 50 digits, SAT on every pair (interval-intersection depth, as the
vendor checker) plus containment of every corner. Violations up to 1e-30 count
as touching. `python -m src.verify records/17.json`

## Annealing

Energy: sum of SAT penetration depths (minimum translation distance) plus wall
penetration depths, float64, numba. Default mode `ladder`: anneal at fixed s
from T = 1e-2 to 1e-10 until energy < 1e-10, then shrink s by a relative step
(1e-2 initially); after 5 failed stages at one step halve it; when the step drops
below 1e-7 restart it from the best state; stop at 1e6 sweeps (1.7e7 moves for
n = 17). Mode `pressure` (anneal s with H = E + mu s first) is available but
tested worse for n = 17.

Tuning notes (n = 17, 56 starts each unless stated):

- Squared depth energy traps starts in squeezed grid rows (E = 2.4e-3 at every
  failure at n = 10, s = 3.89). Linear depth fixes that.
- Stage temperature proportional to current energy freezes after each shrink:
  median start ends at s = 5.0. Absolute T0 = 1e-2 is better than 1e-3 or 3e-2.
- Pressure mode (mu 0.2 to 0.9): median 5.0, best 4.68413. Ladder: best 4.67557.

Validation runs (15 workers):

| Run | Best s | Gap to record | Hits within 1e-3 | Time per 1000 starts |
|---|---|---|---|---|
| n = 10, 200 random starts | 3.707106903 | +1.2e-7 | 45/200 | 0.07 h |
| n = 17, 150 random starts | 4.677657839 | +2.1e-3 | 0/150 | 0.17 h |
| n = 17, 30 seeded (perturb 0.1) | 4.675547772 | +1.8e-5 | 4/30 | 0.16 h |

In an earlier 56-start tuning batch, one random start reached 4.6755681
(+3.8e-5). All anneal output is float64 and unverified: candidates must go
through polish and verify.

## Commands

```
pytest
python -m src.verify records/17.json
python -m src.anneal --n 10 --starts 200
python -m src.anneal --n 17 --starts 5000 --save-below 4.690
python -m src.anneal --n 17 --seed-from records/17.json --perturb 0.1
```
