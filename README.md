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
| 3 | `src/polish.py` | done, tested |
| 4 | `src/export.py` | done, tested |
| extra | `src/certify.py` | interval-arithmetic certificate (rigorous proof of an upper bound) |

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

## Polish (Stage 3)

`python -m src.polish candidates/<file>.json`. Three stages:

1. SLSQP on smooth constraints: for each close pair one separating edge is
   chosen and the four corners of the other square must lie beyond it; every
   corner inside the container. Any feasible point is a packing. Edges are
   re-chosen and the solve repeated until stable.
2. Sequential LP (HiGHS, feasibility tolerance 1e-10 with rows scaled by 1e3,
   trust region, LP-based second-order correction). Needed for n above about
   100, where SLSQP stops on degenerate active sets after a few iterations.
3. Exact stage: the active constraints (gap below 1e-8) are solved as
   equations in 60-digit mpmath by Gauss-Newton with an exact SVD
   pseudo-inverse (float64 pseudo-inverse above 160 unknowns). Residuals reach
   1e-50. If verify.py still fails, the packing is inflated by 1 + 4d.

Output: verify.py summary, first-order rigidity (nullspace of the active
Jacobian; n = 5 reports one flex because the tilted square's rotation is
blocked only at second order), rattlers, JSON in verified/. A verified s more
than 1e-10 below the stored record triggers the record protocol
(RECORD_CANDIDATE_n{N}.json/.svg, fresh-process re-verification).

Validation: s(10) = 3 + 1/sqrt(2) recovered to 3e-34 from an annealed
candidate; all n = 17 candidates within 1e-2 of the record polish to the
record value (4.67553009360455095163...) or to worse local optima
(4.677648, 4.678926, 4.683831, 4.688103).

## Export (Stage 4) and certificate

`python -m src.export verified/<file>.json` writes a records-page style SVG
(50-digit coordinates; vendor parser round trip error 1e-50, vendor
check_packing.py reports VALID) and a JSON.

`python -m src.certify verified/<file>.json` proves s(n) <= s(1 + 1e-40) by
interval arithmetic (mpmath.iv, 80 digits): every corner provably inside the
container and every close pair provably separated on some edge normal. The
inflation is needed because exact contacts have a gap interval containing 0.

## Results

### n = 17 (primary target)

Not beaten. 3000 random starts (0.13 h per 1000 starts at 12 workers) plus
30 seeded starts: every candidate within 1e-2 of the record polishes either to
the record itself (4.67553009360455095163..., 6 first-order flexes, 1 rattler)
or to a worse local optimum (4.677648, 4.678926, 4.680125, 4.681980,
4.682716, 4.683831, 4.688103). Best random-start float s 4.677654 (+2.1e-3).

Basin hopping (`python -m src.hop --n 17 --hours 12`): chains start from a
random, structured (axis-aligned frame plus free interior) or neighbouring
record (n = 18 minus a square, n = 16 plus one) start, polish to the local
optimum of the arrangement, then kick 2 to 3 squares (or swap or straighten a
tilted one), re-anneal and polish again, with Metropolis acceptance on s.
Distinct optima are counted in logs/hop_n17_<stamp>.json.

### Non-analytic records (secondary targets)

All 35 records marked "Not yet analytically optimized" with n <= 307 were
converted from the page SVGs, verified, and polished from the stored packing
(`python -m src.polish records/<n>.json`; `--skip-slsqp` above n = 300).
"gap" is polished s minus the page value. "fresh re-verify" is the CLAUDE.md
record protocol (RECORD_CANDIDATE_n{N}.json/.svg written, verify.py re-run in
a new process). "certified" is `src.certify` (interval arithmetic). The n = 103
and n = 105 SVGs were also checked with the vendor check_packing.py (VALID).
Gaps below 1e-10 (n = 110, 126) are within the page's precision and are not
claimed.

| n | record (page) | polished s | gap | flexes | rattlers | fresh re-verify | certified |
|---|---|---|---|---|---|---|---|
| 103 | 10.70378195534367 | 10.703779843259499 | -2.112e-06 | 49 | 8 | yes | yes |
| 105 | 10.80761933330707 | 10.807588532108522 | -3.080e-05 | 39 | 2 | yes | yes |
| 110 | 10.99679327401957 | 10.996793273953749 | -6.582e-11 | 7 | 1 | rediscovery |  |
| 126 | 11.77473513240654 | 11.774735132387832 | -1.871e-11 | 6 | 0 | rediscovery |  |
| 131 | 11.95652543280926 | 11.956521900968977 | -3.532e-06 | 16 | 0 | yes | yes |
| 132 | 11.99137344423646 | 11.991345299212820 | -2.815e-05 | 15 | 2 | yes | yes |
| 152 | 12.83095954472600 | 12.830718800976609 | -2.407e-04 | 17 | 2 | yes | yes |
| 154 | 12.93166712962655 | 12.931663734803369 | -3.395e-06 | 18 | 0 | yes | yes |
| 155 | 12.95844711161529 | 12.958206089996015 | -2.410e-04 | 54 | 9 | yes | yes |
| 156 | 12.98208376048414 | 12.982082698516893 | -1.062e-06 | 10 | 0 | yes | yes |
| 180 | 13.93508705291129 | 13.935004179860414 | -8.287e-05 | 25 | 1 | yes | yes |
| 181 | 13.95690672341755 | 13.956795253626572 | -1.115e-04 | 39 | 4 | yes | yes |
| 182 | 13.97419105332569 | 13.974090713121439 | -1.003e-04 | 19 | 1 | yes | yes |
| 206 | 14.87221902902620 | 14.872219025607437 | -3.419e-09 | 80 | 16 | yes | yes |
| 207 | 14.89395494255333 | 14.893954634774401 | -3.078e-07 | 228 | 38 | yes | yes |
| 208 | 14.93776656277905 | 14.937661511799434 | -1.051e-04 | 44 | 3 | yes | yes |
| 209 | 14.95861500087481 | 14.958556696196504 | -5.830e-05 | 36 | 5 | yes | yes |
| 210 | 14.97413341886404 | 14.973813332426634 | -3.201e-04 | 36 | 2 | yes | yes |
| 236 | 15.87607539315201 | 15.876062623991186 | -1.277e-05 | 88 | 13 | yes | yes |
| 238 | 15.93965520031394 | 15.939585712190010 | -6.949e-05 | 20 | 2 | yes | yes |
| 239 | 15.95635358406308 | 15.956232274806517 | -1.213e-04 | 38 | 5 | yes | yes |
| 240 | 15.97556282833087 | 15.975365744375358 | -1.971e-04 | 26 | 2 | yes | yes |
| 241 | 15.99080517810520 | 15.990439715337761 | -3.655e-04 | 12 | 0 | yes | yes |
| 268 | 16.87931143465371 | 16.879092978774240 | -2.185e-04 | 79 | 1 | yes | yes |
| 270 | 16.94062059800744 | 16.940571203932158 | -4.939e-05 | 111 | 2 | yes | yes |
| 271 | 16.95499909412532 | 16.954794456493850 | -2.046e-04 | 37 | 4 | yes | yes |
| 272 | 16.96971602419903 | 16.969449023634420 | -2.670e-04 | 30 | 1 | yes | yes |
| 273 | 16.98820725030513 | 16.988114669835056 | -9.258e-05 | 13 | 0 | yes | yes |
| 297 | 17.74106074604732 | 17.740928806205549 | -1.319e-04 | 161 | 19 | yes | yes |
| 301 | 17.86889155557430 | 17.868678365107171 | -2.132e-04 | 58 | 11 | yes | yes |
| 303 | 17.93125509556197 | 17.931065531695180 | -1.896e-04 | 40 | 2 | yes | yes |
| 304 | 17.94910783564662 | 17.948784980743998 | -3.229e-04 | 26 | 2 | yes | yes |
| 305 | 17.96066201401205 | 17.960536672180489 | -1.253e-04 | 43 | 6 | yes | yes |
| 306 | 17.96913960675661 | 17.968466089896201 | -6.735e-04 | 30 | 2 | yes | yes |
| 307 | 17.98272201579610 | 17.982051999528444 | -6.700e-04 | 27 | 2 | yes | yes |

Every improved packing keeps the record's contact structure; the stored
packings were simply not at the local optimum of the contact-constrained
problem (they are numerically found, hence the page's label). Logs: logs/polish_record_n<n>.log.
All 33 improvements are certified by src.certify; HiGHS needs a per-solve
time limit (LP_OPTS) because a degenerate simplex ran for hours at n = 304.


## Commands

```
pytest
python -m src.verify records/17.json
python -m src.anneal --n 10 --starts 200
python -m src.anneal --n 17 --starts 5000 --save-below 4.690
python -m src.anneal --n 17 --seed-from records/17.json --perturb 0.1
python -m src.polish candidates/<file>.json
python -m src.polish records/105.json          # polish a stored record
python -m src.export verified/<file>.json
python -m src.certify verified/<file>.json
python -m src.hop --n 17 --hours 12 --workers 14     # basin hopping, background
```
