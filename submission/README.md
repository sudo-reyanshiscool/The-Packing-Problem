# Refined packings of n unit squares in a square: 33 entries

Refinements of the packings listed as "Not yet analytically optimized" on
https://kingbird.myphotos.cc/packing/squares_in_squares.html (n = 103 to 307).
Each packing keeps the arrangement (contact structure) of the page's packing and
moves it to the exact local optimum of that structure, so every listed contact
is closed to 1e-49 and s is smaller than the page value. The exported files
are then scaled by 1 + 1e-45 about the centre so that every contact has a
strictly positive gap (a 50-digit rounding of an exact contact can otherwise
show an overlap of 1e-48, which check_packing.py flags at its 1e-47
tolerance). This raises s by under 2e-44.

Files: `square-<n>.svg` (records-page format, entity `s`, 50-digit coordinates,
parseable by parse_svg_packing.py) and `square-<n>.json` (centre x, centre y,
angle in radians, all as 50-digit strings; container [0, s] x [0, s], y up).

| n | records page s | refined s (first 20 digits) | improvement | first-order flexes | rattlers |
|---|---|---|---|---|---|
| 103 | 10.70378195534367 | 10.70377984325949967 | 2.11e-06 | 49 | 8 |
| 105 | 10.80761933330707 | 10.80758853210852206 | 3.08e-05 | 39 | 2 |
| 131 | 11.95652543280926 | 11.95652190096897789 | 3.53e-06 | 16 | 0 |
| 132 | 11.99137344423646 | 11.99134529921282089 | 2.81e-05 | 15 | 2 |
| 152 | 12.83095954472600 | 12.83071880097660995 | 2.41e-04 | 17 | 2 |
| 154 | 12.93166712962655 | 12.93166373480336931 | 3.39e-06 | 18 | 0 |
| 155 | 12.95844711161529 | 12.95820608999601522 | 2.41e-04 | 54 | 9 |
| 156 | 12.98208376048414 | 12.98208269851689310 | 1.06e-06 | 10 | 0 |
| 180 | 13.93508705291129 | 13.93500417986041418 | 8.29e-05 | 25 | 1 |
| 181 | 13.95690672341755 | 13.95679525362657222 | 1.11e-04 | 39 | 4 |
| 182 | 13.97419105332569 | 13.97409071312143900 | 1.00e-04 | 19 | 1 |
| 206 | 14.87221902902620 | 14.87221902560743773 | 3.42e-09 | 80 | 16 |
| 207 | 14.89395494255333 | 14.89395463477440103 | 3.08e-07 | 228 | 38 |
| 208 | 14.93776656277905 | 14.93766151179943494 | 1.05e-04 | 44 | 3 |
| 209 | 14.95861500087481 | 14.95855669619650494 | 5.83e-05 | 36 | 5 |
| 210 | 14.97413341886404 | 14.97381333242663476 | 3.20e-04 | 36 | 2 |
| 236 | 15.87607539315201 | 15.87606262399118623 | 1.28e-05 | 88 | 13 |
| 238 | 15.93965520031394 | 15.93958571219001050 | 6.95e-05 | 20 | 2 |
| 239 | 15.95635358406308 | 15.95623227480651755 | 1.21e-04 | 38 | 5 |
| 240 | 15.97556282833087 | 15.97536574437535865 | 1.97e-04 | 26 | 2 |
| 241 | 15.99080517810520 | 15.99043971533776189 | 3.65e-04 | 12 | 0 |
| 268 | 16.87931143465371 | 16.87909297877424011 | 2.18e-04 | 79 | 1 |
| 270 | 16.94062059800744 | 16.94057120393215811 | 4.94e-05 | 111 | 2 |
| 271 | 16.95499909412532 | 16.95479445649385014 | 2.05e-04 | 37 | 4 |
| 272 | 16.96971602419903 | 16.96944902363442011 | 2.67e-04 | 30 | 1 |
| 273 | 16.98820725030513 | 16.98811466983505648 | 9.26e-05 | 13 | 0 |
| 297 | 17.74106074604732 | 17.74092880620554934 | 1.32e-04 | 161 | 19 |
| 301 | 17.86889155557430 | 17.86867836510717185 | 2.13e-04 | 58 | 11 |
| 303 | 17.93125509556197 | 17.93106553169518041 | 1.90e-04 | 40 | 2 |
| 304 | 17.94910783564662 | 17.94878498074399882 | 3.23e-04 | 26 | 2 |
| 305 | 17.96066201401205 | 17.96053667218048935 | 1.25e-04 | 43 | 6 |
| 306 | 17.96913960675661 | 17.96846608989620187 | 6.74e-04 | 30 | 2 |
| 307 | 17.98272201579610 | 17.98205199952844466 | 6.70e-04 | 27 | 2 |

## Method

1. Contacts of the page packing identified; the arrangement is written as a
   smooth constrained problem: minimise s such that, for every close pair, the
   four corners of one square lie beyond a chosen edge line of the other, and
   every corner lies inside [0, s]^2.
2. Local optimum found in float64 (SLSQP, then sequential linear programming
   with a trust region and second-order correction).
3. Active contacts solved as equations in 60-digit arithmetic (Gauss-Newton
   with an SVD pseudo-inverse) so the contacts close to 1e-49.
4. Rigidity: nullspace of the active-contact Jacobian (first order only).

## Verification

* `verify.py` (this project): separating axis theorem on every pair plus corner
  containment, mpmath at 50 digits, overlap tolerance 1e-30. Every packing
  passes, re-run in a fresh process from the saved file.
* `check_packing.py` (David Ellsworth's tools, tolerance 1e-47 derived from the
  digits of s): VALID for every SVG (logs/vendor_check_submission.log).
* `certify.py`: interval arithmetic (mpmath.iv, 80 digits) proves
  s(n) <= s_listed * (1 + 1e-40) for each packing: all corners provably inside
  and every close pair provably separated by an edge normal.

Code: https://github.com/sudo-reyanshiscool/The-Packing-Problem
(`python -m src.polish records/<n>.json` reproduces each refinement from the
page SVG converted with `python -m src.io_records svg`).
