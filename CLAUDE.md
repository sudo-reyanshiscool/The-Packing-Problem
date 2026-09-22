# CLAUDE.md

## Project

Attempt to beat known records for packing n unit squares into the smallest possible square, starting with n = 17.

- **Primary target:** s(17) < 4.67553009360455 (Bidwell, 1998, based on Hämäläinen, 1980).
- **Secondary targets:** records marked "Not yet analytically optimized" on the records page (e.g. n = 103, 105, 126). These are more realistic wins.
- **Records page:** https://kingbird.myphotos.cc/packing/squares_in_squares.html (maintained by David Ellsworth)
- **Reference tools:** https://github.com/Davidebyzero/packing_squares_in_squares__tools (clone into `vendor/`)

A result only counts if it passes `verify.py`. Nothing else is evidence.

## Owner preferences

- British English in docs, comments and output.
- No em dashes anywhere (docs, comments, commit messages, printed output).
- Terse communication. Report results as numbers, not adjectives.
- Metric/SI where units apply.
- When the user says **"Box-Box"**: stage all changes, commit with a clear message, and push.

## Stack

- Python 3.11+
- `numpy`, `numba` for the search hot loop
- `scipy` for polishing (SLSQP / trust-constr)
- `mpmath` (50 digits) for verification only
- `multiprocessing` for parallel random starts
- `pytest` for tests

Install: `pip install numpy numba scipy mpmath pytest`

## Structure

```
.
├── CLAUDE.md
├── README.md
├── vendor/                  # cloned Ellsworth tools (do not edit)
├── records/                 # known record packings as JSON (seeds + test cases)
├── src/
│   ├── geometry.py          # square corners, SAT, penetration depth (float64, numba)
│   ├── verify.py            # exact judge (mpmath)
│   ├── anneal.py            # simulated annealing search
│   ├── polish.py            # contact-constrained optimisation
│   ├── export.py            # SVG + JSON output in records-page format
│   └── io_records.py        # load/save packings, parse vendor SVGs
├── tests/
├── candidates/              # anything under threshold from anneal.py
├── verified/                # anything that passed verify.py after polishing
└── logs/
```

## Data format

A packing is JSON:

```json
{
  "n": 17,
  "s": "4.6755300936045500000000000000",
  "squares": [
    {"cx": "0.5", "cy": "0.5", "theta": "0.0"}
  ],
  "source": "anneal run 2026-09-22T14:03 seed 8812",
  "verified": false
}
```

- Store numbers as strings to preserve full precision.
- `theta` in radians, reduced to [0, pi/2).
- Square side is exactly 1. Container is [0, s] x [0, s].

## Pipeline (build in this order, do not skip ahead)

### Stage 1: Verifier (`verify.py`)

The judge. Build and test this before anything else.

- Separating axis theorem on every pair, plus containment of all four corners of every square.
- mpmath at 50 digits. Touching is allowed; any positive-area overlap fails.
- Tolerance: treat violations below 1e-30 as touching. Anything larger fails.
- Output: pass/fail, worst violation, and which pair or wall caused it.
- **Tests:** known records for n = 5, 10, 11, 17 must pass. Deliberately perturbed copies must fail.

### Stage 2: Annealing search (`anneal.py`)

- Random initial positions and angles in a container ~5% larger than target.
- Energy = sum of pairwise penetration depth (SAT, float64) + wall penetration.
- Moves: translate, rotate, translate+rotate, swap two squares, re-randomise one square (rare).
- When energy reaches ~0, shrink s slightly and continue.
- Adaptive step sizes; temperature schedule configurable.
- Many independent starts in parallel, one per core.
- Save any configuration below `--save-below` threshold to `candidates/`.
- CLI: `python -m src.anneal --n 17 --starts 1000 --save-below 4.690 --workers 0`
- `--seed-from records/17.json --perturb 0.1` to start from a perturbed known record.

### Stage 3: Polish (`polish.py`)

- Identify active contacts (square-square, square-wall) within a tolerance.
- Minimise s subject to non-overlap constraints with scipy, starting from the candidate.
- Re-run `verify.py` on the output. Only move to `verified/` if it passes.
- Report whether the result is rigid (all squares locked) or has rattlers.

### Stage 4: Export (`export.py`)

- SVG matching the records-page style, parseable by the vendor tools.
- JSON with full-precision coordinates.

## Record handling

If a verified s is below the current record for that n:

1. Write `RECORD_CANDIDATE_n{N}.json` and `.svg` to the repo root.
2. Print a loud, unmissable message with the old record, new s, and the margin.
3. Re-verify from the saved file in a fresh process.
4. Do not claim a record in conversation until step 3 passes.

Current records must be read from `records/`, not hard-coded, except the n = 17 value above.

## Rules

- Never weaken `verify.py` tolerances to make a result pass.
- Never report a result from float64 search code as final.
- Validate the whole pipeline by rediscovering s(10) = 3 + 1/sqrt(2) = 3.70710678... from random starts before spending long runs on n = 17.
- Long runs go in the background with output to `logs/`. Give a time estimate per 1,000 starts.
- If results plateau above the record after a sensible budget, say so plainly with the best s found and the gap.
- Keep `vendor/` untouched. Write adapters in `io_records.py`.

## Commands

```
pytest                                               # all tests
python -m src.verify records/17.json                 # verify a packing
python -m src.anneal --n 10 --starts 200             # sanity run
python -m src.anneal --n 17 --starts 5000 --save-below 4.690
python -m src.polish candidates/<file>.json
python -m src.export verified/<file>.json
```
