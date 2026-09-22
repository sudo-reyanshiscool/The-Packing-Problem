"""Load and save packings, and adapt the vendor Ellsworth tools.

Packing JSON (see CLAUDE.md): container [0, s] x [0, s], y up, unit squares
given by centre (cx, cy) and theta in radians reduced to [0, pi/2). All
numbers are stored as strings.

CLI:
    python -m src.io_records svg records/svg/square-17.svg -o records/17.json
    python -m src.io_records svg records/svg/square-5.svg -o records/5.json --entity "s=2+sqrt(2)/2"
    python -m src.io_records index page.html -o records/index.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from decimal import Decimal, getcontext
from pathlib import Path

import mpmath

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "packing_tools"
RECORDS = ROOT / "records"

DIGITS = 50  # digits written to JSON


# ---------------------------------------------------------------- JSON I/O

def load_packing(path) -> dict:
    with open(path) as f:
        p = json.load(f)
    if len(p["squares"]) != p["n"]:
        raise ValueError(f"{path}: n = {p['n']} but {len(p['squares'])} squares")
    return p


def save_packing(path, packing: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(packing, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def fmt(x, digits: int = DIGITS) -> str:
    """Format a number (float, Decimal, mpf, str) as a full-precision string."""
    if isinstance(x, float):
        return repr(x)
    with mpmath.workdps(digits + 10):
        return mpmath.nstr(mpmath.mpf(str(x)), digits, strip_zeros=False,
                           min_fixed=-mpmath.inf, max_fixed=mpmath.inf)


def reduce_theta(theta):
    """Reduce an angle (mpf) to [0, pi/2)."""
    q = mpmath.pi / 2
    t = theta % q
    return t if t >= 0 else t + q


def make_packing(n, s, squares, source, verified=False, digits=DIGITS) -> dict:
    """Build a packing dict. squares: iterable of (cx, cy, theta) in any numeric type."""
    out = []
    with mpmath.workdps(digits + 10):
        for cx, cy, th in squares:
            t = reduce_theta(mpmath.mpf(str(th)) if not isinstance(th, float) else mpmath.mpf(th))
            out.append({"cx": fmt(cx, digits), "cy": fmt(cy, digits), "theta": fmt(t, digits)})
    return {"n": int(n), "s": fmt(s, digits), "squares": out,
            "source": source, "verified": bool(verified)}


def as_float_arrays(packing: dict):
    import numpy as np
    x = np.array([float(q["cx"]) for q in packing["squares"]])
    y = np.array([float(q["cy"]) for q in packing["squares"]])
    t = np.array([float(q["theta"]) for q in packing["squares"]])
    return float(packing["s"]), x, y, t


# ---------------------------------------------------------- record lookup

def record_s(n: int):
    """Current record s for n as a string, read from records/ (never hard-coded).

    Prefers records/<n>.json, falls back to records/index.json. Returns None if unknown.
    """
    p = RECORDS / f"{n}.json"
    if p.exists():
        return load_packing(p)["s"]
    idx = RECORDS / "index.json"
    if idx.exists():
        with open(idx) as f:
            entry = json.load(f).get(str(n))
        if entry:
            return entry["s"]
    return None


# ------------------------------------------------------- vendor SVG adapter

def _vendor_parser():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import parse_svg_packing  # noqa: E402  (vendor, untouched)
    return parse_svg_packing


def _override_entities(svg_text: str, overrides: dict) -> str:
    for name, value in overrides.items():
        pat = re.compile(r'(<!ENTITY\s+' + re.escape(name) + r'\s+")[^"]*(")')
        svg_text, k = pat.subn(lambda m: m.group(1) + value + m.group(2), svg_text)
        if k != 1:
            raise ValueError(f"entity {name!r} found {k} times")
    return svg_text


def svg_to_packing(svg_path, overrides: dict | None = None, precision: int = 60) -> dict:
    """Parse a records-page SVG with the vendor parser and convert to our JSON format.

    overrides: {entity_name: value_string} substituted before parsing, e.g. a
    higher-precision closed-form value for s. The coordinate transform below
    mirrors the __main__ block of vendor/parse_svg_packing.py.
    """
    vp = _vendor_parser()
    text = Path(svg_path).read_text()
    if overrides:
        text = _override_entities(text, overrides)
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as tf:
        tf.write(text)
        tmp = tf.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            squares, entities, prec, centred = vp.parse_svg(tmp, precision)
    finally:
        os.unlink(tmp)

    getcontext().prec = prec + 20
    pi = vp.hp_pi(prec)
    half_pi = pi / Decimal(2)
    thr = Decimal(10) ** (-(prec - 10))
    s = Decimal(entities["s"])
    half_s = Decimal(0) if centred else s / Decimal(2)

    out = []
    for sq in squares:
        th = half_pi - sq["theta"]
        for snap in (Decimal(0), half_pi, -half_pi, pi, -pi):
            if abs(th - snap) < thr:
                th = snap
                break
        th = (th + pi) % half_pi
        x = sq["x"] - half_s          # centred, y up (as vendor)
        y = -(sq["y"] - half_s)
        out.append((x + s / 2, y + s / 2, th))  # corner origin

    src = f"records page SVG {Path(svg_path).name}"
    if overrides:
        src += " with " + ", ".join(f"{k}={v[:24]}..." for k, v in overrides.items())
    return make_packing(len(out), s, out, src, verified=False)


def eval_entity(expr: str, digits: int = 70) -> str:
    """Evaluate an mpmath expression (e.g. '2+sqrt(2)/2') to a digit string."""
    with mpmath.workdps(digits + 10):
        v = eval(expr, {"__builtins__": {}}, {k: getattr(mpmath, k) for k in dir(mpmath) if not k.startswith("_")})
        return mpmath.nstr(mpmath.mpf(v), digits, strip_zeros=False)


# ----------------------------------------------------- records page index

def parse_records_page(html: str) -> dict:
    """Extract {n: {"s", "svg", "analytic"}} from the records page HTML."""
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    boxes = html.split('<div class="box">')[1:]
    index = {}
    for b in boxes:
        m = re.match(r'\s*<font size="\+3">([\d,\s]+)<br>', b)
        if not m:
            continue
        ns = [int(k) for k in re.findall(r"\d+", m.group(1))]
        svg = re.search(r'href="([^"]+\.svg)"', b)
        val = re.search(r"\\Nn\{([\d.]+)\}", b)
        if not val:
            val = re.search(r"\$s\s*=\s*([\d.]+)\s*\$", b)
        if not val:
            continue
        for n in ns:
            index[str(n)] = {
                "s": val.group(1),
                "svg": svg.group(1) if svg else None,
                "analytic": "Not yet analytically optimized" not in b,
            }
    return dict(sorted(index.items(), key=lambda kv: int(kv[0])))


# --------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.io_records")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("svg", help="convert a records-page SVG to packing JSON")
    a.add_argument("svg")
    a.add_argument("-o", "--out", required=True)
    a.add_argument("--entity", action="append", default=[],
                   help="override NAME=<mpmath expr>, e.g. s=2+sqrt(2)/2")
    b = sub.add_parser("index", help="build records/index.json from the records page HTML")
    b.add_argument("html")
    b.add_argument("-o", "--out", default=str(RECORDS / "index.json"))
    args = ap.parse_args(argv)

    if args.cmd == "svg":
        ov = {}
        for e in args.entity:
            k, v = e.split("=", 1)
            ov[k.strip()] = eval_entity(v)
        p = svg_to_packing(args.svg, ov or None)
        save_packing(args.out, p)
        print(f"n = {p['n']}, s = {p['s']} -> {args.out}")
    else:
        idx = parse_records_page(Path(args.html).read_text())
        with open(args.out, "w") as f:
            json.dump(idx, f, indent=1)
            f.write("\n")
        na = [n for n, e in idx.items() if not e["analytic"]]
        print(f"{len(idx)} records -> {args.out}; not analytically optimised: {len(na)}")


if __name__ == "__main__":
    main()
