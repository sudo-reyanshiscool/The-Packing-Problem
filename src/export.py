"""Export a packing as a records-page style SVG and a full-precision JSON.

The SVG follows the conventions of the records page (David Ellsworth): the
container side is the entity &s;, the viewBox is 0 0 &s; &s;, each unit square
is a <rect width="1" height="1"> placed with translate(cx cy) rotate(deg)
translate(-.5 -.5). SVG y points down, so cy_svg = s - cy and the rotation is
negated. The vendor parser (vendor/packing_tools/parse_svg_packing.py) reads
this form; export is checked by a round trip through it.

CLI: python -m src.export verified/<file>.json [-o out/basename]
Writes <basename>.svg and <basename>.json (default: next to the input).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mpmath
from mpmath import mpf

from .io_records import load_packing, save_packing, svg_to_packing

DIGITS = 50


def svg_text(packing: dict, comment: str | None = None) -> str:
    s = packing["s"]
    n = packing["n"]
    head = ['<?xml version="1.0" encoding="UTF-8"?>']
    if comment:
        head += ["<!--", "    " + comment.replace("--", "- -"), "-->"]
    head += ['<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [',
             f'    <!ENTITY s "{s}">', ']>',
             '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             'width="100%" height="100%" viewBox="0 0 &s; &s;" '
             'style="fill:#B2B2B2; stroke:black; stroke-width:0.0012" id="svg">',
             '    <defs>', '        <rect width="&s;" height="&s;" id="outer"/>', '    </defs>',
             '    <use xlink:href="#outer" style="fill:white; stroke:none"/>']
    body = []
    with mpmath.workdps(DIGITS + 10):
        sm = mpf(s)
        for q in packing["squares"]:
            cx, cy, th = mpf(q["cx"]), mpf(q["cy"]), mpf(q["theta"])
            deg = -th * 180 / mpmath.pi
            fx = mpmath.nstr(cx, DIGITS, strip_zeros=True)
            fy = mpmath.nstr(sm - cy, DIGITS, strip_zeros=True)
            if deg == 0:
                tr = f"translate({fx} {fy}) translate(-.5 -.5)"
            else:
                tr = f"translate({fx} {fy}) rotate({mpmath.nstr(deg, DIGITS, strip_zeros=True)}) translate(-.5 -.5)"
            body.append(f'    <rect width="1" height="1" transform="{tr}"/>')
    tail = ['    <use xlink:href="#outer" style="fill:none"/>', '</svg>', '']
    return "\n".join(head + body + tail)


def write_svg(packing: dict, path, comment: str | None = None) -> Path:
    path = Path(path)
    path.write_text(svg_text(packing, comment))
    return path


def export(packing: dict, base, comment: str | None = None) -> tuple[Path, Path]:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    svg = write_svg(packing, base.with_suffix(".svg"), comment)
    js = base.with_suffix(".json")
    save_packing(js, packing)
    return svg, js


def roundtrip_error(packing: dict, svg_path) -> str:
    """Largest coordinate difference after parsing the SVG with the vendor tools (as a string)."""
    back = svg_to_packing(svg_path)
    with mpmath.workdps(60):
        worst = mpf(0)
        worst = max(worst, abs(mpf(back["s"]) - mpf(packing["s"])))
        for a, b in zip(packing["squares"], back["squares"]):
            worst = max(worst, abs(mpf(a["cx"]) - mpf(b["cx"])), abs(mpf(a["cy"]) - mpf(b["cy"])))
            dt = abs(mpf(a["theta"]) - mpf(b["theta"]))
            dt = min(dt, abs(dt - mpmath.pi / 2))
            worst = max(worst, dt)
        return mpmath.nstr(worst, 3)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.export")
    ap.add_argument("file")
    ap.add_argument("-o", "--out", default=None, help="output basename (default: beside the input)")
    ap.add_argument("--comment", default=None)
    args = ap.parse_args(argv)
    p = load_packing(args.file)
    base = Path(args.out) if args.out else Path(args.file).with_suffix("")
    comment = args.comment or f"n = {p['n']}, s = {p['s'][:24]}; source: {p.get('source', '?')}"
    svg, js = export(p, base, comment)
    print(f"wrote {svg} and {js}; vendor parser round trip error {roundtrip_error(p, svg)}")


if __name__ == "__main__":
    main()
