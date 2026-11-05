"""QA-side helpers for tests_spec/test_parser_matpower.py.

These helpers are test oracles written from specs/parser_matpower.md only. They never import
anything from agentigrid except the public names listed in interfaces.md.
"""
from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from pathlib import Path

from hypothesis import assume
from hypothesis import strategies as st

from agentigrid.exago_parsers.matpower_model import (
    Branch,
    Bus,
    GenCost,
    Generator,
    MATNetwork,
)
from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.exago_parsers.matpower_writer import write_matpower

DATA_DIR = Path(__file__).parent / "data"

# Every valid fixture in tests_spec/data/ (SPEC-PARSER "Fixtures").
VALID_FIXTURES = (
    "case_min.m",
    "case_extra.m",
    "case_wide.m",
    "case_syntax.m",
    "case_plain.m",
    "case_nogen.m",
    "case_mixed.m",
)

# ---------------------------------------------------------------------------
# Field layout (SPEC-PARSER-02..05 and "Integer columns")
# ---------------------------------------------------------------------------
BUS_COLS = ("bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va",
            "baseKV", "zone", "Vmax", "Vmin")
BUS_OPT = ("lam_P", "lam_Q", "mu_Vmax", "mu_Vmin")
GEN_COLS = ("bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin")
BRANCH_COLS = ("fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", "ratio",
               "angle", "status", "angmin", "angmax")
GENCOST_COLS = ("model", "startup", "shutdown", "ncost")

INT_FIELDS = {
    "bus": {"bus_i", "type", "area", "zone"},
    "gen": {"bus", "status"},
    "branch": {"fbus", "tbus", "status"},
    "gencost": {"model", "ncost"},
}
CORE_LISTS = (("bus", "buses"), ("gen", "generators"), ("branch", "branches"),
              ("gencost", "gencost"))


# ---------------------------------------------------------------------------
# Float match and equivalence (spec "Definitions")
# ---------------------------------------------------------------------------
def feq(a: float, b: float) -> bool:
    """Float match per spec: isclose(rel 1e-9, abs 1e-12), NaN==NaN, same infinity."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _is_int(v) -> bool:
    return type(v) is int  # bool is excluded on purpose


def _cmp_record(section: str, idx: int, cols, a, b):
    for col in cols:
        va, vb = getattr(a, col), getattr(b, col)
        where = f"{section}[{idx}].{col}"
        if col in INT_FIELDS[section]:
            assert _is_int(va), f"{where}: {va!r} is not int"
            assert _is_int(vb), f"{where}: {vb!r} is not int"
            assert va == vb, f"{where}: {va!r} != {vb!r}"
        else:
            assert feq(float(va), float(vb)), f"{where}: {va!r} !~ {vb!r}"


def _cmp_float_list(where: str, la, lb):
    assert len(la) == len(lb), f"{where}: length {len(la)} != {len(lb)}"
    for k, (x, y) in enumerate(zip(la, lb)):
        assert feq(float(x), float(y)), f"{where}[{k}]: {x!r} !~ {y!r}"


def assert_data_equiv(a: MATNetwork, b: MATNetwork) -> None:
    """Data-equivalent (≈) per spec Definitions."""
    assert a.casename == b.casename, f"casename {a.casename!r} != {b.casename!r}"
    assert a.version == b.version, f"version {a.version!r} != {b.version!r}"
    assert feq(float(a.baseMVA), float(b.baseMVA)), f"baseMVA {a.baseMVA} !~ {b.baseMVA}"
    for section, attr in CORE_LISTS:
        la, lb = getattr(a, attr), getattr(b, attr)
        assert len(la) == len(lb), f"len({attr}) {len(la)} != {len(lb)}"
    for i, (x, y) in enumerate(zip(a.buses, b.buses)):
        _cmp_record("bus", i, BUS_COLS, x, y)
        for col in BUS_OPT:
            vx, vy = getattr(x, col), getattr(y, col)
            assert (vx is None) == (vy is None), f"bus[{i}].{col}: {vx!r} vs {vy!r}"
            if vx is not None:
                assert feq(float(vx), float(vy)), f"bus[{i}].{col}: {vx!r} !~ {vy!r}"
    for i, (x, y) in enumerate(zip(a.generators, b.generators)):
        _cmp_record("gen", i, GEN_COLS, x, y)
        _cmp_float_list(f"gen[{i}].extra", x.extra, y.extra)
    for i, (x, y) in enumerate(zip(a.branches, b.branches)):
        _cmp_record("branch", i, BRANCH_COLS, x, y)
        _cmp_float_list(f"branch[{i}].extra", x.extra, y.extra)
    for i, (x, y) in enumerate(zip(a.gencost, b.gencost)):
        _cmp_record("gencost", i, GENCOST_COLS, x, y)
        _cmp_float_list(f"gencost[{i}].coeffs", x.coeffs, y.coeffs)
    assert set(a.extra_sections) == set(b.extra_sections), (
        f"extra_sections keys {sorted(a.extra_sections)} != {sorted(b.extra_sections)}"
    )


def assert_full_equiv(a: MATNetwork, b: MATNetwork) -> None:
    """Fully equivalent per spec Definitions (≈ plus header/extras equality)."""
    assert_data_equiv(a, b)
    assert a.header_comments == b.header_comments, "header_comments differ"
    assert a.extra_sections == b.extra_sections, "extra_sections values differ"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def fixture_text(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8")


def put_case(directory: Path, name: str, text: str, newline: str = "\n") -> Path:
    """Write case text with a chosen line ending, byte-exact."""
    p = Path(directory) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\r\n", "\n")
    if newline != "\n":
        data = data.replace("\n", newline)
    p.write_bytes(data.encode("utf-8"))
    return p


class Workdir:
    """Hands out fresh file paths inside one directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.n = 0

    def path(self, stem: str = "out") -> Path:
        self.n += 1
        return self.root / f"{stem}_{self.n}.m"


def write_read(net: MATNetwork, path: Path) -> tuple[str, MATNetwork]:
    """write_matpower then parse_matpower; also returns the written text (exact chars)."""
    write_matpower(net, path)
    text = Path(path).read_bytes().decode("utf-8")
    return text, parse_matpower(path)


def must_replace(text: str, old: str, new: str) -> str:
    """str.replace that fails loudly if the fixture anchor is missing."""
    assert old in text, f"fixture anchor not found: {old!r}"
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Text oracle: QA's own minimal tokenizer (SPEC-PARSER-08 c, -12, -14)
# ---------------------------------------------------------------------------
_OPEN_RE = re.compile(r"^[ \t]*mpc\.(\w+)[ \t]*=[ \t]*([\[{])", re.M)
_MPC_LINE_RE = re.compile(r"^[ \t]*mpc\.\w+", re.M)


@dataclass
class Block:
    name: str
    kind: str          # "[" or "{"
    body: str          # text between the opening bracket and the first terminator
    start: int         # offset of the "mpc.<name>" line
    end: int           # offset just after the terminator


def find_blocks(text: str) -> dict[str, Block]:
    blocks: dict[str, Block] = {}
    for m in _OPEN_RE.finditer(text):
        term = "];" if m.group(2) == "[" else "};"
        j = text.find(term, m.end())
        assert j != -1, f"section {m.group(1)!r} has no {term!r} terminator"
        blocks[m.group(1)] = Block(m.group(1), m.group(2), text[m.end():j],
                                   m.start(), j + len(term))
    return blocks


def strip_comment(line: str) -> str:
    i = line.find("%")
    return line if i < 0 else line[:i]


def row_lines(body: str) -> list[str]:
    """Non-empty, comment-stripped lines of a numeric block body."""
    out = []
    for line in body.splitlines():
        s = strip_comment(line).strip()
        if s.rstrip(";").strip():
            out.append(s)
    return out


def rows_from_text(text: str, name: str) -> list[list[float]]:
    """Numeric rows of section `name` in one-row-per-line text (Q4 syntax)."""
    blk = find_blocks(text)[name]
    return [[float(t) for t in line.rstrip(";").split()] for line in row_lines(blk.body)]


def raw_tokens(text: str, name: str) -> list[list[str]]:
    blk = find_blocks(text)[name]
    return [line.rstrip(";").split() for line in row_lines(blk.body)]


def cell_elements(body: str) -> list[str]:
    no_comments = "\n".join(strip_comment(l) for l in body.splitlines())
    return re.findall(r"'([^']*)'", no_comments)


def numeric_elements(body: str) -> list[float]:
    no_comments = " ".join(strip_comment(l) for l in body.splitlines())
    return [float(t) for t in re.split(r"[\s;,]+", no_comments) if t]


def assert_same_elements(src_text: str, out_text: str, name: str) -> None:
    src, out = find_blocks(src_text)[name], find_blocks(out_text)[name]
    assert src.kind == out.kind, f"{name}: kind {src.kind} -> {out.kind}"
    if src.kind == "{":
        assert cell_elements(out.body) == cell_elements(src.body), name
    else:
        _cmp_float_list(name, numeric_elements(out.body), numeric_elements(src.body))


def check_text_rules(text: str, net: MATNetwork) -> None:
    """SPEC-PARSER-12: textual rules every written file must satisfy."""
    assert "\r" not in text, "written file contains CR characters"
    funcs = re.findall(r"^[ \t]*function[ \t]+mpc[ \t]*=[ \t]*(\w+)[ \t]*$", text, re.M)
    assert funcs == [net.casename], f"function lines {funcs!r}, casename {net.casename!r}"
    assert re.search(r"^[ \t]*mpc\.version[ \t]*=[ \t]*'2'[ \t]*;", text, re.M), "version line"
    m = re.search(r"^[ \t]*mpc\.baseMVA[ \t]*=[ \t]*([^;%\n]+);", text, re.M)
    assert m, "baseMVA line"
    assert feq(float(m.group(1)), float(net.baseMVA)), "baseMVA value"

    blocks = find_blocks(text)
    for blk in blocks.values():
        nxt = _MPC_LINE_RE.search(text, blk.end)
        tail = text[blk.end: nxt.start() if nxt else len(text)]
        for line in tail.splitlines():
            s = line.strip()
            assert s == "" or s.startswith("%"), (
                f"content after terminator of {blk.name!r}: {line!r}"
            )
    for section, attr in CORE_LISTS:
        items = getattr(net, attr)
        if section not in blocks:
            assert not items, f"section {section!r} missing from written file"
            continue
        blk = blocks[section]
        assert blk.kind == "[", f"{section} must be a numeric block"
        lines = row_lines(blk.body)
        assert len(lines) == len(items), f"{section}: {len(lines)} row lines for {len(items)} rows"
        for line in lines:
            parts = [p for p in line.split(";") if p.strip()]
            assert len(parts) == 1, f"{section}: more than one row on a line: {line!r}"


def prev_statement_end(text: str, pos: int) -> int:
    """End offset of the last non-comment line before `pos` that ends with ';'."""
    end = 0
    offset = 0
    for line in text[:pos].splitlines(keepends=True):
        if strip_comment(line).strip().endswith(";"):
            end = offset + len(line)
        offset += len(line)
    return end


def section_start(text: str, name: str) -> int:
    m = re.search(rf"^[ \t]*mpc\.{re.escape(name)}[ \t]*=", text, re.M)
    assert m, f"mpc.{name} not found"
    return m.start()


# ---------------------------------------------------------------------------
# Hypothesis strategies (SPEC-PARSER-22)
# ---------------------------------------------------------------------------
ALNUM = string.ascii_letters + string.digits
finite = st.floats(allow_nan=False, allow_infinity=False)
identifier = st.from_regex(r"\A[A-Za-z][A-Za-z0-9_]{0,12}\Z")
EXTRA_NAMES = ("genfuel", "gentype", "bus_name")


@st.composite
def network_rows(draw):
    """Raw rows for bus/gen/branch/gencost inside the R1-Q13 domain, mixed widths allowed."""
    nb = draw(st.integers(1, 5))
    ids = draw(st.lists(st.integers(1, 100_000), min_size=nb, max_size=nb, unique=True))
    bus = []
    for i in ids:
        row = [i, draw(st.integers(1, 4)), *[draw(finite) for _ in range(4)],
               draw(st.integers(1, 999)), *[draw(finite) for _ in range(3)],
               draw(st.integers(1, 999)), draw(finite), draw(finite)]
        row += [draw(finite) for _ in range(draw(st.integers(0, 4)))]
        bus.append(row)
    ref = st.sampled_from(ids)
    gen = []
    for _ in range(draw(st.integers(0, 3))):
        row = [draw(ref), *[draw(finite) for _ in range(6)], draw(st.integers(0, 1)),
               draw(finite), draw(finite)]
        row += [draw(finite) for _ in range(draw(st.integers(0, 11)))]
        gen.append(row)
    branch = []
    for _ in range(draw(st.integers(0, 3))):
        row = [draw(ref), draw(ref), *[draw(finite) for _ in range(8)],
               draw(st.integers(0, 1)), draw(finite), draw(finite)]
        row += [draw(finite) for _ in range(draw(st.integers(0, 4)))]
        branch.append(row)
    gencost = []
    if gen and draw(st.booleans()):
        for _ in gen:
            row = [draw(st.sampled_from([1, 2])), draw(finite), draw(finite),
                   draw(st.integers(0, 5))]
            row += [draw(finite) for _ in range(draw(st.integers(0, 6)))]
            gencost.append(row)
    return {"bus": bus, "gen": gen, "branch": branch, "gencost": gencost}


def _fmt(v) -> str:
    return str(v) if type(v) is int else repr(float(v))


@st.composite
def case_text(draw):
    """Strategy T: (text, rows, extras, casename, baseMVA) following round-1 Q4 syntax."""
    rows = draw(network_rows())
    casename = draw(identifier)
    base = draw(finite)
    names = draw(st.lists(st.sampled_from(EXTRA_NAMES), unique=True, max_size=3))
    extras = {n: draw(st.lists(st.text(ALNUM, min_size=1, max_size=8), min_size=1, max_size=4))
              for n in names}
    header = draw(st.lists(st.text(ALNUM, min_size=1, max_size=20), max_size=3))
    out = [f"function mpc = {casename}"]
    out += [f"% {h}" for h in header]
    out += ["mpc.version = '2';", f"mpc.baseMVA = {_fmt(base)};"]
    for section in ("bus", "gen", "branch", "gencost"):
        if section != "bus" and not rows[section] and draw(st.booleans()):
            continue  # absent section is valid (SPEC-PARSER-07)
        out.append(f"mpc.{section} = [")
        out += ["\t" + "\t".join(_fmt(v) for v in r) + ";" for r in rows[section]]
        out.append("];")
    for n, elems in extras.items():
        out.append(f"mpc.{n} = {{")
        out += [f"\t'{e}';" for e in elems]
        out.append("};")
    return "\n".join(out) + "\n", rows, extras, casename, base


def assert_rows_mapped(net: MATNetwork, rows: dict) -> None:
    """Field-by-field check of a parsed network against raw rows (SPEC-PARSER-02..05)."""
    assert len(net.buses) == len(rows["bus"])
    for i, (b, r) in enumerate(zip(net.buses, rows["bus"])):
        _cmp_row("bus", i, BUS_COLS, b, r)
        for k, col in enumerate(BUS_OPT):
            v = getattr(b, col)
            if 13 + k < len(r):
                assert v is not None and feq(float(v), float(r[13 + k])), f"bus[{i}].{col}"
            else:
                assert v is None, f"bus[{i}].{col} should be None, got {v!r}"
    assert len(net.generators) == len(rows["gen"])
    for i, (g, r) in enumerate(zip(net.generators, rows["gen"])):
        _cmp_row("gen", i, GEN_COLS, g, r)
        _cmp_float_list(f"gen[{i}].extra", g.extra, r[10:])
    assert len(net.branches) == len(rows["branch"])
    for i, (br, r) in enumerate(zip(net.branches, rows["branch"])):
        _cmp_row("branch", i, BRANCH_COLS, br, r)
        _cmp_float_list(f"branch[{i}].extra", br.extra, r[13:])
    assert len(net.gencost) == len(rows["gencost"])
    for i, (gc, r) in enumerate(zip(net.gencost, rows["gencost"])):
        _cmp_row("gencost", i, GENCOST_COLS, gc, r)
        _cmp_float_list(f"gencost[{i}].coeffs", gc.coeffs, r[4:])


def _cmp_row(section, idx, cols, obj, row):
    for k, col in enumerate(cols):
        v, expected = getattr(obj, col), row[k]
        where = f"{section}[{idx}].{col}"
        if col in INT_FIELDS[section]:
            assert _is_int(v), f"{where}: {v!r} is not int"
            assert v == int(expected), f"{where}: {v!r} != {expected!r}"
        else:
            assert feq(float(v), float(expected)), f"{where}: {v!r} !~ {expected!r}"


def network_from_rows(rows: dict, casename: str, base: float, header: str) -> MATNetwork:
    """Strategy N helper: build a MATNetwork from raw rows (extra_sections={})."""
    def bus(r):
        opt = list(r[13:]) + [None] * (4 - len(r[13:]))
        return Bus(*r[:13], lam_P=opt[0], lam_Q=opt[1], mu_Vmax=opt[2], mu_Vmin=opt[3])

    return MATNetwork(
        casename=casename,
        version="2",
        baseMVA=base,
        buses=[bus(r) for r in rows["bus"]],
        generators=[Generator(*r[:10], extra=list(r[10:])) for r in rows["gen"]],
        branches=[Branch(*r[:13], extra=list(r[13:])) for r in rows["branch"]],
        gencost=[GenCost(*r[:4], coeffs=list(r[4:])) for r in rows["gencost"]],
        header_comments=header,
        extra_sections={},
    )


@st.composite
def constructed_network(draw):
    """Strategy N: (MATNetwork, rows). header is '' or contains a foreign function line."""
    rows = draw(network_rows())
    casename = draw(identifier)
    base = draw(finite)
    if draw(st.booleans()):
        header = ""
    else:
        other = draw(identifier)
        assume(other != casename)
        header = f"function mpc = {other}\n% QA constructed header\n"
    return network_from_rows(rows, casename, base, header), rows
