"""Spec-derived tests for the MATPOWER case reader/writer.

Spec: specs/parser_matpower.md (approved 2026-10-05). Every test cites its item ID(s)
with @pytest.mark.spec. Public interface only (interfaces.md):
parse_matpower, write_matpower, MATNetwork and its record types.
No network, LLM or ExaGO access; all data is synthetic (tests_spec/data/ or inline).
"""
from __future__ import annotations

import copy
import dataclasses
import math
import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import given

from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.exago_parsers.matpower_writer import write_matpower

from parser_helpers import (
    BUS_OPT,
    DATA_DIR,
    VALID_FIXTURES,
    Workdir,
    assert_data_equiv,
    assert_full_equiv,
    assert_rows_mapped,
    assert_same_elements,
    case_text,
    check_text_rules,
    constructed_network,
    feq,
    find_blocks,
    fixture_text,
    must_replace,
    prev_statement_end,
    put_case,
    raw_tokens,
    rows_from_text,
    section_start,
    write_read,
)

spec = pytest.mark.spec

# Anchors in tests_spec/data/case_min.m (tab separated).
MIN_BUS1 = "\t1\t3\t0\t0\t0\t0\t1\t1.0\t0\t138\t1\t1.1\t0.9;"
MIN_BUS2 = "\t2\t1\t50\t10\t0\t0\t1\t1.0\t0\t138\t1\t1.1\t0.9;"
MIN_GEN = "\t1\t40\t0\t30\t-30\t1.0\t100\t1\t100\t0;"
MIN_BRANCH = "\t1\t2\t0.01\t0.1\t0.02\t250\t250\t250\t0\t0\t1\t-360\t360;"
MIN_GENCOST = "\t2\t0\t0\t3\t0.01\t20\t100;"


@pytest.fixture
def wd(tmp_path) -> Workdir:
    return Workdir(tmp_path)


def load(wd: Workdir, text: str, newline: str = "\n"):
    """Write `text` to a fresh file and parse it."""
    p = put_case(wd.root, wd.path("in").name, text, newline)
    return parse_matpower(p)


def fixture_path(name: str) -> Path:
    return DATA_DIR / name


def R(net, wd: Workdir):
    """R(x) = parse(write(x)); returns (written_text, network)."""
    return write_read(net, wd.path("rt"))


# ===========================================================================
# SPEC-PARSER-01 — core scalars and casename
# ===========================================================================
@spec("SPEC-PARSER-01")
def test_core_scalars_are_read():
    net = parse_matpower(fixture_path("case_min.m"))
    assert net.version == "2"
    assert net.baseMVA == 100.0
    assert isinstance(net.baseMVA, float)


@spec("SPEC-PARSER-01")
def test_casename_comes_from_function_line_not_file_name(wd):
    text = must_replace(fixture_text("case_min.m"), "function mpc = case_min",
                        "function mpc = case_x")
    p = put_case(wd.root, "completely_different_name.m", text)
    assert parse_matpower(p).casename == "case_x"


@spec("SPEC-PARSER-01")
def test_missing_version_defaults_to_2(wd):
    text = must_replace(fixture_text("case_min.m"), "mpc.version = '2';\n", "")
    assert load(wd, text).version == "2"


# ===========================================================================
# SPEC-PARSER-02 — bus matrix (13–17 columns)
# ===========================================================================
@spec("SPEC-PARSER-02")
def test_bus_13_columns_map_in_order_with_int_columns_and_no_results():
    text = fixture_text("case_min.m")
    net = parse_matpower(fixture_path("case_min.m"))
    assert_rows_mapped(net, {
        "bus": rows_from_text(text, "bus"), "gen": rows_from_text(text, "gen"),
        "branch": rows_from_text(text, "branch"), "gencost": rows_from_text(text, "gencost"),
    })
    b = net.buses[1]
    assert (b.bus_i, b.type, b.area, b.zone) == (2, 1, 1, 1)
    assert (b.Pd, b.Qd, b.baseKV, b.Vmax, b.Vmin) == (50.0, 10.0, 138.0, 1.1, 0.9)
    assert all(getattr(b, c) is None for c in BUS_OPT)


@spec("SPEC-PARSER-02")
@pytest.mark.parametrize("k", [1, 2, 3, 4], ids=lambda k: f"{13 + k}cols")
def test_bus_result_columns_fill_in_order_rest_none(wd, k):
    values = [11.5, -2.25, 0.5, 0.125][:k]
    row = MIN_BUS2[:-1] + "".join(f"\t{v}" for v in values) + ";"
    net = load(wd, must_replace(fixture_text("case_min.m"), MIN_BUS2, row))
    b = net.buses[1]
    for i, col in enumerate(BUS_OPT):
        if i < k:
            assert feq(getattr(b, col), values[i]), col
        else:
            assert getattr(b, col) is None, col
    assert all(getattr(net.buses[0], c) is None for c in BUS_OPT)


@spec("SPEC-PARSER-02")
def test_bus_17_columns_fixture_maps_all_results():
    text = fixture_text("case_wide.m")
    net = parse_matpower(fixture_path("case_wide.m"))
    assert_rows_mapped(net, {s: rows_from_text(text, s)
                             for s in ("bus", "gen", "branch", "gencost")})
    assert feq(net.buses[0].lam_P, 1500.123456789)
    assert feq(net.buses[1].lam_Q, -3.5e-05)


# ===========================================================================
# SPEC-PARSER-03 / -04 / -05 — gen, branch, gencost
# ===========================================================================
@spec("SPEC-PARSER-03")
def test_gen_10_columns_has_empty_extra():
    net = parse_matpower(fixture_path("case_min.m"))
    g = net.generators[0]
    assert (g.bus, g.status) == (1, 1)
    assert (g.Pg, g.Qmax, g.Qmin, g.Vg, g.mBase, g.Pmax, g.Pmin) == (
        40.0, 30.0, -30.0, 1.0, 100.0, 100.0, 0.0)
    assert g.extra == []


@spec("SPEC-PARSER-03")
def test_gen_21_columns_keeps_all_11_extra_columns_in_order():
    net = parse_matpower(fixture_path("case_wide.m"))
    g = net.generators[1]
    expected = [1.5, 2.5, -10, 10, -5, 5, 12.3456789, 30, 60, 7.5, 0.33]
    assert len(g.extra) == 11
    assert all(feq(a, b) for a, b in zip(g.extra, expected))


@spec("SPEC-PARSER-03")
def test_gen_wider_than_21_columns_keeps_every_extra_column(wd):
    row = MIN_GEN[:-1] + "".join(f"\t{i}" for i in range(1, 16)) + ";"
    net = load(wd, must_replace(fixture_text("case_min.m"), MIN_GEN, row))
    assert [float(v) for v in net.generators[0].extra] == [float(i) for i in range(1, 16)]


@spec("SPEC-PARSER-04")
def test_branch_13_columns_maps_in_order():
    net = parse_matpower(fixture_path("case_min.m"))
    br = net.branches[0]
    assert (br.fbus, br.tbus, br.status) == (1, 2, 1)
    assert all(type(v) is int for v in (br.fbus, br.tbus, br.status))
    assert (br.r, br.x, br.b, br.rateA, br.ratio, br.angle, br.angmin, br.angmax) == (
        0.01, 0.1, 0.02, 250.0, 0.0, 0.0, -360.0, 360.0)
    assert br.extra == []


@spec("SPEC-PARSER-04")
def test_branch_extra_columns_kept_in_order():
    net = parse_matpower(fixture_path("case_wide.m"))
    expected = [30.7281, -0.5859, -30.5547, -13.6876]
    assert len(net.branches[1].extra) == 4
    assert all(feq(a, b) for a, b in zip(net.branches[1].extra, expected))


@spec("SPEC-PARSER-05")
def test_gencost_polynomial_row():
    gc = parse_matpower(fixture_path("case_min.m")).gencost[0]
    assert (gc.model, gc.ncost) == (2, 3)
    assert type(gc.model) is int and type(gc.ncost) is int
    assert (gc.startup, gc.shutdown) == (0.0, 0.0)
    assert [float(c) for c in gc.coeffs] == [0.01, 20.0, 100.0]


@spec("SPEC-PARSER-05")
def test_gencost_4_columns_gives_empty_coeffs(wd):
    net = load(wd, must_replace(fixture_text("case_min.m"), MIN_GENCOST, "\t2\t0\t0\t0;"))
    assert net.gencost[0].coeffs == []


@spec("SPEC-PARSER-05")
@pytest.mark.parametrize("row,coeffs", [
    ("\t1\t0\t0\t3\t0\t0\t50\t1000\t100\t2500;", [0, 0, 50, 1000, 100, 2500]),  # model 1
    ("\t2\t0\t0\t4\t5\t7;", [5, 7]),                       # ncost does not match coeffs
    ("\t1\t0\t0\t1\t0\t0\t50\t1000;", [0, 0, 50, 1000]),   # model 1, ncost mismatch
])
def test_gencost_rows_read_as_numbers_without_ncost_check(wd, row, coeffs):
    net = load(wd, must_replace(fixture_text("case_min.m"), MIN_GENCOST, row))
    assert [float(c) for c in net.gencost[0].coeffs] == [float(c) for c in coeffs]


# ===========================================================================
# SPEC-PARSER-06 — order and identifiers, no semantic checks
# ===========================================================================
CASE_IDS = """function mpc = case_ids
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
\t1001\t3\t0\t0\t0\t0\t1\t1\t0\t138\t1\t1.1\t0.9;
\t5\t1\t10\t0\t0\t0\t1\t1\t0\t138\t1\t1.1\t0.9;
\t1\t1\t20\t0\t0\t0\t1\t1\t0\t138\t1\t1.1\t0.9;
\t5\t2\t30\t0\t0\t0\t1\t1\t0\t138\t1\t1.1\t0.9;
];
mpc.gen = [
\t1001\t10\t0\t10\t-10\t1\t100\t1\t50\t0;
\t1001\t20\t0\t10\t-10\t1\t100\t1\t50\t0;
\t9999\t30\t0\t10\t-10\t1\t100\t1\t50\t0;
];
mpc.branch = [
\t5\t7777\t0.01\t0.1\t0\t0\t0\t0\t0\t0\t1\t-360\t360;
\t1001\t5\t0.01\t0.1\t0\t0\t0\t0\t0\t0\t1\t-360\t360;
];
mpc.gencost = [
\t2\t0\t0\t2\t1\t0;
\t2\t0\t0\t2\t2\t0;
\t2\t0\t0\t2\t3\t0;
\t2\t0\t0\t2\t4\t0;
];
"""


@spec("SPEC-PARSER-06")
def test_order_ids_kept_and_no_semantic_checks(wd):
    net = load(wd, CASE_IDS)
    assert [b.bus_i for b in net.buses] == [1001, 5, 1, 5]          # duplicate kept
    assert [b.Pd for b in net.buses] == [0.0, 10.0, 20.0, 30.0]     # file order
    assert [g.bus for g in net.generators] == [1001, 1001, 9999]    # dangling ref kept
    assert [g.Pg for g in net.generators] == [10.0, 20.0, 30.0]
    assert [(br.fbus, br.tbus) for br in net.branches] == [(5, 7777), (1001, 5)]
    assert [gc.coeffs[0] for gc in net.gencost] == [1.0, 2.0, 3.0, 4.0]  # 4 rows, 3 gens


# ===========================================================================
# SPEC-PARSER-07 — empty and absent sections
# ===========================================================================
@spec("SPEC-PARSER-07")
@pytest.mark.parametrize("section,attr", [
    ("gen", "generators"), ("branch", "branches"), ("gencost", "gencost"), ("bus", "buses"),
])
@pytest.mark.parametrize("form", ["one_line", "multi_line", "multi_line_comment"])
def test_present_but_empty_section_gives_empty_list(wd, section, attr, form):
    text = fixture_text("case_min.m")
    blk = find_blocks(text)[section]
    empty = {
        "one_line": f"mpc.{section} = [];",
        "multi_line": f"mpc.{section} = [\n];",
        "multi_line_comment": f"mpc.{section} = [\n\n\t% no rows\n];",
    }[form]
    text = text[:blk.start] + empty + text[blk.end:]
    net = load(wd, text)
    assert getattr(net, attr) == []


@spec("SPEC-PARSER-07")
def test_absent_gen_branch_gencost_give_empty_lists():
    net = parse_matpower(fixture_path("case_nogen.m"))
    assert net.generators == [] and net.branches == [] and net.gencost == []
    assert len(net.buses) == 1


@spec("SPEC-PARSER-07")
def test_absent_gencost_alone_gives_empty_list(wd):
    text = fixture_text("case_min.m")
    blk = find_blocks(text)["gencost"]
    net = load(wd, text[:blk.start] + text[blk.end:])
    assert net.gencost == [] and len(net.generators) == 1


# ===========================================================================
# SPEC-PARSER-08 — extra sections
# ===========================================================================
@spec("SPEC-PARSER-08")
def test_extra_section_keys_without_prefix_and_no_core_keys():
    net = parse_matpower(fixture_path("case_extra.m"))
    assert set(net.extra_sections) == {"gentype", "genfuel", "bus_name", "areas"}
    core = {"version", "baseMVA", "bus", "gen", "branch", "gencost"}
    assert not core & set(net.extra_sections)
    assert not any(k.startswith("mpc.") for k in net.extra_sections)


@spec("SPEC-PARSER-08")
def test_extra_sections_stable_from_second_round_trip(wd):
    _, r1 = R(parse_matpower(fixture_path("case_extra.m")), wd)
    _, r2 = R(r1, wd)
    assert r2.extra_sections == r1.extra_sections


@spec("SPEC-PARSER-08")
@pytest.mark.parametrize("name", ["gentype", "genfuel", "bus_name", "areas"])
def test_extra_section_elements_survive_in_written_text(wd, name):
    src = fixture_text("case_extra.m")
    out, _ = R(parse_matpower(fixture_path("case_extra.m")), wd)
    assert_same_elements(src, out, name)


# ===========================================================================
# SPEC-PARSER-09 — float fidelity, Inf/NaN
# ===========================================================================
@spec("SPEC-PARSER-09")
def test_float_fidelity_on_wide_case(wd):
    first = parse_matpower(fixture_path("case_wide.m"))
    _, again = R(first, wd)
    assert_data_equiv(again, first)


SPELLINGS = [("inf", math.inf, "Inf"), ("Inf", math.inf, "Inf"), ("INF", math.inf, "Inf"),
             ("-inf", -math.inf, "-Inf"), ("-Inf", -math.inf, "-Inf"),
             ("-INF", -math.inf, "-Inf"),
             ("nan", math.nan, "NaN"), ("NaN", math.nan, "NaN"), ("NAN", math.nan, "NaN")]


@spec("SPEC-PARSER-09")
@pytest.mark.parametrize("spelling,value,canonical", SPELLINGS, ids=[s[0] for s in SPELLINGS])
@pytest.mark.parametrize("where", ["gen.Qmax", "bus.lam_P"])
def test_inf_nan_any_case_read_round_trip_and_written_canonically(
        wd, spelling, value, canonical, where):
    text = fixture_text("case_min.m")
    if where == "gen.Qmax":
        text = must_replace(text, MIN_GEN, "\t1\t40\t0\t" + spelling + "\t-30\t1.0\t100\t1\t100\t0;")
        section, col, get = "gen", 3, lambda n: n.generators[0].Qmax
    else:
        text = must_replace(text, MIN_BUS2, MIN_BUS2[:-1] + "\t" + spelling + ";")
        section, col, get = "bus", 13, lambda n: n.buses[1].lam_P
    net = load(wd, text)
    assert feq(get(net), value)
    out, again = R(net, wd)
    assert feq(get(again), value)
    tokens = raw_tokens(out, section)
    row = tokens[0] if section == "gen" else tokens[1]
    assert row[col] == canonical


# ===========================================================================
# SPEC-PARSER-10 — accepted syntax, CRLF
# ===========================================================================
@spec("SPEC-PARSER-10")
def test_syntax_variants_parse_like_plain_case():
    varied = parse_matpower(fixture_path("case_syntax.m"))
    plain = parse_matpower(fixture_path("case_plain.m"))
    assert_data_equiv(varied, plain)


@spec("SPEC-PARSER-10")
def test_integral_floats_in_integer_columns_read_as_int():
    net = parse_matpower(fixture_path("case_syntax.m"))
    b = net.buses[2]
    assert (b.bus_i, b.type, b.area) == (3, 2, 1)
    assert all(type(v) is int for v in (b.bus_i, b.type, b.area, b.zone))
    assert type(net.generators[0].status) is int and net.generators[0].status == 1
    assert (net.branches[1].fbus, net.branches[1].tbus) == (2, 3)
    assert type(net.gencost[1].model) is int and net.gencost[1].ncost == 3


@spec("SPEC-PARSER-10")
def test_scientific_and_negative_numbers():
    net = parse_matpower(fixture_path("case_syntax.m"))
    assert feq(net.buses[2].Qd, 0.0015)
    assert feq(net.buses[2].baseKV, 138.0)
    assert feq(net.buses[2].Pd, -5.0)
    assert feq(net.branches[0].r, 0.01) and feq(net.branches[0].b, 0.02)


@spec("SPEC-PARSER-10")
@pytest.mark.parametrize("name", ["case_extra.m", "case_syntax.m", "case_wide.m"])
def test_crlf_file_reads_exactly_like_lf_file(wd, name):
    text = fixture_text(name)
    lf = load(wd, text, "\n")
    crlf = load(wd, text, "\r\n")
    assert_full_equiv(crlf, lf)


# ===========================================================================
# SPEC-PARSER-11 — round trip, source unchanged
# ===========================================================================
@spec("SPEC-PARSER-11")
@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_round_trip_data_equivalent_then_fully_stable(wd, name):
    first = parse_matpower(fixture_path(name))
    _, r1 = R(first, wd)
    assert_data_equiv(r1, first)
    _, r2 = R(r1, wd)
    assert_full_equiv(r2, r1)


@spec("SPEC-PARSER-11")
@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_source_file_bytes_unchanged(wd, name):
    src = put_case(wd.root, "source.m", fixture_text(name))
    before = src.read_bytes()
    net = parse_matpower(src)
    write_matpower(net, wd.path())
    assert src.read_bytes() == before


# ===========================================================================
# SPEC-PARSER-12 — textual rules of the written file
# ===========================================================================
@spec("SPEC-PARSER-12")
@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_written_file_meets_textual_rules(wd, name):
    net = parse_matpower(fixture_path(name))
    out, _ = R(net, wd)
    check_text_rules(out, net)


@spec("SPEC-PARSER-12")
def test_cell_arrays_closed_at_first_terminator_with_all_elements(wd):
    src = fixture_text("case_extra.m")
    out, _ = R(parse_matpower(fixture_path("case_extra.m")), wd)
    blocks = find_blocks(out)
    for name in ("genfuel", "gentype", "bus_name"):
        assert blocks[name].kind == "{"
        assert_same_elements(src, out, name)
    assert blocks["areas"].kind == "["


@spec("SPEC-PARSER-12")
def test_crlf_input_is_written_with_lf_only(wd):
    net = load(wd, fixture_text("case_extra.m"), "\r\n")
    out, _ = R(net, wd)
    assert "\r" not in out


@spec("SPEC-PARSER-12")
def test_empty_header_writes_function_line_first(wd):
    net = dataclasses.replace(parse_matpower(fixture_path("case_min.m")), header_comments="")
    out, _ = R(net, wd)
    assert out.splitlines()[0].strip() == "function mpc = case_min"
    check_text_rules(out, net)


@spec("SPEC-PARSER-12")
def test_casename_wins_over_function_name_in_header(wd):
    net = dataclasses.replace(
        parse_matpower(fixture_path("case_min.m")),
        header_comments="function mpc = some_other_name\n% kept comment\n",
    )
    out, again = R(net, wd)
    check_text_rules(out, net)             # exactly one function line, naming case_min
    assert "some_other_name" not in out
    assert again.casename == "case_min"


@spec("SPEC-PARSER-12")
def test_casename_change_is_written(wd):
    net = dataclasses.replace(parse_matpower(fixture_path("case_extra.m")), casename="renamed")
    out, again = R(net, wd)
    check_text_rules(out, net)
    assert again.casename == "renamed"


# ===========================================================================
# SPEC-PARSER-13 — idempotent writer
# ===========================================================================
@spec("SPEC-PARSER-13")
@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_writer_is_idempotent(wd, name):
    p1 = wd.path()
    write_matpower(parse_matpower(fixture_path(name)), p1)
    p2 = wd.path()
    write_matpower(parse_matpower(p1), p2)
    assert p2.read_bytes() == p1.read_bytes()


@spec("SPEC-PARSER-13")
def test_file_does_not_grow_over_repeated_round_trips(wd):
    path = wd.path()
    write_matpower(parse_matpower(fixture_path("case_extra.m")), path)
    sizes = []
    for _ in range(4):
        nxt = wd.path()
        write_matpower(parse_matpower(path), nxt)
        sizes.append(len(nxt.read_bytes()))
        path = nxt
    assert len(set(sizes)) == 1


# ===========================================================================
# SPEC-PARSER-14 — header and section comments
# ===========================================================================
@spec("SPEC-PARSER-14")
@pytest.mark.parametrize("name,marker", [
    ("bus", "QA-COMMENT-BUS"), ("gen", "QA-COMMENT-GEN"), ("branch", "QA-COMMENT-BRANCH"),
    ("gencost", "QA-COMMENT-GENCOST"), ("genfuel", "QA-COMMENT-GENFUEL"),
])
def test_comment_above_section_stays_with_section(wd, name, marker):
    out, _ = R(parse_matpower(fixture_path("case_extra.m")), wd)
    sec = section_start(out, name)
    pos = out.find(marker)
    assert pos != -1, f"{marker} dropped"
    assert prev_statement_end(out, sec) <= pos < sec, f"{marker} not directly above mpc.{name}"


@spec("SPEC-PARSER-14")
def test_bus_comment_follows_basemva_line(wd):
    out, _ = R(parse_matpower(fixture_path("case_extra.m")), wd)
    assert section_start(out, "baseMVA") < out.find("QA-COMMENT-BUS") < section_start(out, "bus")


@spec("SPEC-PARSER-14")
def test_header_comments_preserved(wd):
    first = parse_matpower(fixture_path("case_extra.m"))
    out, r1 = R(first, wd)
    first_mpc = re.search(r"^[ \t]*mpc\.", out, re.M).start()
    assert -1 < out.find("QA-HEADER-LINE") < first_mpc
    assert -1 < out.find("second header line") < first_mpc
    _, r2 = R(r1, wd)
    assert r2.header_comments == r1.header_comments


# ===========================================================================
# SPEC-PARSER-15 — path handling
# ===========================================================================
@spec("SPEC-PARSER-15")
def test_str_and_path_give_same_result(wd):
    p = fixture_path("case_extra.m")
    assert_full_equiv(parse_matpower(str(p)), parse_matpower(p))
    net = parse_matpower(p)
    a, b = wd.path("str"), wd.path("path")
    write_matpower(net, str(a))
    write_matpower(net, b)
    assert a.read_bytes() == b.read_bytes()


@spec("SPEC-PARSER-15")
@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
def test_writer_creates_missing_parent_directories(wd, as_str):
    target = wd.root / "new" / "nested" / "dir" / "out.m"
    write_matpower(parse_matpower(fixture_path("case_min.m")), str(target) if as_str else target)
    assert target.is_file()
    assert_data_equiv(parse_matpower(target), parse_matpower(fixture_path("case_min.m")))


@spec("SPEC-PARSER-15")
def test_writer_overwrites_existing_file(wd):
    target = wd.path()
    write_matpower(parse_matpower(fixture_path("case_extra.m")), target)
    small = parse_matpower(fixture_path("case_min.m"))
    write_matpower(small, target)
    assert_data_equiv(parse_matpower(target), small)


# ===========================================================================
# SPEC-PARSER-16 — missing file
# ===========================================================================
@spec("SPEC-PARSER-16")
@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
def test_missing_file_raises_file_not_found(tmp_path, as_str):
    missing = tmp_path / "does_not_exist.m"
    with pytest.raises(FileNotFoundError):
        parse_matpower(str(missing) if as_str else missing)


# ===========================================================================
# SPEC-PARSER-17 / -18 — malformed input raises ValueError naming the section
# ===========================================================================
def _drop_block(text: str, name: str) -> str:
    blk = find_blocks(text)[name]
    return text[:blk.start] + text[blk.end:]


def _unterminated_gencost(text: str) -> str:
    blk = find_blocks(text)["gencost"]
    return text[:blk.end - 2] + text[blk.end:]   # remove the final "];"


BUS2_12 = "\t2\t1\t50\t10\t0\t0\t1\t1.0\t0\t138\t1\t1.1;"
BUS2_18 = MIN_BUS2[:-1] + "\t1\t2\t3\t4\t5;"

MALFORMED = [
    # (id, transform(text) -> text, required message substring or None)
    ("a_no_function_line", lambda t: must_replace(t, "function mpc = case_min\n", ""), None),
    ("b_no_baseMVA", lambda t: must_replace(t, "mpc.baseMVA = 100;\n", ""), "baseMVA"),
    ("c_version_1", lambda t: must_replace(t, "mpc.version = '2';", "mpc.version = '1';"), None),
    ("c_version_3", lambda t: must_replace(t, "mpc.version = '2';", "mpc.version = '3';"), None),
    ("d_no_bus", lambda t: _drop_block(t, "bus"), "bus"),
    ("e_two_bus_rows_one_line",
     lambda t: must_replace(t, MIN_BUS1 + "\n" + MIN_BUS2, MIN_BUS1 + " " + MIN_BUS2.strip()),
     "bus"),
    ("e_two_gencost_rows_one_line",
     lambda t: must_replace(t, MIN_GENCOST, MIN_GENCOST + " " + MIN_GENCOST.strip()), "gencost"),
    ("f_commas_in_gen_row",
     lambda t: must_replace(t, MIN_GEN, "\t1,40,0,30,-30,1.0,100,1,100,0;"), "gen"),
    ("f_commas_in_bus_row",
     lambda t: must_replace(t, MIN_BUS1, MIN_BUS1.replace("\t", ", ").lstrip(", ")), "bus"),
    ("g_continuation_in_branch_row",
     lambda t: must_replace(t, MIN_BRANCH,
                            "\t1\t2\t0.01\t0.1\t0.02\t250 ...\n\t250\t250\t0\t0\t1\t-360\t360;"),
     "branch"),
    ("h_bus_12_columns", lambda t: must_replace(t, MIN_BUS2, BUS2_12), "bus"),
    ("h_bus_18_columns", lambda t: must_replace(t, MIN_BUS2, BUS2_18), "bus"),
    ("i_gen_9_columns",
     lambda t: must_replace(t, MIN_GEN, "\t1\t40\t0\t30\t-30\t1.0\t100\t1\t100;"), "gen"),
    ("i_branch_12_columns",
     lambda t: must_replace(t, MIN_BRANCH, "\t1\t2\t0.01\t0.1\t0.02\t250\t250\t250\t0\t0\t1\t-360;"),
     "branch"),
    ("i_gencost_3_columns", lambda t: must_replace(t, MIN_GENCOST, "\t2\t0\t0;"), "gencost"),
    ("j_non_numeric_in_bus",
     lambda t: must_replace(t, MIN_BUS2, MIN_BUS2.replace("\t50\t", "\tabc\t")), "bus"),
    ("j_non_numeric_in_branch",
     lambda t: must_replace(t, MIN_BRANCH, MIN_BRANCH.replace("\t0.1\t", "\tx0.1\t")), "branch"),
    ("j_non_numeric_in_gencost",
     lambda t: must_replace(t, MIN_GENCOST, "\t2\t0\t0\t3\t0.01\tfoo\t100;"), "gencost"),
    ("k_bus_type_1.5",
     lambda t: must_replace(t, MIN_BUS2, MIN_BUS2.replace("\t2\t1\t", "\t2\t1.5\t", 1)), "bus"),
    ("k_gen_status_NaN",
     lambda t: must_replace(t, MIN_GEN, "\t1\t40\t0\t30\t-30\t1.0\t100\tNaN\t100\t0;"), "gen"),
    ("k_branch_fbus_Inf",
     lambda t: must_replace(t, MIN_BRANCH, MIN_BRANCH.replace("\t1\t2\t", "\tInf\t2\t", 1)),
     "branch"),
    ("k_gencost_ncost_2.5",
     lambda t: must_replace(t, MIN_GENCOST, "\t2\t0\t0\t2.5\t0.01\t20\t100;"), "gencost"),
    ("l_unterminated_gencost", _unterminated_gencost, "gencost"),
    ("l_unterminated_cell_array",
     lambda t: t + "mpc.genfuel = {\n\t'coal';\n", "genfuel"),
    ("l_unterminated_numeric_extra",
     lambda t: t + "mpc.areas = [\n\t1\t1;\n", "areas"),
]


@spec("SPEC-PARSER-17", "SPEC-PARSER-18")
@pytest.mark.parametrize("transform,needle", [(m[1], m[2]) for m in MALFORMED],
                         ids=[m[0] for m in MALFORMED])
def test_malformed_input_raises_value_error_naming_section(wd, transform, needle):
    text = transform(fixture_text("case_min.m"))
    path = put_case(wd.root, "bad.m", text)
    with pytest.raises(ValueError) as exc:
        parse_matpower(path)
    if needle is not None:
        assert needle in str(exc.value), f"message {str(exc.value)!r} lacks {needle!r}"


@spec("SPEC-PARSER-17")
def test_empty_file_raises_value_error(wd):
    path = put_case(wd.root, "empty.m", "")
    with pytest.raises(ValueError):
        parse_matpower(path)


# ===========================================================================
# SPEC-PARSER-19 — non-core sections are opaque
# ===========================================================================
@spec("SPEC-PARSER-19")
@pytest.mark.parametrize("extra,key", [
    ("mpc.areas = [\n\t1, 2;\n\t3 abc;\n];\n", "areas"),
    ("mpc.dcline = [\n\t1\t2\t3; 4\t5\t6;\n\tnot numbers at all\n];\n", "dcline"),
    ("mpc.genfuel = {'coal', 'ng'};\n", "genfuel"),
    ("mpc.custom = [\n\t1 ...\n\t2;\n];\n", "custom"),
])
def test_opaque_extra_sections_are_not_validated(wd, extra, key):
    net = load(wd, fixture_text("case_min.m") + extra)
    assert key in net.extra_sections
    assert len(net.buses) == 2


# ===========================================================================
# SPEC-PARSER-20 — deterministic reading
# ===========================================================================
@spec("SPEC-PARSER-20")
@pytest.mark.parametrize("name", VALID_FIXTURES)
def test_reading_is_deterministic(name):
    assert_full_equiv(parse_matpower(fixture_path(name)), parse_matpower(fixture_path(name)))


# ===========================================================================
# SPEC-PARSER-21 — no shared state, writer does not mutate input
# ===========================================================================
@spec("SPEC-PARSER-21")
def test_mutating_result_does_not_affect_next_parse():
    p = fixture_path("case_wide.m")
    n = parse_matpower(p)
    original_pd = n.buses[1].Pd
    n.buses[1].Pd += 1
    n.generators[1].extra.append(99.0)
    n.buses.pop()
    n.extra_sections["qa_injected"] = "x"
    again = parse_matpower(p)
    assert again.buses[1].Pd == original_pd
    assert len(again.generators[1].extra) == 11
    assert len(again.buses) == 3
    assert "qa_injected" not in again.extra_sections


@spec("SPEC-PARSER-21")
@pytest.mark.parametrize("name", ["case_wide.m", "case_extra.m", "case_mixed.m"])
def test_writer_does_not_mutate_its_input(wd, name):
    n = parse_matpower(fixture_path(name))
    c = copy.deepcopy(n)
    write_matpower(n, wd.path())
    assert n == c


# ===========================================================================
# SPEC-PARSER-22 — Hypothesis invariants
# ===========================================================================
@spec("SPEC-PARSER-22")
@given(case=case_text())
def test_property_text_strategy_round_trip(case):
    text, rows, extras, casename, base = case
    with tempfile.TemporaryDirectory() as d:
        wd = Workdir(Path(d))
        src = put_case(wd.root, "gen.m", text)
        first = parse_matpower(src)
        # reader maps generated rows exactly (SPEC-PARSER-01..05)
        assert first.casename == casename and feq(first.baseMVA, base)
        assert_rows_mapped(first, rows)
        assert set(first.extra_sections) == set(extras)
        # round trip, textual rules, idempotence, extra elements
        out1, r1 = write_read(first, wd.path())
        assert_data_equiv(r1, first)
        check_text_rules(out1, first)
        for name in extras:
            assert_same_elements(text, out1, name)
        out2, r2 = write_read(r1, wd.path())
        assert_full_equiv(r2, r1)
        assert out2 == out1


@spec("SPEC-PARSER-22")
@given(item=constructed_network())
def test_property_constructed_network_round_trip(item):
    net, rows = item
    with tempfile.TemporaryDirectory() as d:
        out, again = write_read(net, Workdir(Path(d)).path())
        check_text_rules(out, net)
        assert_data_equiv(again, net)
        assert_rows_mapped(again, rows)
        if net.header_comments == "":
            assert out.splitlines()[0].strip() == f"function mpc = {net.casename}"


# ===========================================================================
# SPEC-PARSER-23 — mixed row widths round-trip per row
# ===========================================================================
@spec("SPEC-PARSER-23")
def test_mixed_widths_read_per_row():
    text = fixture_text("case_mixed.m")
    net = parse_matpower(fixture_path("case_mixed.m"))
    assert_rows_mapped(net, {s: rows_from_text(text, s)
                             for s in ("bus", "gen", "branch", "gencost")})
    set_counts = [sum(getattr(b, c) is not None for c in BUS_OPT) for b in net.buses]
    assert set_counts == [0, 2, 4]
    assert [len(g.extra) for g in net.generators] == [0, 1, 11]
    assert [len(br.extra) for br in net.branches] == [0, 4]
    assert [len(gc.coeffs) for gc in net.gencost] == [0, 2, 3]


@spec("SPEC-PARSER-23")
def test_mixed_widths_round_trip_without_padding(wd):
    first = parse_matpower(fixture_path("case_mixed.m"))
    out, again = R(first, wd)
    assert_data_equiv(again, first)          # None stays None, extra lengths kept
    widths = {s: [len(r) for r in raw_tokens(out, s)] for s in ("bus", "gen", "branch", "gencost")}
    assert widths == {"bus": [13, 15, 17], "gen": [10, 11, 21],
                      "branch": [13, 17], "gencost": [4, 6, 7]}


@spec("SPEC-PARSER-23", "SPEC-PARSER-13")
def test_mixed_widths_writer_idempotent(wd):
    p1 = wd.path()
    write_matpower(parse_matpower(fixture_path("case_mixed.m")), p1)
    p2 = wd.path()
    write_matpower(parse_matpower(p1), p2)
    assert p1.read_bytes() == p2.read_bytes()
