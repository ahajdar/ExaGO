#!/usr/bin/env python3
"""Network-neutral rendering of worked examples (corpus v2).

Worked examples teach the *structure* of a correct response. Concrete bus
numbers tie an example to the network it was produced on, and a model grounded
on it tends to copy them into whatever network it is working on. Observed on
2026-10-07: in a corrective-RAG run on case118, 63 of 68 skipped commands
targeted ACTIVSg500 buses copied from retrieved examples.

This module renders examples with role placeholders instead, e.g.
``{"action": "set_gen_status", "bus": <offline generator bus>, "status": 1}``.
Certification is unchanged: schema exemplars are still parsed and applied with
their concrete case118 numbers, run exemplars were certified by the solver on
their own network; only the rendering is neutral.

Used by rag_schema_exemplars.py and rag_scrape_journal.py. Run directly to
convert an existing v1 run-exemplar file (deterministic, idempotent):

    python rag/tools/exemplar_neutral.py rag/corpus/exago_exemplars_from_runs.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PLACEHOLDER_NOTE = (
    "Placeholders in angle brackets (e.g. <bus>, <offline generator bus 1>) stand for numbers "
    "from the network you are working on: take them from its network summary and network facts "
    "(e.g. the 'Offline generators' line). Never copy bus numbers from an example."
)
NEUTRAL_LABEL = "bus numbers: placeholders"

BUS_KEYS = ("bus", "fbus", "tbus")
_GEN_ACTIONS = ("set_gen_dispatch", "set_gen_voltage", "set_cost_coeffs")
# Values that only make sense for one specific generator of the source network.
_GEN_VALUE_PLACEHOLDERS = {"Pg": "<MW between that generator's Pmin and Pmax>"}
_COLLAPSE_OVER = 3  # more identical-action commands than this are shown as 2 + "..."


def _role(cmd: dict, key: str) -> str:
    action = str(cmd.get("action", ""))
    if key == "fbus":
        return "from bus"
    if key == "tbus":
        return "to bus"
    if action == "set_gen_status":
        return "offline generator bus" if cmd.get("status") == 1 else "online generator bus"
    if action in _GEN_ACTIONS:
        return "generator bus"
    return "bus"


def placeholder_map(cmds: list[dict]) -> dict[tuple[str, int], str]:
    """Map (role, original id) -> placeholder text, numbering roles used more than once."""
    seen: dict[str, list[int]] = {}
    for c in cmds:
        for key in BUS_KEYS:
            v = c.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                ids = seen.setdefault(_role(c, key), [])
                if v not in ids:
                    ids.append(v)
    out: dict[tuple[str, int], str] = {}
    for role, ids in seen.items():
        for i, v in enumerate(ids, 1):
            out[(role, v)] = f"<{role}>" if len(ids) == 1 else f"<{role} {i}>"
    return out


def neutral_cmd(cmd: dict, pmap: dict, neutral_values: bool) -> dict:
    out = {}
    for k, v in cmd.items():
        if k in BUS_KEYS and isinstance(v, int) and not isinstance(v, bool):
            out[k] = ("__PH__", pmap[(_role(cmd, k), v)])
        elif neutral_values and k in _GEN_VALUE_PLACEHOLDERS:
            out[k] = ("__PH__", _GEN_VALUE_PLACEHOLDERS[k])
        else:
            out[k] = v
    return out


def dump_with_placeholders(obj) -> str:
    """json.dumps that writes placeholder markers as bare <...> tokens."""
    marks: list[str] = []

    def conv(o):
        if isinstance(o, tuple) and len(o) == 2 and o[0] == "__PH__":
            marks.append(o[1])
            return f"@@PH{len(marks) - 1}@@"
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        return o

    text = json.dumps(conv(obj), separators=(", ", ": "))
    for i, m in enumerate(marks):
        text = text.replace(f'"@@PH{i}@@"', m)
    return text


def neutral_commands_json(cmds: list[dict], neutral_values: bool = False) -> str:
    """JSON-like list of commands with placeholders; long uniform lists are collapsed."""
    pmap = placeholder_map(cmds)
    ncmds = [neutral_cmd(c, pmap, neutral_values) for c in cmds]
    actions = {c.get("action") for c in cmds}
    if len(cmds) > _COLLAPSE_OVER and len(actions) == 1:
        head = dump_with_placeholders(ncmds[:2])[:-1]
        what = _role(cmds[0], "bus")
        return f"{head}, ... one command per {what} ({len(cmds)} in this example)]"
    return dump_with_placeholders(ncmds)


def neutral_summary(cmds: list[dict], neutral_values: bool = False) -> str:
    """Deterministic one-line description of the commands, with placeholders."""
    pmap = placeholder_map(cmds)
    groups: dict[str, list[dict]] = {}
    for c in cmds:
        groups.setdefault(str(c.get("action", "?")), []).append(c)
    parts = []
    for action, cs in groups.items():
        if len(cs) == 1:
            nc = neutral_cmd(cs[0], pmap, neutral_values)
            kv = ", ".join(f"{k}={v[1] if isinstance(v, tuple) else _fmt(v)}"
                           for k, v in nc.items() if k != "action")
            parts.append(f"{action}({kv})" if kv else action)
        else:
            others = {k for c in cs for k in c if k not in BUS_KEYS and k != "action"}
            shared = {k: cs[0][k] for k in others if all(c.get(k) == cs[0].get(k) for c in cs)}
            desc = f"{action} on {len(cs)} {_role(cs[0], 'bus')}es"
            if shared:
                desc += " (" + ", ".join(f"{k}={_fmt(v)}" for k, v in shared.items()) + ")"
            parts.append(desc)
    return "; ".join(parts)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def neutralize_text(text: str, cmds: list[dict]) -> str:
    """Replace the commands' bus ids in free text ('bus 12', 'buses 23 and 24', '8-5')."""
    pmap = placeholder_map(cmds)
    by_id: dict[int, str] = {}
    for (role, v), ph in pmap.items():
        by_id.setdefault(v, ph)
    for v, ph in sorted(by_id.items(), key=lambda kv: -len(str(kv[0]))):
        # a standalone integer: not part of a longer number or a decimal ("1.04", "44.5")
        text = re.sub(rf"(?<!\d)(?<!\d\.){v}(?!\d)(?!\.\d)", ph, text)
    # "bus <bus>" / "Bus <generator bus>" read better without the doubled noun
    return re.sub(r"\b[Bb]us (<[^>]*bus[^>]*>)", r"\1", text)


# --- run exemplars ---------------------------------------------------------------

def render_run_exemplar(tag: str, goal: str | None, mode: str, cmds: list[dict], app: str,
                        delta_pct: float | None) -> str:
    """One run exemplar in the v2 (network-neutral) format."""
    if NEUTRAL_LABEL not in tag:
        tag = tag[:-1] + f" | {NEUTRAL_LABEL}]" if tag.endswith("]") else tag
    step = neutral_summary(cmds, neutral_values=True)
    resp = ('{"action": "modify", "mode": ' + json.dumps(mode) + ', "description": '
            + json.dumps(step) + ', "commands": ' + neutral_commands_json(cmds, neutral_values=True) + "}")
    lines = [tag]
    if goal:
        lines.append(f"Goal: {goal}")
    lines += [f"Step applied: {step}", f"Correct response (JSON): {resp}"]
    result = f"Result: {app.upper()} converged, feasible, 0 violations"
    if delta_pct is not None:
        result += f"; objective {delta_pct:+.1f}% vs. the base case of the source network"
    lines.append(result + ".")
    if placeholder_map(cmds) or any(k in c for c in cmds for k in _GEN_VALUE_PLACEHOLDERS):
        lines.append(f"Note: {PLACEHOLDER_NOTE}")
    return "\n".join(lines)


_JSON_LINE = "Correct response (JSON): "
_DELTA = re.compile(r"\(([+-]\d+(?:\.\d+)?)% vs\. base")
_APP = re.compile(r"\| app: ([a-z]+) \|")


def convert_v1_block(block: str) -> str:
    """Convert one v1 run-exemplar block (concrete bus numbers) to v2. v2 blocks pass through."""
    lines = block.strip("\n").split("\n")
    if lines and NEUTRAL_LABEL in lines[0]:
        return "\n".join(lines)
    tag = lines[0]
    goal = next((l[len("Goal: "):] for l in lines if l.startswith("Goal: ")), None)
    resp_line = next(l for l in lines if l.startswith(_JSON_LINE))
    resp = json.loads(resp_line[len(_JSON_LINE):])
    result = next((l for l in lines if l.startswith("Result: ")), "")
    m = _DELTA.search(result)
    app = (_APP.search(tag) or re.search(r"Result: ([A-Z]+) ", result))
    app_name = app.group(1).lower() if app else "opflow"
    return render_run_exemplar(tag, goal, resp.get("mode", "accumulative"), resp["commands"], app_name,
                               float(m.group(1)) if m else None)


def convert_file(path: Path) -> int:
    blocks = [b for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    out = [convert_v1_block(b) for b in blocks]
    path.write_text("\n\n".join(out) + "\n", encoding="utf-8")
    return len(out)


def main(argv: list[str] | None = None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        n = convert_file(p)
        print(f"{p}: {n} exemplar(s) rendered network-neutral")
    print("Then: python rag/tools/corpus_guard.py rag/corpus --freeze  and re-ingest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
