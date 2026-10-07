#!/usr/bin/env python3
"""Network-neutral rendering of worked examples (corpus v2).

Worked examples teach the *structure* of a correct response. Concrete bus
numbers tie an example to the network it was produced on, and a model grounded
on it tends to copy them into whatever network it is working on. Observed on
2026-10-07: in a corrective-RAG run on case118, 63 of 68 skipped commands
targeted ACTIVSg500 buses copied from retrieved examples.

Corpus v2 rendered such examples with role placeholders inside the JSON
(``"bus": <offline generator bus 1>``). The corpus-v2 pilot (2026-10-07) showed
local models copying those placeholders literally (llama3: every rejected
iteration; qwen2.5: quoted placeholders and invented bus numbers). Corpus v3
therefore gives no JSON template for a command that names a bus: it says in
words which field takes which number and where in THIS network's data to find
it. Commands that name no bus (scale_all_loads, set_all_bus_vlimits, ...) keep
their JSON. Certification is unchanged: schema exemplars are still parsed and applied with
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
    "Bus numbers are described, not given: take them from THIS network's data. "
    "A response never contains angle brackets or placeholder words, only real numbers."
)
NEUTRAL_LABEL = "bus numbers: described in words"
V2_LABEL = "bus numbers: placeholders"

# Where in THIS network's data each role's number comes from (wording of the
# network facts: agentigrid/parsers/network_metadata.py).
_SOURCE = {
    "offline generator bus": "a bus number listed on the \"Offline generators (status=0)\" line of THIS network's facts",
    "online generator bus": "the bus number of an in-service generator of THIS network",
    "generator bus": "the bus number of a generator of THIS network",
    "from bus": "the from-bus number of a branch of THIS network",
    "to bus": "the to-bus number of the same branch",
    "bus": "a bus number of THIS network",
}
_NOT_APPLICABLE = ("If that line says (none), this step does not apply to THIS network.")

BUS_KEYS = ("bus", "fbus", "tbus")
_GEN_ACTIONS = ("set_gen_dispatch", "set_gen_voltage", "set_cost_coeffs")
# Values that only make sense for one specific generator of the source network.
_GEN_VALUE_PLACEHOLDERS = {"Pg": "<MW between that generator's Pmin and Pmax>"}
_GEN_VALUE_WORDS = {"Pg": "a value in MW between that generator's Pmin and Pmax"}
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


def _has_bus(cmd: dict) -> bool:
    return any(isinstance(cmd.get(k), int) and not isinstance(cmd.get(k), bool) for k in BUS_KEYS)


def _target(cmd: dict, n: int) -> str:
    if isinstance(cmd.get("fbus"), int):
        return "one branch" if n == 1 else f"{n} branches"
    role = _role(cmd, "bus")
    return f"one {role}" if n == 1 else f"{n} {role}es"


def _shared_values(cs: list[dict], neutral_values: bool) -> str:
    others = [k for k in cs[0] if k not in BUS_KEYS and k != "action"]
    shown = []
    for k in others:
        if neutral_values and k in _GEN_VALUE_PLACEHOLDERS:
            shown.append(f"{k} within the generator's limits")
        elif all(c.get(k) == cs[0].get(k) for c in cs):
            shown.append(f"{k}={_fmt(cs[0][k])}")
    return f" ({', '.join(shown)})" if shown else ""


def neutral_summary(cmds: list[dict], neutral_values: bool = False) -> str:
    """Deterministic one-line description of the commands, without bus numbers or brackets."""
    groups: dict[str, list[dict]] = {}
    for c in cmds:
        groups.setdefault(str(c.get("action", "?")), []).append(c)
    parts = []
    for action, cs in groups.items():
        if _has_bus(cs[0]):
            parts.append(f"{action} on {_target(cs[0], len(cs))}{_shared_values(cs, neutral_values)}")
        elif len(cs) == 1:
            kv = ", ".join(f"{k}={_fmt(v)}" for k, v in cs[0].items() if k != "action")
            parts.append(f"{action}({kv})" if kv else action)
        else:
            parts.append(f"{action} x{len(cs)}{_shared_values(cs, neutral_values)}")
    return "; ".join(parts)


# --- v3: commands that name a bus are described in words -------------------------

def _field_words(cmd: dict, k, v, neutral_values: bool, several: bool) -> str:
    if k in BUS_KEYS and isinstance(v, int) and not isinstance(v, bool):
        src = _SOURCE[_role(cmd, k)]
        if several and k != "tbus":
            src += ", a different one for each command"
        return f'"{k}" set to {src} (a plain integer, not in quotes)'
    if neutral_values and k in _GEN_VALUE_WORDS:
        return f'"{k}" set to {_GEN_VALUE_WORDS[k]}'
    return f'"{k}": {json.dumps(v)}'


def _a(word: str) -> str:
    return ("an " if word.lstrip('"')[:1].lower() in "aeiou" else "a ") + word


def command_in_words(cmd: dict, neutral_values: bool = False, several: bool = False) -> str:
    fields = [_field_words(cmd, k, v, neutral_values, several) for k, v in cmd.items() if k != "action"]
    return f'a command object with "action": {json.dumps(cmd.get("action"))}, ' + ", ".join(fields)


def commands_in_words(cmds: list[dict], neutral_values: bool = False) -> str:
    groups: dict[str, list[dict]] = {}
    for c in cmds:
        groups.setdefault(str(c.get("action", "?")), []).append(c)
    parts = []
    for action, cs in groups.items():
        if not _has_bus(cs[0]):
            parts += [json.dumps(c, separators=(", ", ": ")) for c in cs]
            continue
        others = [k for k in cs[0] if k not in BUS_KEYS and k != "action"]
        uniform = all(all(c.get(k) == cs[0].get(k) for k in others) for c in cs)
        if len(cs) > 1 and uniform:
            what = "branch" if isinstance(cs[0].get("fbus"), int) else _role(cs[0], "bus")
            parts.append(f"one command object per {what} ({len(cs)} in this example), each with "
                         f'"action": {json.dumps(action)}, ' + ", ".join(_field_words(cs[0], k, v, neutral_values, True)
                                     for k, v in cs[0].items() if k != "action"))
        else:
            parts += [command_in_words(c, neutral_values, several=len(cs) > 1) for c in cs]
    return "; ".join(parts)


def response_in_words(resp: dict, cmds: list[dict], neutral_values: bool = False) -> str:
    """The response, described in words (no JSON template to copy)."""
    action = resp.get("action")
    if action == "modify":
        text = (f'a "modify" action with "mode": {json.dumps(resp.get("mode", "accumulative"))}, '
                f'a short "description" and a "commands" list holding, in order: '
                f"{commands_in_words(cmds, neutral_values)}.")
    else:
        fields = []
        for k, v in resp.items():
            if k == "action":
                continue
            if k in ("reasoning", "description"):
                fields.append(f'a short "{k}"')
            else:
                fields.append(_field_words(resp, k, v, neutral_values, False))
        text = f'{_a(json.dumps(action))} action with ' + ", ".join(fields) + "."
    if any(_role(c, "bus") == "offline generator bus" for c in cmds if _has_bus(c)):
        text += " " + _NOT_APPLICABLE
    return text


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


_WORDS = [
    (re.compile(r"\bbuses <from bus> and <to bus>"), "two buses"),
    (re.compile(r"<from bus>-<to bus>"), "of a branch"),
    (re.compile(r"<online generator bus(?: \d+)?>"), "an in-service generator bus"),
    (re.compile(r"<offline generator bus(?: \d+)?>"), "an offline generator bus"),
    (re.compile(r"<generator bus(?: \d+)?>"), "a generator bus"),
    (re.compile(r"<(?:from |to )?bus(?: \d+)?>"), "a bus"),
]


def placeholders_to_words(text: str) -> str:
    """'<bus>' -> 'a bus' etc., so no reference text contains an angle-bracket token (v3)."""
    for pat, words in _WORDS:
        text = pat.sub(words, text)
    return text


# --- run exemplars ---------------------------------------------------------------

def render_run_exemplar(tag: str, goal: str | None, mode: str, cmds: list[dict], app: str,
                        delta_pct: float | None) -> str:
    """One run exemplar in the v3 format (bus numbers described in words)."""
    if V2_LABEL in tag:
        tag = tag.replace(V2_LABEL, NEUTRAL_LABEL)
    if NEUTRAL_LABEL not in tag:
        tag = tag[:-1] + f" | {NEUTRAL_LABEL}]" if tag.endswith("]") else tag
    step = neutral_summary(cmds, neutral_values=True)
    lines = [tag]
    if goal:
        lines.append(f"Goal: {goal}")
    lines.append(f"Step applied: {step}")
    if any(_has_bus(c) for c in cmds):
        resp = {"action": "modify", "mode": mode}
        lines.append(f"How to write the response: {response_in_words(resp, cmds, neutral_values=True)}")
    else:
        lines.append('Correct response (JSON): {"action": "modify", "mode": ' + json.dumps(mode)
                     + ', "description": ' + json.dumps(step) + ', "commands": '
                     + json.dumps(cmds, separators=(", ", ": ")) + "}")
    result = f"Result: {app.upper()} converged, feasible, 0 violations"
    if delta_pct is not None:
        result += f"; objective {delta_pct:+.1f}% vs. the base case of the source network"
    lines.append(result + ".")
    if any(_has_bus(c) for c in cmds):
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
    if lines and V2_LABEL in lines[0]:
        raise ValueError("corpus-v2 block (placeholders in JSON): convert the v1 file or re-scrape the journals")
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
