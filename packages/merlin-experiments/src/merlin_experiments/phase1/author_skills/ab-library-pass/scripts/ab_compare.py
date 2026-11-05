"""Compare two self-check responses case by case: cycles before/after, delta and numerics.

    python ab_compare.py AFTER [BEFORE] [--channel .qa_channel] [--cases A,B] [--json]

AFTER and BEFORE are each a response JSON file (``--out`` of ``agent_selfcheck.py``), a request id
(read as ``<channel>/resp_<id>.json``), or a tree -- ``submission`` or ``variants/NAME`` -- meaning
the newest completed response of that tree in the channel. Without BEFORE, each case of AFTER is
compared with the newest ``submission/`` response row that has comparable cycles for it.

Cycles are compared only within one kind and identity: measured simulator cycles of the certifying
tier (``barrier_cycles`` at L3, same simulator) first, else the timing model estimate
(``performance.estimated_whole_soc_cycles``, complete provisional/qualified accounting with matching
selection/model/facts/domain/scope/case pins, timing mode and epoch). Numerics are each
row's own numerical check. Exit status: 0 keep (every case comparable, total cycles fall, no case slower, numerics
unchanged), 1 do not keep or mixed, 2 nothing comparable or unreadable input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_MODEL_PINS = ("selection_sha256", "model_sha256", "facts_sha256", "comparison_domain_sha256",
               "scope_sha256", "capsule_sha256")
_MODEL_IDENTITY = (*_MODEL_PINS, "timing_mode", "timing_epoch")


def _load(path: Path):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("result_index"):
            for parent in Path(path).resolve().parents:
                helper = parent / "qa_result_details.py"
                if helper.is_file():
                    namespace = {"__name__": "_public_result_details"}
                    exec(compile(helper.read_bytes(), str(helper), "exec"), namespace)
                    return namespace["hydrate"](document, workspace=parent)
        return document
    except (OSError, ValueError):
        return None


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def tree(document: dict) -> str:
    """Which tree a response graded: ``submission``, ``variants/NAME`` or a knob-override copy."""
    variant = document.get("variant") if isinstance(document.get("variant"), dict) else None
    if variant and variant.get("kind") == "knob_overrides":
        return "knobs " + json.dumps(variant.get("tuning_overrides"), sort_keys=True)
    label = document.get("selfcheck_submission")
    if isinstance(label, str) and label:
        return label
    return variant.get("path") if variant and variant.get("path") else "submission"


def passed(row: dict) -> bool:
    """A row's own numerical check: its L2 tier when present, else its pass flags."""
    tiers = row.get("tiers")
    if isinstance(tiers, dict) and "L2" in tiers:
        tier = tiers["L2"]
        return (tier.get("status") if isinstance(tier, dict) else tier) == "pass"
    if type(row.get("numerical_execution_passed")) is bool:
        return row["numerical_execution_passed"]
    return row.get("pass") is True or row.get("status") == "pass"


def evidence(row: dict) -> dict:
    """``{kind: (cycles, identity, label)}`` for the cycle evidence one response row carries."""
    found = {}
    if row.get("barrier_tier") == "L3" and _positive(row.get("barrier_cycles")) and row.get("pass") is True:
        timing = row.get("barrier_timing_identity") if isinstance(row.get("barrier_timing_identity"), dict) else {}
        engine = timing.get("engine")
        if isinstance(engine, str) and engine and _digest(timing.get("simulator_sha256")):
            found["measured"] = (row["barrier_cycles"], (engine, timing["simulator_sha256"]),
                                 f"measured {engine} cycles")
    performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
    status = str(performance.get("status") or "unknown").lower()
    if (status in ("provisional", "qualified")
            and performance.get("complete_event_accounting") is True
            and all(_digest(performance.get(key)) for key in _MODEL_PINS)
            and _positive(performance.get("estimated_whole_soc_cycles"))):
        identity = tuple(performance.get(key) for key in _MODEL_IDENTITY)
        found["modeled"] = (performance["estimated_whole_soc_cycles"], identity, f"timing model ({status})")
    return found


def rows(document: dict) -> dict:
    found = {}
    for row in document.get("per_capsule") or document.get("cases") or ():
        if isinstance(row, dict) and (row.get("capsule") or row.get("name")):
            found[str(row.get("capsule") or row.get("name"))] = row
    return found


def _responses(channel: Path):
    for path in channel.glob("resp_*.json"):
        document = _load(path)
        if isinstance(document, dict) and rows(document):
            yield document.get("graded_at_unix_ns") or 0, path.name, document


def resolve(text: str, channel: Path) -> dict:
    """A response document from a file, a request id or a tree name (see the module docstring)."""
    path = Path(text)
    if path.is_file():
        document = _load(path)
    elif (channel / f"resp_{text}.json").is_file():
        document = _load(channel / f"resp_{text}.json")
    elif text == "submission" or text.startswith("variants/"):
        newest = sorted((r for r in _responses(channel) if tree(r[2]) == text), key=lambda r: (r[0], r[1]))
        if not newest:
            raise SystemExit(f"no completed response of {text} with result rows in {channel}")
        document = newest[-1][2]
    else:
        raise SystemExit(f"{text}: not a file, a request id in {channel}, 'submission' or 'variants/NAME'")
    if not isinstance(document, dict):
        raise SystemExit(f"{text}: not a JSON response object")
    if document.get("error") and not rows(document):
        raise SystemExit(f"{text}: the check returned an error, not results: {str(document['error'])[:300]}")
    return document


def _pair(before: dict, after: dict, kinds=("measured", "modeled")):
    """``(kind, before_cycles, after_cycles, label)`` of the first comparable kind, else a reason."""
    for kind in kinds:
        if kind in before and kind in after:
            if before[kind][1] == after[kind][1]:
                return kind, before[kind][0], after[kind][0], after[kind][2]
    kinds = sorted(set(before) & set(after))
    if kinds:
        return f"{kinds[0]} cycles come from different simulators or timing models"
    return "no common cycle evidence (run both on the same cases, suite and sim)"


def _baseline_rows(channel: Path, names) -> dict:
    """Per case, the newest submission/ row of each evidence kind (for a missing BEFORE)."""
    found: dict[str, list] = {}
    for _at, _name, document in sorted(_responses(channel), key=lambda r: (r[0], r[1])):
        if tree(document) != "submission":
            continue
        for name, row in rows(document).items():
            if name in names:
                found.setdefault(name, []).append((row, document.get("submission_sha256")))
    return found


def _numerics(before_ok: bool, after_ok: bool) -> str:
    if before_ok and after_ok:
        return "unchanged (both pass)"
    if before_ok:
        return "CHANGED: candidate fails"
    return "candidate passes, baseline failed" if after_ok else "both fail"


def compare(after: dict, before: dict | None, *, channel: Path, cases=None) -> dict:
    after_rows = rows(after)
    names = [n for n in after_rows if not cases or n in cases]
    if cases:
        names.extend(sorted(set(cases) - set(after_rows)))
    history = _baseline_rows(channel, set(names)) if before is None else {}
    before_rows = rows(before) if before is not None else {}
    report = []
    for name in names:
        if name not in after_rows:
            report.append({"test_case": name, "after_passed": None, "status": "no candidate row for requested case"})
            continue
        row = after_rows[name]
        candidates = ([(before_rows[name], before.get("submission_sha256"))] if name in before_rows
                      else list(reversed(history.get(name, ()))))
        entry = {"test_case": name, "after_passed": passed(row)}
        if not candidates:
            entry.update(status="no baseline row for this case")
            report.append(entry)
            continue
        chosen = None  # measured cycles first, from the newest row that has them; then the model
        for kind in ("measured", "modeled"):
            for base_row, digest in candidates:
                paired = _pair(evidence(base_row), evidence(row), (kind,))
                if chosen is None and not isinstance(paired, str):
                    chosen = (base_row, digest, paired)
        if chosen is None:
            base_row, digest = candidates[0]
            entry.update(before_passed=passed(base_row), status=_pair(evidence(base_row), evidence(row)),
                         numerics=_numerics(passed(base_row), passed(row)))
            report.append(entry)
            continue
        base_row, digest, (kind, cycles_before, cycles_after, label) = chosen
        entry.update(before_passed=passed(base_row), kind=kind, evidence=label, cycles_before=cycles_before,
                     cycles_after=cycles_after, delta=cycles_after - cycles_before,
                     delta_pct=round(100.0 * (cycles_after - cycles_before) / cycles_before, 2),
                     numerics=_numerics(passed(base_row), passed(row)), baseline_sha256=digest,
                     status="compared")
        report.append(entry)
    return {"after": tree(after), "after_sha256": (after.get("variant") or {}).get("sha256")
            or after.get("submission_sha256"),
            "before": tree(before) if before is not None else "newest submission/ rows",
            "cases": report, "verdict": verdict(report)}


def verdict(report: list[dict]) -> dict:
    compared = [e for e in report if e.get("status") == "compared"]
    broken = [e["test_case"] for e in report if e.get("after_passed") is False]
    if broken:
        return {"decision": "do not keep", "exit": 1, "why": f"candidate numerical check did not pass: {broken}"}
    if any(e.get("after_passed") is None for e in report):
        return {"decision": "inconclusive", "exit": 2, "why": "some requested cases have no candidate result"}
    if any(e.get("before_passed") is not True for e in report):
        return {"decision": "inconclusive", "exit": 2, "why": "every baseline numerical check must pass"}
    if not compared:
        return {"decision": "inconclusive", "exit": 2, "why": "no case has comparable cycles on both sides"}
    if len(compared) != len(report):
        return {"decision": "inconclusive", "exit": 2,
                "why": "some cases lack complete, identity-matched cycle evidence on both sides"}
    total_before = sum(e["cycles_before"] for e in compared)
    total_after = sum(e["cycles_after"] for e in compared)
    slower = [e["test_case"] for e in compared if e["delta"] > 0]
    change = round(100.0 * (total_after - total_before) / total_before, 2)
    if total_after >= total_before:
        return {"decision": "do not keep", "exit": 1, "why": f"total cycles {change:+}% over {len(compared)} case(s)"}
    if slower:
        return {"decision": "mixed", "exit": 1, "why": f"total {change:+}% but slower on {slower}: restrict where "
                "the change applies (a predicate on the facts that differ) and re-check"}
    return {"decision": "keep", "exit": 0, "why": f"total cycles {change:+}% over {len(compared)} case(s), "
            "no case slower, numerics unchanged"}


def _table(result: dict) -> str:
    lines = [f"after: {result['after']} ({str(result['after_sha256'])[:12]})   before: {result['before']}",
             f"{'test case':44} {'before':>12} {'after':>12} {'delta':>9}  numerics / status"]
    for e in result["cases"]:
        if e.get("status") == "compared":
            lines.append(f"{e['test_case'][:44]:44} {e['cycles_before']:>12} {e['cycles_after']:>12} "
                         f"{e['delta_pct']:>+8}%  {e['numerics']} [{e['evidence']}; baseline "
                         f"{str(e.get('baseline_sha256'))[:12]}]")
        else:
            lines.append(f"{e['test_case'][:44]:44} {'-':>12} {'-':>12} {'-':>9}  {e['status']}"
                         + (f"; {e['numerics']}" if e.get("numerics") else ""))
    v = result["verdict"]
    lines.append(f"verdict: {v['decision'].upper()} -- {v['why']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("after")
    parser.add_argument("before", nargs="?")
    parser.add_argument("--channel", default=".qa_channel", help="the self-check channel directory")
    parser.add_argument("--cases", default="", help="comma-separated test cases to compare (default: all)")
    parser.add_argument("--json", action="store_true", help="print the comparison as JSON")
    args = parser.parse_args(argv)
    channel = Path(args.channel)
    after = resolve(args.after, channel)
    before = resolve(args.before, channel) if args.before else None
    cases = {c.strip() for c in args.cases.split(",") if c.strip()} or None
    result = compare(after, before, channel=channel, cases=cases)
    print(json.dumps(result, indent=2) if args.json else _table(result))
    return result["verdict"]["exit"]


if __name__ == "__main__":
    sys.exit(main())
