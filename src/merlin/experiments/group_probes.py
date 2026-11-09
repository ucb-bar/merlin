"""Single-group probes for ``merlin experiment inspect <candidate> --group gN --time --profile``.

:mod:`.group_inspect` rebuilds one group of a measured candidate as its own one-step program. These two
probes say what that program DOES, without a whole-model run:

* **timing** (:func:`time_group`): the program on the elaborated-RTL emulator, on the model's own inputs
  to the group, graded exactly against the reference recomputed from those inputs -- the same run the
  phase-2 fast tier makes (:func:`merlin.perf.whole_model_group_timing.time_group_programs`), for one
  named group. The cycles are the EMULATOR's own device and a RANKING signal: the run's own measurement
  ladder adjudication rides with them, and the board alone adjudicates a cycle count.
* **profile** (:func:`profile_group`): the group's hardware-counter facts and its instruction census
  by role. Counter NAMES come from the target's own counter header
  (:func:`merlin.perf.hw_counters.counters_for_target`): the per-engine busy/overlap block factored out of
  the header, and every other cycle-unit counter it declares under its own name. Counter VALUES are read
  from a console the program printed and are admitted only through
  :func:`merlin.perf.counter_trust.values_or_refusal` for the engine that ran it, and only when the
  console's counter-schema digest (if it states one) is the header the names came from. The census is
  the whole one-group ELF's custom instructions by derived role, split into the group's kernel and the
  program's own code (:func:`.phase2.whole_model_measured.group_capsules.isa_scan`).

Nothing here names a target, an engine, an opcode or a counter: each comes from the target's facts, its
header, the run's own record, or the reviewed counter-trust table. What cannot be read is ``UNKNOWN``
with the reason, never a zero.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "UNKNOWN",
    "counter_facts",
    "counter_values",
    "engine_of",
    "instruction_census",
    "profile_group",
    "time_group",
]

PROFILE_SCHEMA = "merlin_group_profile_v1"
TIMING_SCHEMA = "merlin_group_timing_probe_v1"
UNKNOWN = "UNKNOWN"
#: The unit token a counter header declares in a duration counter's name (the unit query
#: :func:`merlin.perf.hw_counters.counters_with_unit` answers); a busy figure is a duration.
_DURATION_UNIT = "CYCLES"
_NOT_A_CLAIM = (
    "a ranking and structure signal from the elaborated-RTL emulator; the cycle figure is that device's "
    "own and is never quoted as the board's cycle count"
)


def _unknown(why: str, **known: Any) -> dict[str, Any]:
    return {"status": UNKNOWN, "why": why, **known}


# ----------------------------------------------------------------------------------------- timing


def time_group(
    record: Mapping[str, Any],
    *,
    target: str,
    model_capsule: str | Path,
    out: str | Path,
    max_cycles: int = 60_000_000,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Run the rebuilt group's program once on the elaborated-RTL emulator and grade it exactly.

    Never served from the timing cache: a probe asked for is a run made, so its console exists for the
    counter profile. Returns the timing row (``status``, ``cycles``, ``correct``, ``refusal`` ...) with
    the device label, the run's own cycle adjudication and where its console is.
    """
    from merlin.perf import whole_model_group_timing as GT

    group = int(record["group"])
    timed = GT.time_group_programs(
        {group: record},
        target=target,
        model_capsule=model_capsule,
        out=out,
        max_parallel=1,
        max_cycles=max_cycles,
        timeout_s=timeout_s,
        cache=None,
    )
    row = dict(timed[group]) if group in timed else {"group": group, "status": "refused", "refusal": "no result"}
    run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
    verdict_path = run_dir / "verdict.json" if run_dir is not None else None
    verdict = json.loads(verdict_path.read_text(encoding="utf-8")) if verdict_path and verdict_path.is_file() else {}
    adjudication = verdict.get("cycles_adjudication")
    console = run_dir / "console.txt" if run_dir is not None else None
    return {
        "schema": TIMING_SCHEMA,
        **row,
        "signal": "ranking",
        "device": GT.DEVICE_LABEL,
        "not_a_claim": _NOT_A_CLAIM,
        "cycles_adjudication": adjudication
        if isinstance(adjudication, Mapping)
        else _unknown("the run recorded no measurement-ladder adjudication"),
        "console": str(console) if console is not None and console.is_file() else None,
    }


def engine_of(timing: Mapping[str, Any]) -> str | None:
    """The engine a timing run names in its own adjudication (the ladder instrument it ran on), or None."""
    adjudication = timing.get("cycles_adjudication")
    instrument = adjudication.get("instrument") if isinstance(adjudication, Mapping) else None
    return str(instrument) if instrument else None


# ---------------------------------------------------------------------------------------- counters


def counter_facts(target: str) -> dict[str, Any]:
    """The counter names the target's own header declares: the busy/overlap block and other durations."""
    from merlin.perf import hw_counters as HWC

    discovery = HWC.counters_for_target(target)
    if discovery.get("status") != "derived":
        return {"status": discovery.get("status") or UNKNOWN, "why": discovery.get("why")}
    header = Path(str(discovery["header"]))
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _unknown(f"the derived counter header {header} could not be read again: {exc}")
    occupancy = HWC.derive_occupancy_counters(text)
    busy = dict(sorted(("+".join(sorted(combo)), name) for combo, name in occupancy.by_combination.items()))
    durations = HWC.counters_with_unit(text, _DURATION_UNIT)
    return {
        "status": "derived",
        "header": str(header),
        "header_sha256": discovery.get("header_sha256"),
        "engines": list(occupancy.engines),
        "complete": occupancy.complete(),
        "busy_counters": busy,
        "other_duration_counters": sorted(name for name in durations if name not in set(busy.values())),
    }


def counter_values(facts: Mapping[str, Any], console: str | None, *, engine: str | None) -> dict[str, Any]:
    """The derived counters' values in ``console``, admitted only for a trusted engine and the same header."""
    from merlin.perf import counter_trust as CT
    from merlin.perf import hw_counters as HWC

    if facts.get("status") != "derived":
        return _unknown("no counter names were derived for this target, so nothing can be read")
    if not console:
        return _unknown("no console was supplied (run with --time, or name one with --counter-console)")
    busy, others = dict(facts["busy_counters"]), list(facts["other_duration_counters"])
    readings = HWC.parse_counter_output(console)
    read = {name: readings[name] for name in [*busy.values(), *others] if name in readings}
    if not read:
        return _unknown("the console carries no reading of any derived counter; the program did not print them")
    schema = HWC.parse_counter_schema(console)
    if schema is not None and schema != facts.get("header_sha256"):
        return {
            "status": "refused",
            "why": f"the program printed counter schema {schema[:12]}, but the names were derived from header "
            f"{str(facts.get('header_sha256'))[:12]}; the same name may be a different event",
        }
    admitted, refusal = CT.values_or_refusal(engine, read)
    if admitted is None:
        return {"status": "refused", "engine": engine, "why": refusal}
    return {
        "status": "measured",
        "engine": engine,
        "schema": schema if schema is not None else UNKNOWN,
        "busy_cycles": {combo: admitted[name] if name in admitted else None for combo, name in busy.items()},
        "other_duration_counters": {name: admitted[name] if name in admitted else None for name in others},
    }


# ------------------------------------------------------------------------------------------ census


def instruction_census(record: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """The one-group ELF's custom instructions by derived role: the group's kernel and the program's code."""
    from merlin.perf import isa_prohibition as ISA

    from .phase2.whole_model_measured.group_capsules import GroupCapsuleError, isa_scan

    try:
        report = isa_scan(record, target=target, roles=())
    except (GroupCapsuleError, ValueError, OSError, subprocess.SubprocessError) as exc:
        return _unknown(f"the program could not be disassembled for a census: {exc}")
    census = report.get("census")
    if report.get("status") != "measured" or not isinstance(census, Mapping):
        return _unknown(str(report.get("error") or report.get("detail") or "the census did not run"))
    per_group = census["per_group"]
    empty = {"total": 0, "by_kind": {}, "by_symbol": {}}
    return {
        "status": "measured",
        "kernel": per_group.get(str(record["group"]), empty),
        "program_code": per_group.get(ISA.PROGRAM_CODE, empty),
        "scope": "every custom instruction in the linked one-group program, by derived role; static, not executed",
    }


def profile_group(
    record: Mapping[str, Any],
    *,
    target: str,
    console: str | None = None,
    engine: str | None = None,
) -> dict[str, Any]:
    """One group's counter facts, its counter values from ``console`` (run on ``engine``), and its census."""
    facts = counter_facts(target)
    return {
        "schema": PROFILE_SCHEMA,
        "target": target,
        "group": int(record["group"]),
        "not_a_claim": "measurements and derived facts only; no ranking and no verdict is rendered here",
        "counters": {"facts": facts, "values": counter_values(facts, console, engine=engine)},
        "instruction_census": instruction_census(record, target=target),
    }
