"""Per-group EFFICIENCY diagnostics: what a group's program issued, counted per execution, against its
derived roofline -- numbers that say WHAT is inefficient and never HOW to fix it.

    census = dynamic_census(elf, target=t, compiler=cc, group_objects={g: obj}, functional_model=spec, out=d)
    report = efficiency(census[g], roofline=bound, cycles=measured)

WHY DYNAMIC. A static census counts instruction WORDS in the code. A kernel that loops issues a word
once per trip, so the static count says nothing about how often anything ran: measured on one 1x1
group, 784 weight-load words in the code issued 3,136 times per invocation. Per-output-tile counts
from the static census would be fiction. So the census is static in what it DECODES -- each word's
role comes from the target's own ISA facts, by address, from the program's own disassembly -- and
dynamic in what it COUNTS: every address is weighted by how often the functional model executed it
(its PC histogram), divided by the invocations the program itself reports running.

WHAT IS COUNTED. Each accelerator instruction by its DERIVED ROLE (configuration, operand load, weight
load, accumulate, readout, sync), never by a name or an encoding, plus the host's memory-ordering
instruction, which on a co-processor stalls the host until the accelerator drains. That one is the
host ISA's, named by the disassembler that ships with the compiler (:data:`HOST_ORDERING_MNEMONIC`);
it is not a fact about any accelerator. Only code in the group's own kernel object is counted.

WHAT IS REPORTED (:func:`efficiency`): computes issued against the roofline's minimum, cycles per
compute, cycles against the roofline, and each role per OUTPUT TILE (the roofline's tile count). The
cycles are the measurement's; the census is the functional model's, which runs the same program bytes.
Nothing here names a target.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "HOST_ORDERING_MNEMONIC",
    "REPORTED_ROLES",
    "SCHEMA",
    "dynamic_census",
    "efficiency",
    "invocations",
    "pc_histogram",
]

SCHEMA = "merlin_group_efficiency_v1"
#: The host ISA's memory-ordering instruction as the toolchain's own disassembler spells it.
HOST_ORDERING_MNEMONIC = "fence"
#: The derived instruction roles reported per output tile, in reading order.
REPORTED_ROLES = ("config", "operand_load", "weight_load", "accumulate", "readout", "sync")
#: The role whose count is "computes issued".
COMPUTE_ROLE = "accumulate"
#: The functional model's own banner for the PC histogram it writes at exit (``-g``).
_HISTOGRAM_BANNER = "PC Histogram size:"
#: The program's own statement of how many times it ran the group.
_INVOCATIONS_PREFIX = "MERLIN_INVOCATIONS"


def pc_histogram(text: str) -> dict[int, int]:
    """``{address: executions}`` from the functional model's histogram (the lines after its banner)."""
    found: dict[int, int] = {}
    started = False
    for line in text.splitlines():
        if not started:
            started = line.strip().startswith(_HISTOGRAM_BANNER)
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            found[int(parts[0], 16)] = found.get(int(parts[0], 16), 0) + int(parts[1])
        except ValueError:
            continue
    return found


def invocations(console: str) -> int | None:
    """How many times the program ran the group (warm-up plus measured), from its own line."""
    for line in console.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_INVOCATIONS_PREFIX + " "):
            continue
        total = 0
        for token in stripped.split()[1:]:
            key, sep, value = token.partition("=")
            if sep and key and value.isdigit():
                total += int(value)
        return total or None
    return None


def run_functional_model(spec: Mapping[str, Any], elf: Path, out: Path, *, timeout_s: float = 900) -> dict[str, Any]:
    """Run ``elf`` on the functional model with its PC histogram on; the console and histogram on disk."""
    command = [str(c) for c in spec.get("command") or ()]
    if not command:
        raise ValueError("the functional-model spec names no command")
    out.mkdir(parents=True, exist_ok=True)
    console, histogram = out / "console.txt", out / "histogram.txt"
    with console.open("wb") as stdout, histogram.open("wb") as stderr:
        done = subprocess.run(
            [*command, "-g", str(elf)],
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            env={**os.environ, **{str(k): str(v) for k, v in (spec.get("environment") or {}).items()}},
            timeout=timeout_s,
            check=False,
        )
    return {"returncode": done.returncode, "console": console, "histogram": histogram}


def dynamic_census(
    elf: Path,
    *,
    target: str,
    compiler: str | Path,
    group_objects: Mapping[str, Path],
    functional_model: Mapping[str, Any],
    out: Path,
    timeout_s: float = 900,
) -> dict[str, Any]:
    """Per group: each derived role's executions PER INVOCATION, and the host ordering instructions.

    ``{"invocations", "per_group": {group: {"roles": {...}, "host_ordering": n, "uneven": [...]}}}``,
    or ``{"refusal": ...}`` when the functional model did not run the program to its end or the program
    did not say how often it ran the group -- never a guessed count."""
    from merlin.analysis.perf import isa_prohibition as ISA

    run = run_functional_model(functional_model, Path(elf), Path(out), timeout_s=timeout_s)
    console = Path(run["console"]).read_text(encoding="utf-8", errors="replace")
    if run["returncode"] != 0:
        return {"refusal": f"the functional model exited {run['returncode']}"}
    times = invocations(console)
    if not times:
        return {"refusal": "the program did not state how many times it ran the group"}
    histogram = pc_histogram(Path(run["histogram"]).read_text(encoding="utf-8", errors="replace"))
    if not histogram:
        return {"refusal": "the functional model wrote no PC histogram"}
    objdump = ISA.disassembler_for(compiler)
    owner = ISA._owner_of(group_objects, nm=objdump.with_name(objdump.name.replace("objdump", "nm")))
    roles_of = ISA._declared_by_selector(target)
    opcode = ISA.custom_opcode(target)
    totals: dict[str, dict[str, int]] = {}
    for row in ISA.listing(Path(elf), objdump=objdump):
        group = owner.get(str(row["function"]))
        executed = histogram.get(row["address"], 0)
        if group is None or not executed:
            continue
        bucket = totals.setdefault(group, {})
        if row["word"] & ISA.MAJOR_OPCODE_MASK == opcode:
            known = roles_of.get(row["word"] >> ISA.SELECTOR_SHIFT)
            for role in (known or {}).get("roles") or [f"{ISA.UNNAMED_SELECTOR}_{row['word'] >> ISA.SELECTOR_SHIFT}"]:
                bucket[role] = bucket.get(role, 0) + executed
        elif row["mnemonic"] == HOST_ORDERING_MNEMONIC:
            bucket[HOST_ORDERING_MNEMONIC] = bucket.get(HOST_ORDERING_MNEMONIC, 0) + executed
    per_group = {}
    for group, counts in totals.items():
        uneven = sorted(k for k, v in counts.items() if v % times)
        per_group[group] = {
            "roles": {k: v // times for k, v in counts.items() if k != HOST_ORDERING_MNEMONIC},
            "host_ordering": counts.get(HOST_ORDERING_MNEMONIC, 0) // times,
            # A count that does not divide by the invocations means the runs did not issue the same
            # stream; the per-invocation figure is then an average, and says so.
            "uneven": uneven,
        }
    return {"invocations": times, "per_group": per_group, "functional_model": dict(functional_model)}


def efficiency(census: Mapping[str, Any] | None, *, roofline: Mapping[str, Any] | None, cycles: Any) -> dict[str, Any]:
    """The diagnostic numbers for one group. Every ratio is ``None`` when an operand is missing."""
    census, roofline = census or {}, roofline or {}
    roles = dict(census.get("roles") or {})
    issued = roles.get(COMPUTE_ROLE)
    floor_computes = roofline.get("min_computes")
    bound = roofline.get("roofline_cycles") if roofline.get("status") == "derived" else None
    tiles = roofline.get("output_tiles")
    measured = cycles if isinstance(cycles, int) and not isinstance(cycles, bool) else None

    def ratio(a: Any, b: Any, places: int = 3) -> float | None:
        return round(a / b, places) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None

    per_tile = {role: ratio(roles.get(role, 0), tiles, 2) for role in REPORTED_ROLES}
    per_tile[HOST_ORDERING_MNEMONIC] = ratio(census.get("host_ordering", 0), tiles, 2) if census else None
    return {
        "schema": SCHEMA,
        "cycles": measured,
        "roofline_cycles": bound,
        "cycles_over_roofline": ratio(measured, bound),
        "limiter": roofline.get("limiter"),
        "computes_issued": issued,
        "computes_min": floor_computes,
        "computes_over_min": ratio(issued, floor_computes),
        "cycles_per_compute": ratio(measured, issued, 2),
        "cycles_per_compute_floor": roofline.get("cycles_per_compute_floor"),
        "output_tiles": tiles,
        "per_output_tile": per_tile if tiles and census else None,
        "uneven": list(census.get("uneven") or ()),
    }


def describe(report: Mapping[str, Any]) -> str:
    """One line of numbers for a prompt: what ran and how much, against the floor -- no advice."""
    parts = []
    if report.get("roofline_cycles"):
        parts.append(
            f"{report['cycles']:,} cyc vs roofline {report['roofline_cycles']:,} "
            f"({report.get('cycles_over_roofline')}x, {report.get('limiter')}-bound)"
            if isinstance(report.get("cycles"), int)
            else f"roofline {report['roofline_cycles']:,} ({report.get('limiter')}-bound)"
        )
    if report.get("computes_issued") is not None:
        parts.append(
            f"computes {report['computes_issued']:,} vs min {report.get('computes_min') or 0:,} "
            f"({report.get('computes_over_min')}x), {report.get('cycles_per_compute')} cyc/compute "
            f"(floor {report.get('cycles_per_compute_floor')})"
        )
    per_tile = report.get("per_output_tile")
    if per_tile:
        parts.append(
            f"per output tile ({report.get('output_tiles')}): "
            + " ".join(f"{role}={value}" for role, value in per_tile.items() if value)
        )
    return "; ".join(parts) or "no diagnostic derivable"
