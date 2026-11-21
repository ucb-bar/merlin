"""Tier B of the phase-2 fast feedback: the groups an edit CHANGED, timed on the elaborated-RTL emulator.

    diff = changed_groups(candidate_record, baseline_record)
    timing = changed_group_timing(candidate, baseline, groups=diff["changed"], target=..., ...)

A whole-model measurement on the board costs a build, a queue slot and ~12 minutes of a shared FPGA; a
whole model on the emulator costs ~2 hours. Neither is a per-edit signal. What an edit changes is
usually a handful of groups, and a group's cycles are a property of its own program -- so each changed
group is built as a SMALL bare-metal program of its own and run on the emulator, for the candidate and
for the baseline, in parallel:

* **Which groups changed** is read off the two build records, never guessed: a group changed when who
  answered it, why, which object implements it (by content digest), or whether it gathers an operand
  differs (:func:`changed_groups`).
* **The per-group program is the whole-model program with one step.** It is rendered by the target's
  own whole-model driver from the same extraction, with the same kernel object (rebuilt from the
  package's own reply, compared by digest with the one the record names) or the same library call; the
  group's inputs are the values the reference statement computes for them (the model with no package),
  embedded as read-only data, so the group runs on the inputs the model would hand it. Each embedded
  value is laid out in the element type the program's own ABI header declares for its buffer
  (:func:`ctype_dtypes`), never in an assumed width.
* **Correctness is the group's own**: the emulator dumps the group's output at exit and
  :func:`merlin.analysis.whole_model.build.stages.grade_memory` grades it LOCALLY against the reference recomputed
  from the embedded inputs (:class:`~merlin.analysis.whole_model.build.stages.LocalReference`).
* **Cycles are the emulator's own device.** The emulator elaborates a different design from the board
  (a full-width accumulator readout), so its cycle counts are labelled as its own and are NEVER
  compared with a board number. What they are for is DIRECTION: did this edit make these groups faster
  or slower. :func:`direction_agreement` measures how often that direction matches the board's on
  pairs measured on both, per kind, and a kind whose agreement is poor is flagged.

Concurrency is capped by the host's load (the emulator is single-threaded and the host is shared): at
most ``max_parallel`` at once, and fewer while the one-minute load average is above the core count.
Nothing here names a target: the driver, the ISA and the emulator are the target's, reached by name.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ---- whole_model_group_timing ------------------------------------------------------------------


def build_reference_group_programs(
    groups: Sequence[int] | None,
    *,
    model_capsule: str | Path,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    out: str | Path,
    verify: str = "host_dump",
    harness_overrides: Sequence[str | Path] = (),
    exactness: ContractOf | None = None,
) -> dict[int, dict[str, Any]]:
    """The REFERENCE arm's one-group programs: the target's library answers the group (graded under the
    same ``exactness`` contract as the package arm, see :func:`build_group_programs`).

    :func:`merlin.analysis.whole_model.build.builder.build_reference` restricted to one step -- the same driver,
    header assertion, harness overrides and no instruction rule (the reference arm is the bar, and the
    bar is the library as the vendor ships it) -- over the same one-group model and embedded inputs
    :func:`build_group_programs` builds a package's program from, so the two arms of one group differ
    only in who answered it."""
    from merlin.analysis.whole_model.build import stages as W
    from merlin.analysis.whole_model.grading.group_programs import (
        ContractOf,
        _board_variant,
        _contract,
        _entries,
        _graded_as,
        _group_map,
        _one_group_model,
        _require_headers_read,
        ctype_dtypes,
        reference_context,
    )
    from merlin.runtime.backends import registry as backends

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    abi = W.machine_header(machine, header, header_sha256)
    capsule = W.load_model_capsule(model_capsule)
    driver = backends.whole_model_driver(target)
    (value,) = capsule.inputs.values()
    (golden,) = capsule.outputs.values()
    model = driver.program.extract(
        None,
        target,
        sources={
            "linalg": capsule.interface,
            "weights_manifest": capsule.weights_manifest,
            "weights": capsule.weights,
            "input": value,
            "golden": golden,
        },
    )
    reference = reference_context(model_capsule, target=target)
    (out / "oracle.json").write_text(json.dumps(reference["oracle"], indent=1) + "\n", encoding="utf-8")
    stated = W.state(capsule, target=target)
    full_rows = _graded_as(stated)
    entries = _entries(stated)
    dtypes = ctype_dtypes(header)
    producers = {str(s["out"]): int(s["group"]) for s in model["steps"]}
    if groups is None:
        groups = [int(s["group"]) for s in model["steps"]]
    overrides = {str(Path(o).resolve()): W._sha256(o) for o in harness_overrides}
    records: dict[int, dict[str, Any]] = {}
    for group in groups:
        here = out / f"g{int(group)}"
        record: dict[str, Any] = {"group": int(group), "on": "vendor", "arm": "reference"}
        try:
            one = _one_group_model(model, int(group), reference["values"], dtypes)
            recipe = W._with_header(
                backends.harness_build_recipe(target), Path(header), here / "harness", [Path(o) for o in overrides]
            )
            receipt = driver.program.build(one, None, None, here / "program", recipe=recipe, verify=verify)
            _require_headers_read(receipt, abi, overrides)
            layout = _group_map(
                Path(receipt["elf"]),
                one,
                full_rows[int(group)],
                producers,
                contract=_contract(exactness, group, entries),
            )
            (here / "memory_map.json").write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
            record.update(
                {
                    "elf": receipt["elf"],
                    "elf_sha256": receipt["elf_sha256"],
                    "memory_map": str(here / "memory_map.json"),
                    "oracle": str(out / "oracle.json"),
                    "linked": "vendor",
                    "call": driver.program._call(one["steps"][0]).split("(", 1)[0].strip(),
                    "abi_header": abi,
                    "harness_overrides": overrides,
                    "host_routed": receipt.get("host_routed") or [],
                    "library_paths": receipt.get("library_paths"),
                    "program_source": receipt.get("program_source"),
                    "variant": _board_variant(receipt),
                }
            )
        except (Exception, SystemExit) as error:  # noqa: BLE001 -- a group that cannot be built is a named refusal
            record["refusal"] = f"{type(error).__name__}: {str(error)[-600:]}"
        records[int(group)] = record
    (out / "group_programs.json").write_text(json.dumps(records, indent=1, default=str) + "\n", encoding="utf-8")
    return records


def _slots_free(max_parallel: int, running: int, *, min_parallel: int = 2) -> bool:
    """Room for one more emulator: under the cap, and within what the OTHER load leaves free.

    The emulator is single-threaded. The one-minute load average counts this caller's own runs too, so
    the room is the core count minus the load everyone else puts on the host; ``min_parallel`` runs are
    always allowed, so a busy host slows the tier down rather than stopping it.
    """
    if running >= max_parallel:
        return False
    if running < min_parallel:
        return True
    try:
        others = os.getloadavg()[0] - running
    except OSError:
        return True
    return running + 1 <= (os.cpu_count() or 1) - others


def time_group_programs(
    programs: Mapping[int, Mapping[str, Any]],
    *,
    target: str,
    model_capsule: str | Path,
    out: str | Path,
    max_parallel: int = 8,
    max_cycles: int = 60_000_000,
    timeout_s: float = 6 * 3600,
    emulator: str | Path | None = None,
    cache: str | Path | None = "default",
) -> dict[int, dict[str, Any]]:
    """Run each program on the dump-capable emulator; ``{group: {cycles, correct, grade, wall_s, ...}}``.

    Correctness is the group's LOCAL grade over its dumped output. A refused run (no line, no dump, a
    timeout) is reported as refused with its reason and carries no cycles.

    A GRADED run is kept in ``cache`` keyed by the program's bytes and the emulator's bytes, so the
    baseline side of the next comparison (usually the same best) is not paid for again; a carried
    row says so (``carried: true``). ``cache=None`` turns it off.
    """
    from merlin.analysis.whole_model.grading import whole_model_gsim as G

    out = Path(out)
    local = _local_for(model_capsule, target)
    engine = None
    if cache == "default":
        from merlin.common.artifacts import cache_dir

        cache = cache_dir("group-timing")
    try:
        engine = G._engine(target, emulator)["sha256"]
    except Exception:  # noqa: BLE001 -- no engine: every run refuses below, nothing is carried
        cache = None
    todo = [(g, p) for g, p in sorted(programs.items()) if not p.get("refusal")]
    results: dict[int, dict[str, Any]] = {
        g: {"group": g, "status": "refused", "refusal": p["refusal"]} for g, p in programs.items() if p.get("refusal")
    }

    def carried(program: Mapping[str, Any]) -> Path | None:
        if cache is None or not program.get("elf_sha256") or not engine:
            return None
        # A grade is a function of the program AND of the contract it was held to: a contract (in the
        # memory map, not the ELF) keys its own entry, so a changed contract is never served an old grade.
        try:
            rows = json.loads(Path(str(program.get("memory_map"))).read_text(encoding="utf-8")).get("groups") or []
        except (OSError, ValueError):
            rows = []
        contract = next((r.get("exactness") for r in rows if isinstance(r, Mapping) and r.get("exactness")), None)
        suffix = (
            "." + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16] if contract else ""
        )
        return Path(cache) / engine[:16] / f"{program['elf_sha256']}{suffix}.json"

    def one(group: int, program: Mapping[str, Any]) -> dict[str, Any]:
        kept = carried(program)
        if kept is not None and kept.is_file():
            row = json.loads(kept.read_text(encoding="utf-8"))
            return {**row, "group": group, "carried": True}
        started = time.monotonic()
        verdict = G.run_gsim_whole_model(
            program["elf"],
            program["memory_map"],
            program["oracle"],
            target=target,
            out=out / f"g{group}",
            emulator=emulator,
            max_cycles=max_cycles,
            timeout_s=timeout_s,
            local=local,
        )
        row = {
            "group": group,
            "status": verdict.get("status"),
            "refusal": verdict.get("refusal"),
            "wall_s": round(time.monotonic() - started, 1),
            "elf_sha256": program.get("elf_sha256"),
            "emulator_sha256": (verdict.get("emulator") or {}).get("sha256"),
            "machine": (verdict.get("machine") or {}).get("registry_entry"),
        }
        if verdict.get("status") == "graded":
            (entry,) = verdict["per_group"]
            grade = verdict.get("grade") or {}
            row.update(
                {
                    "kind": entry["kind"],
                    "cycles": entry["cycles"],
                    "correct": bool(entry["correct"]),
                    "failure": (grade.get("disagree") or grade.get("unverified") or [None])[0],
                    "chained_digest_agrees": str(group) in ((grade.get("chained") or {}).get("agree") or ()),
                    # Which exactness contract the grade held the group to, and its own numbers.
                    "exactness": (grade.get("contracts") or {}).get(str(group)),
                    "evidence": (grade.get("evidence") or {}).get(str(group)),
                }
            )
            if kept is not None:
                kept.parent.mkdir(parents=True, exist_ok=True)
                kept.write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
        # Where THIS run's console, dump and verdict are. Not cached: a carried row names no run of its own.
        return {**row, "run_dir": str(out / f"g{group}")}

    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        pending, running = list(todo), {}
        while pending or running:
            while pending and _slots_free(max_parallel, len(running)):
                group, program = pending.pop(0)
                running[group] = pool.submit(one, group, program)
            finished = [g for g, f in running.items() if f.done()]
            for group in finished:
                try:
                    results[group] = running.pop(group).result()
                except Exception as error:  # noqa: BLE001 -- recorded per group, never dropped
                    results[group] = {
                        "group": group,
                        "status": "refused",
                        "refusal": f"{type(error).__name__}: {error}",
                    }
            if not finished:
                time.sleep(5)
    return results


def _local_for(model_capsule: str | Path, target: str) -> Any:
    from merlin.analysis.whole_model.grading.group_programs import reference_context

    try:
        return reference_context(model_capsule, target=target)["local"]
    except Exception:  # noqa: BLE001 -- no local reference: an exact group is reported unverified
        return None


def changed_group_timing(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    target: str,
    model_capsule: str | Path,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    out: str | Path,
    groups: Sequence[int] | None = None,
    max_parallel: int = 8,
    jobs: int = 8,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    harness_overrides: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Tier B end to end: the changed groups of ``candidate`` against ``baseline``, both timed.

    ``candidate`` / ``baseline`` are ``{"package": dir or None, "record": build record, "decline": [...]}``
    (the record is the whole-model build record the groups are diffed on). ``groups`` overrides the
    diff. Returns per-group rows (candidate and baseline cycles on the emulator, the ratio, the
    direction, each side's local correctness) with the device labelled as its own.
    """
    from merlin.analysis.whole_model.grading.group_programs import (
        DEVICE_LABEL,
        SCHEMA,
        _attribution,
        build_group_programs,
        changed_groups,
    )

    out = Path(out)
    diff = changed_groups(candidate["record"], baseline["record"])
    chosen = [int(g) for g in (groups if groups is not None else diff["changed"])]
    started = time.monotonic()
    sides: dict[str, dict[int, dict[str, Any]]] = {}
    built: dict[str, dict[int, dict[str, Any]]] = {}
    for name, side in (("candidate", candidate), ("baseline", baseline)):
        objects = {g: r.get("object_sha256") for g, r in _attribution(side["record"]).items() if r.get("object_sha256")}
        built[name] = build_group_programs(
            side.get("package"),
            chosen,
            model_capsule=model_capsule,
            target=target,
            machine=machine,
            header=header,
            header_sha256=header_sha256,
            out=out / name,
            decline=side.get("decline") or (),
            expect_objects=None if side.get("allow_object_drift") else objects,
            jobs=jobs,
            phase0_recipe=phase0_recipe,
            descriptor=descriptor,
            harness_overrides=harness_overrides,
        )
    build_s = round(time.monotonic() - started, 1)
    together = {("candidate", g): p for g, p in built["candidate"].items()}
    together.update({("baseline", g): p for g, p in built["baseline"].items()})
    keyed = {index: program for index, program in enumerate(together.values())}
    names = dict(enumerate(together))
    timed = time_group_programs(
        keyed, target=target, model_capsule=model_capsule, out=out / "runs", max_parallel=max_parallel
    )
    for index, row in timed.items():
        side, group = names[index]
        sides.setdefault(side, {})[group] = {**row, "group": group}
    rows = []
    for group in chosen:
        a, b = sides.get("candidate", {}).get(group, {}), sides.get("baseline", {}).get(group, {})
        row = {
            "group": group,
            "kind": a.get("kind") or b.get("kind") or diff["kinds"].get(group),
            "candidate_cycles": a.get("cycles"),
            "baseline_cycles": b.get("cycles"),
            "candidate_correct": a.get("correct"),
            "baseline_correct": b.get("correct"),
            "candidate_refusal": a.get("refusal"),
            "baseline_refusal": b.get("refusal"),
            "wall_s": {"candidate": a.get("wall_s"), "baseline": b.get("wall_s")},
            "why_changed": diff["why"].get(group),
        }
        if row["candidate_cycles"] and row["baseline_cycles"]:
            row["ratio"] = round(row["candidate_cycles"] / row["baseline_cycles"], 4)
            row["direction"] = "faster" if row["ratio"] < 1 else "slower" if row["ratio"] > 1 else "unchanged"
        rows.append(row)
    document = {
        "schema": SCHEMA,
        "device": DEVICE_LABEL,
        "feeds_objective": False,
        "target": target,
        "machine": machine,
        "diff": diff,
        "groups": rows,
        "build_wall_s": build_s,
        "total_wall_s": round(time.monotonic() - started, 1),
        "max_parallel": max_parallel,
    }
    (out / "group_timing.json").write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    return document


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (vx * vy), 4) if vx and vy else None


def direction_agreement(
    rows: Sequence[Mapping[str, Any]],
    board: Mapping[int, tuple[int, int]],
    *,
    threshold: float = 0.05,
    poor: float = 0.7,
) -> dict[str, Any]:
    """How often the emulator's per-group direction matches the board's, overall and per kind.

    ``rows`` are :func:`changed_group_timing` rows; ``board`` is ``{group: (candidate, baseline)}`` board
    cycles for the same two programs. Only groups whose BOARD cycles moved by more than ``threshold``
    count (a change inside the noise has no direction). A kind agreeing on fewer than ``poor`` of its
    groups is flagged ``direction_unreliable``. Also the rank correlation of the two log-ratios.
    """
    import math

    counted, agree, by_kind = [], 0, {}
    for row in rows:
        group = int(row["group"])
        if group not in board or not row.get("ratio"):
            continue
        cand, base = board[group]
        if not cand or not base:
            continue
        board_ratio = cand / base
        if abs(board_ratio - 1) <= threshold:
            continue
        same = (board_ratio < 1) == (float(row["ratio"]) < 1)
        agree += same
        kind = str(row.get("kind"))
        slot = by_kind.setdefault(kind, {"n": 0, "agree": 0})
        slot["n"] += 1
        slot["agree"] += int(same)
        counted.append((math.log(float(row["ratio"])), math.log(board_ratio)))
    for slot in by_kind.values():
        slot["agreement"] = round(slot["agree"] / slot["n"], 3)
        slot["direction_unreliable"] = slot["agreement"] < poor
    return {
        "groups_counted": len(counted),
        "threshold": threshold,
        "agreement": round(agree / len(counted), 3) if counted else None,
        "spearman_log_ratio": _spearman([c[0] for c in counted], [c[1] for c in counted]),
        "by_kind": by_kind,
    }


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- group_efficiency --------------------------------------------------------------------------
# Per-group EFFICIENCY diagnostics: what a group's program issued, counted per execution, against its
# derived roofline -- numbers that say WHAT is inefficient and never HOW to fix it.
#
#     census = dynamic_census(elf, target=t, compiler=cc, group_objects={g: obj}, functional_model=spec, out=d)
#     report = efficiency(census[g], roofline=bound, cycles=measured)
#
# WHY DYNAMIC. A static census counts instruction WORDS in the code. A kernel that loops issues a word
# once per trip, so the static count says nothing about how often anything ran: measured on one 1x1
# group, 784 weight-load words in the code issued 3,136 times per invocation. Per-output-tile counts
# from the static census would be fiction. So the census is static in what it DECODES -- each word's
# role comes from the target's own ISA facts, by address, from the program's own disassembly -- and
# dynamic in what it COUNTS: every address is weighted by how often the functional model executed it
# (its PC histogram), divided by the invocations the program itself reports running.
#
# WHAT IS COUNTED. Each accelerator instruction by its DERIVED ROLE (configuration, operand load, weight
# load, accumulate, readout, sync), never by a name or an encoding, plus the host's memory-ordering
# instruction, which on a co-processor stalls the host until the accelerator drains. That one is the
# host ISA's, named by the disassembler that ships with the compiler (:data:`HOST_ORDERING_MNEMONIC`);
# it is not a fact about any accelerator. Only code in the group's own kernel object is counted.
#
# WHAT IS REPORTED (:func:`efficiency`): computes issued against the roofline's minimum, cycles per
# compute, cycles against the roofline, and each role per OUTPUT TILE (the roofline's tile count). The
# cycles are the measurement's; the census is the functional model's, which runs the same program bytes.
# Nothing here names a target.

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
    from merlin.analysis.whole_model.grading import isa_prohibition as ISA

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


__all__ = [
    "HOST_ORDERING_MNEMONIC",
    "REPORTED_ROLES",
    "SCHEMA",
    "dynamic_census",
    "efficiency",
    "invocations",
    "pc_histogram",
]
