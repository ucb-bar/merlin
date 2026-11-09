"""Per-group perf capsules on the BOARD: both arms' one-group programs and the whole-model reference
control in ONE board batch, each count admitted only on its own evidence.

Each program is built twice -- its timing build (``words``) and a locally verified twin (``local``)
that must be the same program outside the verification fence (:func:`window_identity`).  The twin is
graded on the functional model (:func:`local_grade`), the package arm's timing ELF is held to its rule,
and only a program that clears all three is linked into the batch.  A board count is admitted when its
block printed the group, its words equal its functionally graded twin's, the twin graded correct, and
the batch's control passes this mode's drift rule (:func:`.batch.control_check`): when the control
drifts, NOTHING in the batch is admitted, because the batch is then a statement about the board.

The board spec is the whole-model job's own ``machine.timing`` minus any host preparation step (an
operator's glue for the run that owns it), keeping the shared load-path lock and the queue command.
Linking and splitting a batch are the target driver's (``link_batch`` / ``split_batch``).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import group_capsules as GC

BOARD_DEVICE = "fpga_firesim"


def board_machine_spec(timing: Mapping[str, Any]) -> dict[str, Any]:
    """The board spec a per-group batch runs under, without the run owner's host preparation step."""
    spec = {k: v for k, v in dict(timing).items() if k != "prepare_command"}
    for key in ("hw_config", "lock_path", "queue_command"):
        if not spec.get(key):
            raise GC.GroupCapsuleError(f"the board spec names no {key}; it cannot share the board safely")
    return spec


def window_identity(record: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """What must agree between a board build and its locally graded twin: the program outside its
    verification fence, the kernel objects linked in, the compiler and flags, and the ABI header."""
    from merlin.runtime.backends import base as backends

    strip = backends.whole_model_driver(target).program.program_without_verification
    source = record.get("program_source")
    if not source:
        raise GC.GroupCapsuleError(f"the program record for g{record.get('group')} names no program_source")
    window = strip(Path(str(source)).read_text(encoding="utf-8"))
    variant = record.get("variant") or {}
    program = variant.get("program") or {}
    return {
        "window_source_sha256": hashlib.sha256(window.encode("utf-8")).hexdigest(),
        "linked_objects": sorted(
            hashlib.sha256(Path(str(o)).read_bytes()).hexdigest() for o in variant.get("objects") or ()
        ),
        "compiler": program.get("compiler"),
        "flags": program.get("flags"),
        "abi_header_sha256": (record.get("abi_header") or {}).get("sha256"),
    }


def local_grade(
    programs: Mapping[str, Mapping[str, Any]], *, local: Mapping[str, Any], out: str | Path, timeout_s: float = 3600
) -> dict[str, dict[str, Any]]:
    """Run each locally verified program on the functional model ``local`` (a machine spec): correct
    only when the program printed its local line with zero mismatches over a nonzero count, and its words."""
    from merlin.perf import whole_model_verdict as V

    from .machines import machine_from_spec

    machine = machine_from_spec(dict(local))
    graded: dict[str, dict[str, Any]] = {}
    for label, record in sorted(programs.items()):
        if record.get("refusal"):
            graded[label] = {"correct": False, "refusal": record["refusal"]}
            continue
        run = machine.run(Path(str(record["elf"])), Path(out) / label, timeout_s=timeout_s)
        group = str(record["group"])
        row: dict[str, Any] = {"run": {k: v for k, v in run.items() if k != "device"}, "device": run.get("device")}
        if not run.get("completed"):
            row.update(
                correct=False, refusal=f"the functional-model run did not complete: {run.get('incomplete_reason')}"
            )
        else:
            parsed = V.parse_log(Path(run["uart_log"]).read_text(encoding="utf-8", errors="replace"))
            checked = parsed.local.get(group)
            row["words"] = list(parsed.words[group]) if group in parsed.words else None
            if checked is None:
                row.update(correct=False, refusal=f"the program printed no local check for group {group}")
            else:
                mismatches, elements = checked
                row.update(correct=mismatches == 0 and elements > 0, mismatches=mismatches, elements=elements)
            if row["words"] is None:
                row.update(correct=False, refusal=f"the program printed no words for group {group}")
        graded[label] = row
    return graded


def board_batch(
    entries: Mapping[str, Mapping[str, Any]],
    *,
    control: Mapping[str, Any] | None,
    machine: Mapping[str, Any],
    target: str,
    out: str | Path,
    timeout_s: float = 3600,
    runner: Any = None,
) -> dict[str, Any]:
    """Link every entry (``{label: {"timing": record, "local": graded row}}``) and the control into one
    board program, run it ONCE, and admit each entry's group count on its own evidence."""
    from merlin.perf import whole_model_verdict as V
    from merlin.runtime.backends import base as backends

    from .batch import control_check
    from .machines import machine_from_spec

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    driver = backends.whole_model_driver(target).program
    variants = []
    for label in sorted(entries):
        timing = entries[label]["timing"]
        if timing.get("refusal"):
            raise GC.GroupCapsuleError(f"{label}: its board program was refused: {timing['refusal']}")
        variants.append({**dict(timing["variant"]), "label": label})
    control_index = None
    solo_cycles, solo_words = None, {}
    if control is not None:
        request = json.loads(Path(str(control["board_request"])).read_text(encoding="utf-8"))
        solo = json.loads(Path(str(control["solo_result"])).read_text(encoding="utf-8"))
        solo_cycles = (solo.get("verdict") or {}).get("whole_window_cycles")
        uart = Path(str((solo.get("run") or {}).get("uart_log") or ""))
        solo_words = V.parse_log(uart.read_text(encoding="utf-8", errors="replace")).words if uart.is_file() else {}
        variants.insert(0, {**dict(request["variant"]), "label": "control"})
        control_index = 1
    first = variants[0]["program"]
    linked = driver.link_batch(
        variants,
        out / "link",
        compiler=Path(str(first["compiler"])),
        compile_flags=first["flags"],
        link_flags=first["link_flags"],
        link_script=first["link_script"],
        supports=variants[0]["supports"],
    )
    board = machine_from_spec(dict(machine))
    if runner is not None:
        board._run = runner  # test seam: the queue is never reached from a unit test
    started = time.time()
    run = board.run(Path(linked["elf"]), out / "run", timeout_s=timeout_s * max(1, len(variants)))
    record: dict[str, Any] = {
        "schema": GC.SCHEMA,
        "linked": linked,
        "variants": [v["label"] for v in variants],
        "run": {k: v for k, v in run.items() if k != "device"},
        "device": run.get("device"),
        "wall_s": round(time.time() - started, 1),
    }
    text = Path(run["uart_log"]).read_text(encoding="utf-8", errors="replace") if run.get("completed") else ""
    blocks = driver.split_batch(text, len(variants)) if run.get("completed") else {}
    if control_index is not None:
        record["control"] = control_check(dict(control), blocks.get(control_index), int(solo_cycles or 0), solo_words)
    trusted = bool(run.get("completed")) and (control_index is None or record["control"]["ok"])
    rows: dict[str, dict[str, Any]] = {}
    for position, variant in enumerate(variants, 1):
        label = variant["label"]
        if label == "control":
            continue
        entry = entries[label]
        group = str(entry["timing"]["group"])
        row: dict[str, Any] = {
            "group": entry["timing"]["group"],
            "arm": entry["timing"].get("arm"),
            "position": position,
        }
        block = blocks.get(position)
        if block is not None:
            (out / f"{label}.uart.log").write_text(block, encoding="utf-8")
        if not run.get("completed"):
            row.update(admitted=False, refusal=f"the batch did not complete: {run.get('incomplete_reason')}")
        elif block is None:
            row.update(admitted=False, refusal=f"variant {position} printed no complete block")
        else:
            parsed = V.parse_log(block)
            line = parsed.groups.get(group)
            words = list(parsed.words[group]) if group in parsed.words else None
            twin = entry.get("local") or {}
            row.update(cycles=line.cycles if line else None, words=words, local_words=twin.get("words"))
            problems = []
            if line is None:
                problems.append(f"the block printed no line for group {group}")
            if not twin.get("correct"):
                problems.append(
                    f"its functional-model twin is not correct ({twin.get('refusal') or twin.get('mismatches')})"
                )
            if words is None or words != twin.get("words"):
                problems.append("the board's words differ from its functional-model twin's")
            if not trusted:
                problems.append("the batch's control drifted, so no count in it is trusted")
            row.update(admitted=not problems, refusal="; ".join(problems) or None)
        row["device"] = BOARD_DEVICE
        rows[label] = row
    record["rows"] = rows
    (out / "board_batch.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    return record


def measure_on_board(
    arms: Mapping[str, GC.Arm],
    groups: Sequence[int],
    *,
    package_dir: str | Path | None,
    model_capsule: str | Path,
    target: str,
    out: str | Path,
    machine: Mapping[str, Any],
    local: Mapping[str, Any],
    control: Mapping[str, Any] | None,
    timeout_s: float = 1800,
    submit: bool = True,
    model: str | None = None,
    runner: Any = None,
) -> dict[str, Any]:
    """Both arms on the board in ONE batch with the reference control (see the module docstring).
    ``submit=False`` stops before the board and returns the prepared evidence."""
    out = Path(out)
    built = {}
    for name, arm in arms.items():
        for verify in ("words", "local"):
            built[(name, verify)] = GC.build_arm_programs(
                arm, groups, package_dir=package_dir, model_capsule=model_capsule, target=target,
                out=out / name / verify, verify=verify,
            )  # fmt: skip
    prepared: dict[str, dict[str, Any]] = {}
    twins: dict[str, dict[str, Any]] = {}
    for name, arm in arms.items():
        for group in groups:
            timing, twin = built[(name, "words")][int(group)], built[(name, "local")][int(group)]
            label = GC.label_of(name, group, model)
            entry: dict[str, Any] = {
                "timing": timing,
                "row": {"label": label, "model": model, **GC.program_row(timing)},
            }
            prepared[label] = entry
            refusal = timing.get("refusal") or twin.get("refusal")
            if refusal:
                entry["refusal"] = refusal
                continue
            if window_identity(timing, target=target) != window_identity(twin, target=target):
                entry["refusal"] = "the timing build and its locally verified twin differ outside verification"
                continue
            if arm.prohibited_roles:
                scan = GC.isa_scan(timing, target=target, roles=arm.prohibited_roles)
                entry["row"].update(
                    isa_clean=bool(scan.get("clean")), census=(scan.get("census") or {}).get("per_group")
                )
                if not scan.get("clean"):
                    entry["refusal"] = "isa_prohibited: " + ", ".join(sorted(scan.get("summary") or {}))
                    continue
            twins[label] = twin
    graded = local_grade(twins, local=local, out=out / "functional_model")
    for label, row in graded.items():
        prepared[label]["local"] = row
        if not row.get("correct"):
            prepared[label]["refusal"] = (
                f"its functional-model twin is not correct: {row.get('refusal') or row.get('mismatches')}"
            )
    ready = {label: e for label, e in prepared.items() if not e.get("refusal")}
    document: dict[str, Any] = {
        "schema": GC.SCHEMA,
        "device": BOARD_DEVICE,
        "arms": {k: a.to_dict() for k, a in arms.items()},
    }
    batch = None
    if submit and ready:
        batch = board_batch(
            {label: {"timing": e["timing"], "local": e["local"]} for label, e in ready.items()},
            control=control, machine=machine, target=target, out=out / "batch", timeout_s=timeout_s, runner=runner,
        )  # fmt: skip
        document.update(control=batch.get("control"), run=batch.get("run"), board_device=batch.get("device"))
    rows = []
    for label, entry in sorted(prepared.items()):
        row = {**entry["row"], "device": BOARD_DEVICE, "local": entry.get("local")}
        if entry.get("refusal"):
            row.update(admitted=False, refusal=entry["refusal"])
        elif batch is None:
            row.update(admitted=False, refusal="prepared; not submitted")
        else:
            row.update((batch.get("rows") or {}).get(label) or {"admitted": False, "refusal": "no batch row"})
        rows.append(row)
    document["rows"] = rows
    out.mkdir(parents=True, exist_ok=True)
    (out / "board_rows.json").write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    return document


__all__ = ["BOARD_DEVICE", "board_batch", "board_machine_spec", "local_grade", "measure_on_board", "window_identity"]
