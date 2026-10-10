"""EXECUTED accelerator-command profile of one program, from its functional run's commit log.

A static decode of the emitted code (``instruction_trace.json``) counts each accelerator instruction
once however many times a loop executes it, so it cannot say how many commands a program actually
issued. The functional engine's commit log can: every retired instruction is printed with the values it
wrote, so replaying the integer register writes recovers each executed accelerator instruction's source
operand VALUES. This module turns that log into an executed trace in the same shape as the static one
(``{"instructions": [{"index", "class", "funct", "rs1", "rs2", "decoded"}]}``) and summarises it:

* ``by_class`` -- executed accelerator commands per instruction class (the contract's
  ``encoding.semantic_class`` vocabulary, refined by the generic RoCC decoder);
* ``retired_instructions`` -- every instruction the host retired in the run (harness included);
* ``local_high_water`` -- the highest scratchpad / accumulator row any executed command addressed,
  against the RTL-derived capacities (the generic local-address bounds rule run over the executed trace).

Every value is decoded through :mod:`merlin.targetgen.rocc.semantics` and the target's RTL facts and
contract; nothing here names an accelerator, an opcode or a class. The RoCC transport fields of an
R-type instruction (``opcode``, ``rs1``, ``rs2``, ``funct7``) are the RISC-V encoding, not a target
fact. Anything underivable leaves the profile UNKNOWN with its reason.

The commit-log line format is the functional simulator's ``-l --log-commits`` text: an execution line
``core N: 0x<pc> (0x<insn>) <disasm>`` and a commit line ``core N: <priv> 0x<pc> (0x<insn>) [x<r>
0x<value>] ...``. Only commit lines are read, and only integer-register writes are replayed.
"""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "merlin_executed_command_profile_v1"
#: Bound on retired instructions read from one log; beyond it the profile is UNKNOWN, never partial.
DEFAULT_MAX_RETIRED = 200_000_000

_OPCODE_MASK = 0x7F
_FULL_WIDTH_LOW_BITS = 0b11  # an uncompressed RISC-V instruction has both low bits set


def _hex(token: str) -> int | None:
    token = token.strip("()")
    if not token.startswith("0x"):
        return None
    try:
        return int(token, 16)
    except ValueError:
        return None


def executed_trace(
    lines: Iterable[str],
    *,
    isa: Mapping[str, Any],
    max_retired: int = DEFAULT_MAX_RETIRED,
) -> dict[str, Any]:
    """The executed accelerator trace and retired-instruction count of one commit log."""
    from merlin.targetgen.rocc import semantics

    opcode = isa.get("CUSTOM_OPCODE")
    if isinstance(opcode, bool) or not isinstance(opcode, int):
        return {"status": "unknown", "why": "the RTL facts give no custom opcode to recognise commands"}
    regs = [0] * 32
    retired = 0
    instructions: list[dict[str, Any]] = []
    for raw in lines:
        parts = raw.split()
        # commit line: core <hart>: <priv> 0x<pc> (0x<insn>) [writes...]
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        insn = _hex(parts[4])
        if insn is None:
            continue
        retired += 1
        if retired > max_retired:
            return {"status": "unknown", "why": f"the run retired more than {max_retired} instructions"}
        if insn & _FULL_WIDTH_LOW_BITS == _FULL_WIDTH_LOW_BITS and insn & _OPCODE_MASK == opcode:
            rs1_index, rs2_index = (insn >> 15) & 31, (insn >> 20) & 31
            funct = insn >> 25
            rs1 = {"raw": regs[rs1_index], "kind": "const"}
            rs2 = {"raw": regs[rs2_index], "kind": "const"}
            cls, decoded = semantics.decode_instruction(funct, rs1, rs2, dict(isa))
            instructions.append(
                {"index": len(instructions), "class": cls, "funct": funct, "rs1": rs1, "rs2": rs2, "decoded": decoded}
            )
        writes = parts[5:]
        for at in range(len(writes) - 1):
            name = writes[at]
            if len(name) > 1 and name[0] == "x" and name[1:].isdigit():
                value = _hex(writes[at + 1])
                index = int(name[1:])
                if value is not None and 0 < index < 32:
                    regs[index] = value
    return {"status": "measured", "instructions": instructions, "retired_instructions": retired}


def summarize(trace: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """``by_class``, counts and local-memory high water of an executed trace."""
    if trace.get("status") != "measured":
        return {"schema": SCHEMA, "status": "unknown", "why": str(trace.get("why") or "no executed trace")}
    instructions = list(trace.get("instructions") or ())
    by_class = Counter(str(row.get("class")) for row in instructions)
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "measured",
        "retired_instructions": int(trace.get("retired_instructions") or 0),
        "accelerator_commands": len(instructions),
        "by_class": dict(sorted(by_class.items())),
        "local_high_water": local_high_water(trace, target=target),
    }
    return out


def local_high_water(trace: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """Highest scratchpad / accumulator row (exclusive) the executed commands addressed, with the
    RTL-derived capacities, from the generic local-address bounds rule over the executed trace."""
    try:
        from merlin.targetgen import rtl_checks_generic as G

        facts = G.load_default_facts(target)
        check = G._check_local_address_bounds(dict(trace), facts, G.protocol_for(target))
    except Exception as exc:  # noqa: BLE001 - an unavailable protocol leaves occupancy UNKNOWN
        return {"status": "unknown", "why": f"local-address protocol unavailable ({type(exc).__name__})"}
    evidence = getattr(check, "evidence", None) or {}
    if getattr(check, "status", None) == "skipped" or not evidence:
        return {"status": "unknown", "why": str(getattr(check, "message", "") or "no decodable local address")}

    def share(used: Any, capacity: Any) -> float | None:
        if isinstance(used, int) and isinstance(capacity, int) and capacity > 0:
            return round(used / capacity, 4)
        return None

    spad, spad_cap = evidence.get("scratchpad_max_row_exclusive"), evidence.get("scratchpad_rows_capacity")
    acc, acc_cap = evidence.get("accumulator_max_row_exclusive"), evidence.get("accumulator_rows_capacity")
    touched = _rows_touched(trace, facts, G.protocol_for(target))
    return {
        "status": "measured",
        "scratchpad_rows_high_water": spad,
        "scratchpad_rows_touched": touched.get("scratchpad"),
        "scratchpad_rows_capacity": spad_cap,
        "scratchpad_touched_share": share(touched.get("scratchpad"), spad_cap),
        "accumulator_rows_high_water": acc,
        "accumulator_rows_touched": touched.get("accumulator"),
        "accumulator_rows_capacity": acc_cap,
        "accumulator_touched_share": share(touched.get("accumulator"), acc_cap),
    }


def _rows_touched(trace: Mapping[str, Any], facts: Mapping[str, Any], protocol: Any) -> dict[str, int | None]:
    """Distinct rows of each local memory the executed commands addressed (the union of every
    addressed row range), read through the protocol's declared local-address fields."""
    select = facts.get("accumulator_select_bit")
    spad_mask, acc_mask = facts.get("scratchpad_row_mask"), facts.get("accumulator_row_mask")
    sentinel = facts.get("local_address_sentinel")
    if not isinstance(spad_mask, int):
        return {"scratchpad": None, "accumulator": None}
    rows: dict[str, set[int]] = {"scratchpad": set(), "accumulator": set()}
    for inst in trace.get("instructions") or ():
        decoded = inst.get("decoded") or {}
        for entry in protocol.local_addresses:
            if inst.get("class") not in protocol.classes(str(entry["role"])):
                continue
            addr = decoded.get(entry["address"])
            if not isinstance(addr, int) or isinstance(addr, bool) or addr == sentinel:
                continue
            count = decoded.get(entry["rows"]) if entry.get("rows") else 1
            if not isinstance(count, int) or count <= 0:
                continue
            if isinstance(select, int) and select and addr & select and isinstance(acc_mask, int):
                rows["accumulator"].update(range(addr & acc_mask, (addr & acc_mask) + count))
            else:
                rows["scratchpad"].update(range(addr & spad_mask, (addr & spad_mask) + count))
    return {name: len(found) for name, found in rows.items()}


def trace_invocation(target: str, elf: Path) -> tuple[list[str], dict[str, str]] | None:
    """``(argv, env)`` that runs ``elf`` on the target's functional engine with a commit log on stderr.

    Read from the target support backend: ``functional_trace_invocation(elf) -> (argv, env)`` when it
    provides one, else its ``spike_path()`` and ``spike_extension() -> (flags, library_dir)``. None when
    the backend exposes neither (the profile is then UNKNOWN)."""
    from merlin.runtime.backends.base import get_backend

    backend = get_backend(target)
    hook = getattr(backend, "functional_trace_invocation", None)
    if callable(hook):
        argv, env = hook(Path(elf))
        return list(argv), dict(env)
    path, extension = getattr(backend, "spike_path", None), getattr(backend, "spike_extension", None)
    if not callable(path) or not callable(extension):
        return None
    flags, library_dir = extension()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(library_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
    return [str(path()), *[str(f) for f in flags], "-l", "--log-commits", str(elf)], env


def profile_elf(elf: Path, *, target: str, timeout_s: int = 120, max_retired: int = DEFAULT_MAX_RETIRED) -> dict:
    """Run ``elf`` once on the functional engine with a commit log and summarise what it executed."""
    from merlin.targetgen.rocc import semantics

    try:
        invocation = trace_invocation(target, Path(elf))
    except Exception as exc:  # noqa: BLE001
        return {"schema": SCHEMA, "status": "unknown", "why": f"no functional trace engine ({type(exc).__name__})"}
    if invocation is None:
        return {"schema": SCHEMA, "status": "unknown", "why": "the target's functional engine exposes no trace run"}
    try:
        isa = semantics.isa_constants(target)
    except Exception as exc:  # noqa: BLE001
        return {"schema": SCHEMA, "status": "unknown", "why": f"the RoCC decode is unavailable ({type(exc).__name__})"}
    argv, env = invocation
    try:
        process = subprocess.Popen(  # noqa: S603 - argv is the target's own declared engine invocation
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True, errors="replace"
        )
    except OSError as exc:
        return {"schema": SCHEMA, "status": "unknown", "why": f"the functional engine did not start ({exc})"}
    import threading

    timer = threading.Timer(timeout_s, process.kill)
    timer.start()
    try:
        assert process.stderr is not None
        trace = executed_trace(process.stderr, isa=isa, max_retired=max_retired)
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
    if trace.get("status") == "measured" and process.returncode != 0:
        trace = {"status": "unknown", "why": f"the functional run exited {process.returncode} (timeout {timeout_s}s)"}
    return summarize(trace, target=target)
