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

Every value is decoded through :mod:`merlin.target.isa.rocc.semantics` and the target's RTL facts and
contract; nothing here names an accelerator, an opcode or a class. The RoCC transport fields of an
R-type instruction (``opcode``, ``rs1``, ``rs2``, ``funct7``) are the RISC-V encoding, not a target
fact. Anything underivable leaves the profile UNKNOWN with its reason.

The commit-log line format is the functional simulator's ``--log-commits`` text: a commit line
``core N: <priv> 0x<pc> (0x<insn>) [x<r> 0x<value>] ...`` per retired instruction (with ``-l`` each is
also preceded by an execution line ``core N: 0x<pc> (0x<insn>) <disasm>``, which readers skip). Only
commit lines are read, and only integer-register writes are replayed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from merlin.common import digest as _mjson

# ---- executed_trace ----------------------------------------------------------------------------

SCHEMA = "merlin_executed_command_profile_v1"
#: Bound on retired instructions read from one log; beyond it the profile is UNKNOWN, never partial.
DEFAULT_MAX_RETIRED = 200_000_000

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
    from merlin.analysis.observation.rocc_commands import fields_of
    from merlin.target.isa.rocc import semantics

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
        fields = fields_of(insn)
        if insn & _FULL_WIDTH_LOW_BITS == _FULL_WIDTH_LOW_BITS and fields["opcode"] == opcode:
            rs1_index, rs2_index = fields["rs1"], fields["rs2"]
            funct = fields["funct"]
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
        from merlin.target.isa.rocc import rtl_checks_generic as G

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
    from merlin.runtime.backends.registry import get_backend

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
    return [str(path()), *[str(f) for f in flags], "--log-commits", str(elf)], env


def profile_elf(elf: Path, *, target: str, timeout_s: int = 120, max_retired: int = DEFAULT_MAX_RETIRED) -> dict:
    """Run ``elf`` once on the functional engine with a commit log and summarise what it executed."""
    from merlin.target.isa.rocc import semantics

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


# ---- streamed_executed_trace -------------------------------------------------------------------
# Incremental exact register replay for complete committed command streams.
#
# Every input line is consumed. Retired CPU instructions use constant state;
# commands and fences are yielded in their original order without a trace list.

#: Retired-instruction bound of one complete source trace.
DEFAULT_MAX_RETIRED = 200_000_000
#: Issued accelerator commands plus fences of one complete source trace.
DEFAULT_MAX_COMMANDS = 4_000_000


def iter_source_commands(view, *, isa, max_retired=DEFAULT_MAX_RETIRED, max_commands=DEFAULT_MAX_COMMANDS, raw=False):
    """Yield each executed command (decoded) and each fence in order.

    ``raw=True`` yields ``(funct, rs1, rs2)`` integer tuples and ``None`` per fence instead, for a
    consumer that reads operand slots from derived facts and needs no per-command decode.
    """
    from merlin.target.isa.rocc import semantics

    opcode = isa.get("CUSTOM_OPCODE")
    if (
        type(opcode) is not int
        or not 0 < opcode < 128
        or any(type(v) is not int or v < 1 for v in (max_retired, max_commands))
    ):
        raise ValueError("streamed trace needs exact custom opcode and finite command/retirement bounds")
    registers = [0] * 32
    retired = commands = events = 0
    for line in view.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        if parts[1] != "0:":
            raise ValueError("complete source trace requires the original single hart")
        word = _hex(parts[4])
        if word is None:
            continue
        retired += 1
        if retired > max_retired:
            raise ValueError("complete source trace exceeds retired instruction limit")
        if word & 3 == 3 and word & 127 == opcode:
            if raw:
                events += 1
                if events > max_commands:
                    raise ValueError("complete source trace exceeds issued event limit")
                yield (word >> 25, registers[(word >> 15) & 31], registers[(word >> 20) & 31])
                commands += 1
                _write_registers(parts, registers)
                continue
            rs1 = {"raw": registers[(word >> 15) & 31], "kind": "const"}
            rs2 = {"raw": registers[(word >> 20) & 31], "kind": "const"}
            funct = word >> 25
            cls, decoded = semantics.decode_instruction(funct, rs1, rs2, dict(isa))
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield dict(index=commands, **{"class": cls}, funct=funct, rs1=rs1, rs2=rs2, decoded=decoded)
            commands += 1
        elif word & 127 == 15 and word >> 12 & 7 == 0:
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield None if raw else {"class": "FENCE"}
        if len(parts) > 6:
            _write_registers(parts, registers)
    if not commands:
        raise ValueError("complete source trace contains no accelerator commands")


def _write_registers(parts, registers):
    writes = parts[5:]
    for at in range(len(writes) - 1):
        name = writes[at]
        if len(name) > 1 and name[0] == "x" and name[1:].isdigit():
            value, index = _hex(writes[at + 1]), int(name[1:])
            if value is not None and 0 < index < 32:
                registers[index] = value


def iter_recorded_commands(records, *, isa, max_commands=DEFAULT_MAX_COMMANDS):
    """:func:`iter_source_commands` with ``raw=True`` over recorded ``(bits, rs1, rs2)`` retirements.

    ``records`` holds, in program order, every retired instruction of the kinds this reader yields (the
    command recorder's stream, :mod:`merlin.analysis.perf.measurement.command_recorder`) with the values its source
    registers held when it issued. The commit log's register replay gives exactly those values, so the
    yielded items, their order and every refusal are the commit-log reader's. The retired-instruction
    bound is the recorder's to check (it records the count); every other line of the commit log yields
    nothing.
    """
    opcode = isa.get("CUSTOM_OPCODE")
    if type(opcode) is not int or not 0 < opcode < 128 or type(max_commands) is not int or max_commands < 1:
        raise ValueError("streamed trace needs exact custom opcode and finite command/retirement bounds")
    commands = events = 0
    for word, rs1, rs2 in records:
        if word & 3 == 3 and word & 127 == opcode:
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield (word >> 25, rs1, rs2)
            commands += 1
        elif word & 127 == 15 and word >> 12 & 7 == 0:
            events += 1
            if events > max_commands:
                raise ValueError("complete source trace exceeds issued event limit")
            yield None
    if not commands:
        raise ValueError("complete source trace contains no accelerator commands")


# ---- capture_store -----------------------------------------------------------------------------
# A content-addressed store for engine cross-validation captures.
#
# A capture is one ELF run on the reference engine and the candidate engine and compared. It costs a
# REFERENCE-engine execution, which is the expensive half by more than an order of magnitude: building
# one 90-case functional certificate cost over ninety minutes of serial reference simulation, in front
# of a run that could not start without it.
#
# That work is perfectly reusable and was being thrown away. The engine outputs are a pure function of
# the bytes that were run -- the ELF -- and the engines that ran them. A stored *capture* also says that
# those outputs satisfy one canonical workload/oracle binding. Multiple descriptors can lower to the
# same ELF while carrying different bindings, so those records must coexist instead of overwriting one
# another.
#
# WHAT THE KEY MUST COVER, or a hit is a lie: the ELF digest, every engine pin, the workload identity,
# and the semantic-reference document. The same program on a rebuilt simulator is a different
# measurement -- that is the entire reason the pins exist -- and the same program presented under a
# different oracle is a different correctness assertion.
#
# A HIT IS RE-CHECKED, NOT TRUSTED. The stored document states the ELF and pins it was taken under, and
# those are compared against the ones being asked about before it is returned. A cache that hands back a
# document it did not verify is worse than no cache: it converts a stale answer into a fresh-looking one.
#
# PURGEABLE. This is a cache under ``out/artifacts/cache/``, never a product: every entry is exactly
# reproducible by running the two engines again, and deleting it costs time and no evidence.

#: The pins a capture's validity depends on. A capture is about these engines and no others.
_PIN_NAMES = ("gsim_binary", "gsim_firrtl", "gsim_model", "verilator_binary", "verilator_firrtl")


def store_root(target: str) -> Path:
    """``out/artifacts/cache/gsim_captures/<target>/`` -- created on demand."""
    from merlin.common.artifacts import cache_dir

    root = cache_dir("gsim_captures") / str(target)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pin_shas(pins: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _PIN_NAMES:
        entry = pins.get(name) if isinstance(pins, Mapping) else None
        sha = entry.get("sha256") if isinstance(entry, Mapping) else None
        if isinstance(sha, str) and sha:
            out[name] = sha
    return out


def _canonical(value: object) -> bytes:
    return _mjson.canonical_json(value, trailing_newline=True)


def capture_key(
    elf_sha256: str,
    pins: Mapping[str, Any],
    *,
    workload_sha256: str | None = None,
    semantic_reference: Mapping[str, Any] | None = None,
) -> str | None:
    """The store key, or ``None`` when the inputs do not fully determine a capture.

    Refuses rather than keying on a partial pin set: an entry filed under an incomplete key would be
    returned for a question it does not answer.
    """
    if not isinstance(elf_sha256, str) or len(elf_sha256) != 64:
        return None
    shas = _pin_shas(pins)
    if set(shas) != set(_PIN_NAMES):
        return None
    if (workload_sha256 is None) != (semantic_reference is None) or (
        workload_sha256 is not None and (not isinstance(workload_sha256, str) or len(workload_sha256) != 64)
    ):
        return None
    digest = hashlib.sha256()
    digest.update(elf_sha256.encode())
    for name in _PIN_NAMES:
        digest.update(b"\0")
        digest.update(f"{name}={shas[name]}".encode())
    if workload_sha256 is not None:
        try:
            reference = _canonical(semantic_reference)
        except (TypeError, ValueError):
            return None
        digest.update(b"\0workload_sha256=")
        digest.update(workload_sha256.encode())
        digest.update(b"\0semantic_reference=")
        digest.update(hashlib.sha256(reference).hexdigest().encode())
    return digest.hexdigest()


def _answers(
    document: Any,
    *,
    elf_sha256: str,
    pins: Mapping[str, Any],
    workload_sha256: str | None = None,
    semantic_reference: Mapping[str, Any] | None = None,
) -> bool:
    """Does this stored document answer the question being asked?"""
    if not isinstance(document, Mapping):
        return False
    if document.get("elf_sha256") != elf_sha256:
        return False
    want = _pin_shas(pins)
    for side, binary, firrtl in (
        ("reference", "verilator_binary", "verilator_firrtl"),
        ("candidate", "gsim_binary", "gsim_firrtl"),
    ):
        arm = document.get(side)
        if not isinstance(arm, Mapping):
            return False
        if arm.get("binary_sha256") != want.get(binary):
            return False
        if arm.get("firrtl_sha256") != want.get(firrtl):
            return False
    candidate = document.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("model_sha256") != want.get("gsim_model"):
        return False
    if workload_sha256 is not None and document.get("workload_sha256") != workload_sha256:
        return False
    if semantic_reference is not None and document.get("semantic_reference") != semantic_reference:
        return False
    return True


def lookup(
    target: str,
    *,
    elf_sha256: str,
    pins: Mapping[str, Any],
    workload_sha256: str | None = None,
    semantic_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """A stored capture for these exact bytes and engines, or ``None``."""
    key = capture_key(elf_sha256, pins, workload_sha256=workload_sha256, semantic_reference=semantic_reference)
    if key is None:
        return None
    root = store_root(target)
    paths = [root / f"{key}.json"]
    # Entries written before workload/oracle scoping used only the ELF and engine pins. Adopt one
    # only after checking the full requested binding, then promote it so another same-ELF workload
    # can no longer evict this result.
    if workload_sha256 is not None:
        legacy = capture_key(elf_sha256, pins)
        if legacy is not None and legacy != key:
            paths.append(root / f"{legacy}.json")
    for path in paths:
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _answers(
            document,
            elf_sha256=elf_sha256,
            pins=pins,
            workload_sha256=workload_sha256,
            semantic_reference=semantic_reference,
        ):
            continue
        if path != paths[0]:
            store(target, elf_sha256=elf_sha256, pins=pins, document=document)
        return dict(document)
    return None


def store(target: str, *, elf_sha256: str, pins: Mapping[str, Any], document: Mapping[str, Any]) -> Path | None:
    """File a capture. Returns the path, or ``None`` when it was not storable.

    A document that does not answer for the inputs it is filed under is REFUSED rather than written:
    the whole value of the store is that a hit needs no further checking.
    """
    workload_sha256 = document.get("workload_sha256")
    semantic_reference = document.get("semantic_reference")
    key = capture_key(elf_sha256, pins, workload_sha256=workload_sha256, semantic_reference=semantic_reference)
    if key is None or not _answers(
        document,
        elf_sha256=elf_sha256,
        pins=pins,
        workload_sha256=workload_sha256,
        semantic_reference=semantic_reference,
    ):
        return None
    path = store_root(target) / f"{key}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a reader never sees a half-written capture
    return path


def census(target: str) -> dict[str, Any]:
    """How much reference-engine time this store is currently holding."""
    root = store_root(target)
    entries = sorted(root.glob("*.json"))
    agree = 0
    for path in entries:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("agreement") == "AGREE":
                agree += 1
        except (OSError, ValueError):
            continue
    return {"root": str(root), "entries": len(entries), "agreeing": agree}


__all__ = [
    "capture_key",
    "census",
    "lookup",
    "store",
    "store_root",
]
