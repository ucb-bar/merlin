"""Which accelerator instructions a whole-model program actually emits, and whether any is prohibited.

A rule such as "no hardware loop-descriptor instructions" is a claim about the PROGRAM, not about a
package's source: the target's library, called for the groups the package declines, and the host code
around every group emit accelerator commands too. So the check reads the linked ELF itself. Every
instruction word whose major opcode is the target's custom opcode is decoded to its selector and
named from the target's own funct table; the prohibited set is every instruction the target's ISA
facts give a prohibited ROLE -- derived, never a list of names. Each hit is attributed to the function
that holds it, and a function to the group whose kernel object defines it; everything else is the
program's own code (the library calls for the groups routed to it, and host code).

Nothing here knows a target: the opcode, the selector field and the roles all come from the target's
derived facts, and the disassembler is the one beside the compiler that built the program.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "merlin_isa_prohibition_v1"
PROGRAM_CODE = "program"

#: The base ISA's R-type layout, which every custom-opcode instruction uses: the major opcode in the
#: low seven bits, the selector (funct7) in the top seven. A property of the instruction FORMAT the
#: disassembler also decodes by -- the opcode VALUE and each selector's meaning are the target's facts.
#: A linked program at least this large is scanned in address ranges by worker processes
#: (:mod:`merlin.perf.isa_scan`); a smaller one in one pass. The result is the same either way.
PARALLEL_SCAN_MIN_BYTES = 64 << 20
SCAN_PARTS = 16
MAJOR_OPCODE_MASK = (1 << 7) - 1
SELECTOR_SHIFT = 25


def prohibited_instructions(target: str, roles: Sequence[str]) -> dict[int, str]:
    """``{selector: name}`` of every instruction the target's ISA facts give one of ``roles``."""
    from .task_instruction_evidence import declared_instruction_set, target_instruction_facts

    declared = declared_instruction_set(target_instruction_facts(target))
    if declared.get("status") not in (None, "derived"):
        raise ValueError(f"the target's instruction set is not derived: {declared.get('reason')}")
    wanted = set(roles)
    out: dict[int, str] = {}
    for entry in declared.get("instructions") or ():
        if wanted.intersection(entry.get("roles") or ()):
            out[int(str(entry["funct"]), 0)] = str(entry["name"])
    return out


def custom_opcode(target: str) -> int:
    from .task_instruction_evidence import target_instruction_facts

    value = (target_instruction_facts(target).get("isa") or {}).get("CUSTOM_OPCODE")
    if not isinstance(value, int):
        raise ValueError("the target's facts declare no custom opcode")
    return value


def disassembler_for(compiler: str | Path) -> Path:
    """The disassembler that sits beside the compiler that built the program (same toolchain prefix)."""
    compiler = Path(str(compiler))
    stem, _sep, _tool = compiler.name.rpartition("-")
    candidate = compiler.with_name(f"{stem}-objdump" if stem else "objdump")
    if not candidate.is_file():
        raise ValueError(f"no disassembler beside {compiler}")
    return candidate


def listing(elf: Path, *, objdump: Path) -> list[dict[str, Any]]:
    """Every 32-bit instruction in ``elf``: its address (int), word (hex text), the disassembler's own
    mnemonic, and the function that holds it. Parsed structurally from the listing (address, tab, hex
    word, tab, mnemonic); a compressed (16-bit) instruction or a data word is skipped."""
    text = subprocess.run([str(objdump), "-d", str(elf)], capture_output=True, text=True, check=True).stdout
    rows: list[dict[str, Any]] = []
    function = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith(">:") and "<" in stripped:
            function = stripped[stripped.index("<") + 1 : -2]
            continue
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0].strip().endswith(":"):
            continue
        word_text = fields[1].strip()
        if len(word_text) != 8:
            continue
        try:
            word, address = int(word_text, 16), int(fields[0].strip()[:-1], 16)
        except ValueError:
            continue
        mnemonic = fields[2].strip() if len(fields) > 2 else ""
        rows.append(
            {"address": address, "word": word, "word_text": word_text, "mnemonic": mnemonic, "function": function}
        )
    return rows


def custom_instructions(elf: Path, *, objdump: Path, opcode: int) -> list[dict[str, Any]]:
    """Every 32-bit instruction word in ``elf`` whose major opcode is ``opcode``: its address, word,
    selector (the top seven bits) and the function that holds it."""
    return [
        {
            "address": f"{row['address']:x}",
            "word": row["word_text"],
            "selector": row["word"] >> SELECTOR_SHIFT,
            "function": row["function"],
        }
        for row in listing(elf, objdump=objdump)
        if row["word"] & MAJOR_OPCODE_MASK == opcode
    ]


def defined_symbols(obj: Path, *, nm: Path) -> set[str]:
    listing = subprocess.run([str(nm), "--defined-only", str(obj)], capture_output=True, text=True, check=True).stdout
    return {fields[-1] for fields in (line.split() for line in listing.splitlines()) if len(fields) >= 3}


#: Schema for the diagnostic-only per-group instruction census (never a prohibition verdict).
CENSUS_SCHEMA = "merlin_isa_instruction_census_v1"

#: A hit whose selector names no declared instruction at all: the derived facts have a gap, and that
#: gap must be visible in the census rather than silently dropped from every group's count.
UNNAMED_SELECTOR = "unnamed_selector"


def _declared_by_selector(target: str) -> dict[int, dict[str, Any]]:
    """``{selector: {"name":..., "roles": [...]}}`` for every instruction the target's facts derive --
    not filtered to any subset of roles, unlike :func:`prohibited_instructions`."""
    from .task_instruction_evidence import declared_instruction_set, target_instruction_facts

    declared = declared_instruction_set(target_instruction_facts(target))
    if declared.get("status") not in (None, "derived"):
        raise ValueError(f"the target's instruction set is not derived: {declared.get('reason')}")
    return {
        int(str(entry["funct"]), 0): {"name": str(entry["name"]), "roles": sorted(entry.get("roles") or ())}
        for entry in declared.get("instructions") or ()
    }


def _owner_of(group_objects: Mapping[str, Path], *, nm: Path) -> dict[str, str]:
    """``{symbol: group}`` from the kernel object each group linked; a symbol two groups both define
    keeps whichever group named it first (groups never share a kernel symbol in a real build)."""
    owner: dict[str, str] = {}
    for group, obj in group_objects.items():
        if Path(obj).is_file():
            for symbol in defined_symbols(Path(obj), nm=nm):
                owner.setdefault(symbol, str(group))
    return owner


def check_program(
    elf: Path,
    *,
    target: str,
    roles: Sequence[str],
    compiler: str | Path,
    group_objects: Mapping[str, Path],
    library_groups: Sequence[str] = (),
) -> dict[str, Any]:
    """The prohibited instructions ``elf`` emits, each attributed to a group or to the program's code.

    ``group_objects`` maps a group to the kernel object linked for it (its symbols name that group's
    code); ``library_groups`` are the groups the program answers with the target's library, whose code
    is part of the program's own. Also carries a per-group instruction CENSUS under ``"census"`` --
    every custom instruction the program emits, by its derived role, whether prohibited or not. That
    field is diagnostic only: a role's count names no fix, it is read from the same disassembly pass
    this check already pays for.

    ``status`` is ``measured``, or ``unmeasured`` with ``clean: None`` when ``roles`` are declared but
    the target's facts give none of them to any instruction -- the same rule as :func:`scan_elf`. A
    prohibited set that is empty would make every program clean without reading it, which is how a
    no-FSM rule once passed programs full of loop-descriptor instructions.

    The scan reads the WHOLE linked program as the disassembler lists it: every executable section,
    every function, called or not. An instruction reached only through a function pointer, or never
    executed at all, is in the binary and is refused exactly like one on the hot path."""
    by_selector = _declared_by_selector(target)
    prohibited = {
        selector: entry["name"] for selector, entry in by_selector.items() if roles and set(roles) & set(entry["roles"])
    }
    if roles and not prohibited:
        return {
            "schema": SCHEMA,
            "status": "unmeasured",
            "roles": list(roles),
            "prohibited": {},
            "clean": None,
            "detail": f"the target's instruction facts give no instruction any of the roles {list(roles)}",
            "hits": [],
            "summary": {},
            "library_groups": list(library_groups),
        }
    objdump = disassembler_for(compiler)
    nm = objdump.with_name(objdump.name.replace("objdump", "nm"))
    owner = _owner_of(group_objects, nm=nm)
    opcode = custom_opcode(target)
    # (function, selector, count) in first-seen order, and the prohibited rows in address order: from one
    # pass over the listing, or -- for a large program -- from address ranges scanned in parallel and
    # folded back in order (merlin.perf.isa_scan), which yields the same sequence.
    pairs: list[tuple[Any, int, int]] = []
    rows: list[dict[str, Any]] = []
    if Path(elf).is_file() and Path(elf).stat().st_size >= PARALLEL_SCAN_MIN_BYTES:
        from . import isa_scan

        for part in isa_scan.scan(
            Path(elf), objdump=objdump, nm=nm, opcode=opcode, keep=list(prohibited), parts=SCAN_PARTS
        ):
            pairs.extend((f, int(sel), int(n)) for f, sel, n in part["pairs"])
            rows.extend(part["rows"])
    else:
        seen: dict[tuple[Any, int], int] = {}
        for row in custom_instructions(Path(elf), objdump=objdump, opcode=opcode):
            seen[(row["function"], row["selector"])] = seen.get((row["function"], row["selector"]), 0) + 1
            if row["selector"] in prohibited:
                rows.append(row)
        pairs = [(f, sel, n) for (f, sel), n in seen.items()]
    counts: dict[str, int] = {}
    census: dict[str, dict[str, Any]] = {}
    for function, selector, n in pairs:
        where = owner.get(str(function), PROGRAM_CODE)
        known = by_selector.get(selector)
        bucket = census.setdefault(where, {"total": 0, "by_kind": {}, "by_symbol": {}})
        bucket["total"] += n
        for kind in known["roles"] if known and known["roles"] else [f"{UNNAMED_SELECTOR}_{selector}"]:
            bucket["by_kind"][kind] = bucket["by_kind"].get(kind, 0) + n
        symbol = str(function or "")
        if symbol:
            bucket["by_symbol"][symbol] = bucket["by_symbol"].get(symbol, 0) + n
        name = prohibited.get(selector)
        if name is not None:
            key = f"{name} in {'g' + where if where != PROGRAM_CODE else 'the program code'}"
            counts[key] = counts.get(key, 0) + n
    hits = [
        {**row, "instruction": prohibited[row["selector"]], "group": owner.get(str(row["function"]), PROGRAM_CODE)}
        for row in rows
    ]
    return {
        "schema": SCHEMA,
        "status": "measured",
        "roles": list(roles),
        "prohibited": {str(k): v for k, v in sorted(prohibited.items())},
        "clean": not counts,
        "hits": hits[:200],
        "summary": counts,
        "library_groups": list(library_groups),
        "program_code_note": "instructions outside every package kernel object are the program's own code: "
        "the target library called for the groups routed to it, and host code",
        "census": {
            "schema": CENSUS_SCHEMA,
            "per_group": census,
            "library_groups": list(library_groups),
        },
    }


def scan_elf(elf: str | Path, *, target: str, roles: Sequence[str], max_hits: int = 32) -> dict[str, Any]:
    """The prohibited instructions a linked ELF carries, read with no external tool.

    The whole-model check above disassembles with the toolchain beside the compiler so each hit can be
    attributed to a group's object. A capsule's program has one kernel and one harness, so the question
    there is only WHETHER the executable carries a prohibited instruction and which; this answers it by
    walking the executable sections (:mod:`merlin.targetgen.elf_lanes`, the same walk the lane scan
    uses). The opcode, the selector field and the prohibited set are the target's derived facts.

    ``status`` is ``measured`` or ``unmeasured`` -- an ELF that cannot be walked, or a target whose
    opcode or instruction roles are not derivable, is never reported clean.
    """
    from merlin.targetgen import elf_lanes as EL

    path = Path(elf)
    base = {"schema": SCHEMA, "roles": list(roles), "elf": str(path)}
    try:
        prohibited = prohibited_instructions(target, roles)
        opcode, source = EL.accelerator_opcode(target)
        if opcode is None:
            raise ValueError(source)
    except Exception as exc:  # noqa: BLE001 -- an underivable fact is unmeasured, never clean
        return {**base, "status": "unmeasured", "clean": None, "detail": f"{type(exc).__name__}: {exc}"}
    if not prohibited:
        return {
            **base,
            "status": "unmeasured",
            "clean": None,
            "detail": f"the target's instruction facts give no instruction any of the roles {list(roles)}",
        }
    try:
        blob = path.read_bytes()
        sections = EL.executable_sections(blob)
    except (OSError, EL.ElfUnreadable, IndexError, ValueError) as exc:
        return {**base, "status": "unmeasured", "clean": None, "detail": f"{type(exc).__name__}: {exc}"}
    if not sections:
        return {**base, "status": "unmeasured", "clean": None, "detail": "no executable sections to walk"}
    counts: dict[str, int] = {}
    hits: list[dict[str, Any]] = []
    for name, offset, size, address in sections:
        for at, word in EL.instruction_words(blob[offset : offset + size], address):
            if word & MAJOR_OPCODE_MASK != opcode:
                continue
            instruction = prohibited.get(word >> SELECTOR_SHIFT)
            if instruction is None:
                continue
            counts[instruction] = counts.get(instruction, 0) + 1
            if len(hits) < max_hits:
                hits.append({"section": name, "address": hex(at), "instruction": instruction})
    return {
        **base,
        "status": "measured",
        "clean": not counts,
        "prohibited": {str(k): v for k, v in sorted(prohibited.items())},
        "summary": dict(sorted(counts.items())),
        "hits": hits,
    }


def refusal_line(report: Mapping[str, Any]) -> str:
    """``isa_prohibited: <instr> in g<N>[, ...]`` -- the one line a refused candidate leads with."""
    return "isa_prohibited: " + ", ".join(sorted(report.get("summary") or {})[:12])


def check_build(build: Mapping[str, Any], *, target: str, roles: Sequence[str]) -> dict[str, Any]:
    """:func:`check_program` on a whole-model build, from the build's own record: its ELF, the compiler
    that built it, the kernel object linked for each group, and the groups routed to the library."""
    import json

    notes = build.get("notes") or build.get("builder_notes") or {}
    record = json.loads(Path(str(notes["build_record"])).read_text(encoding="utf-8"))
    program = record.get("program") or {}
    objects = {}
    for entry in record.get("linked_objects") or ():
        name = Path(str(entry.get("path") or "")).name
        stem = name.split(".", 1)[0]
        if stem.startswith("g") and stem[1:].isdigit():
            objects[stem[1:]] = Path(str(entry["path"]))
    library = [
        str(row.get("group"))
        for row in (record.get("attribution") or {}).get("per_group") or ()
        if isinstance(row, Mapping) and row.get("on") != "package"
    ]
    return check_program(
        Path(str(build["elf"])),
        target=target,
        roles=roles,
        compiler=program["compiler"],
        group_objects=objects,
        library_groups=library,
    )


__all__ = [
    "SCHEMA",
    "check_build",
    "check_program",
    "custom_instructions",
    "prohibited_instructions",
    "refusal_line",
    "scan_elf",
]
