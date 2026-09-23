"""Which of a target's DECLARED instructions an emitted program actually uses.

A compiler can be complete, correct, and quietly ignore most of the machine. Measured on a
whole-model ResNet-50 emission: the program used 8 of the 25 functs the target's own RTL facts
declare, and the 17 it never emitted included the entire device-side convolution sequencer
(`LOOP_CONV_WS` and its six config functs). Because the sequencer was never emitted, every
convolution's im2col patch generation ran as host scalar code -- 37% of all host dynamic
operations -- and the accelerator idled behind the CPU on every packed row.

Nothing in the compiler's own output says that. The command buffer is well formed, the gate
passes, the counters look busy. The absence is only visible by comparing what the program emits
against what the target DECLARES it can do, which is exactly what this module does.

Target-neutral by construction: the funct table is a PARAMETER, derived by the caller from the
target's own facts. This module never names a target, an opcode or an instruction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

__all__ = [
    "ASM_MARKER",
    "emitted_functs",
    "capability_utilization",
    "capability_utilization_for_target",
    "field_utilization",
]

#: The attribute an emitted LLVM-dialect inline-assembly operation carries its template in. Kept
#: because it names the attribute, not because the scan depends on it: see :func:`_templates`, which
#: reads BOTH this spelling and the pretty-printed operation form of the same op.
ASM_MARKER = 'asm_string = "'

#: The assembler directive that introduces an explicitly-encoded instruction. The one token this
#: module recognises, and a stated convention rather than an anonymous prefix test.
_ASM_DIRECTIVE = ".insn"

#: Position of the funct field in a `.insn r` template's comma-separated operand list
#: (`.insn r <opcode>, <funct3>, <funct7>, ...`). Named so the parse is a stated convention
#: rather than an anonymous index.
_FUNCT_FIELD = 2
_OPCODE_FIELD = 0


def _templates(artifact_text: str) -> list[str]:
    """Every inline-assembly template in the artifact, in order.

    Parsed structurally -- on each line the quoted strings are the odd fields of a split on the
    quote character, and a template is one whose first token is the assembler's instruction
    directive. Both surface spellings of an LLVM-dialect inline assembly carry it that way: the
    ATTRIBUTE form (``{asm_string = ".insn ..."}``, which a generic printer emits) and the
    OPERATION form (``llvm.inline_asm has_side_effects ".insn ..."``, which the pretty printer
    emits for the same op).

    Recognising only the attribute marker is how this measurement read 0 of 26 declared
    instructions on a program that had just emitted one: the assembler emitted the pretty form, no
    template was found, and the result was a clean-looking number produced by not looking. Same
    defect class as a too-narrow line pattern, which is why the scan is over the quoting structure
    and not over a spelling. A quoted operand-constraint string (``"r,r"``) is skipped because its
    first token is not the directive, so no vocabulary of constraints is assumed either.
    """
    out: list[str] = []
    for line in artifact_text.splitlines():
        for quoted in line.split('"')[1::2]:
            candidate = quoted.strip()
            if candidate.split()[:1] == [_ASM_DIRECTIVE]:
                out.append(candidate)
    return out


def emitted_functs(artifact_text: str, *, custom_opcode: int) -> Counter:
    """`{funct: occurrences}` for accelerator instructions in ``artifact_text``.

    Only templates whose opcode field equals ``custom_opcode`` are counted, so an unrelated
    inline assembly (a fence, a CSR read) is not mistaken for an accelerator instruction.
    """
    counts: Counter = Counter()
    for template in _templates(artifact_text):
        fields = [field.strip() for field in template.split(",")]
        if len(fields) <= _FUNCT_FIELD:
            continue
        head = fields[_OPCODE_FIELD].split()
        if not head or head[0] != _ASM_DIRECTIVE:
            continue
        try:
            opcode = int(head[-1], 0)
            funct = int(fields[_FUNCT_FIELD], 0)
        except ValueError:
            continue
        if opcode == custom_opcode:
            counts[funct] += 1
    return counts


def capability_utilization(
    artifact_text: str, *, declared_functs: Mapping[int, str], custom_opcode: int
) -> dict[str, Any]:
    """What the program emits against what the target declares.

    ``declared_functs`` maps a funct code to the target's own name for it, and comes from the
    target's facts -- never from a table in this module. ``unused`` is the finding: an
    instruction the hardware offers and the compiler never reaches for.
    """
    used = emitted_functs(artifact_text, custom_opcode=custom_opcode)
    declared = {int(code): str(name) for code, name in declared_functs.items()}
    unused = sorted((code, name) for code, name in declared.items() if code not in used)
    undeclared = sorted(code for code in used if code not in declared)
    return {
        "schema": "isa_capability_utilization_v1",
        "declared_count": len(declared),
        "used_count": sum(1 for code in declared if code in used),
        "used": {declared[code]: used[code] for code in sorted(declared) if code in used},
        "unused": [{"funct": code, "name": name} for code, name in unused],
        # An emitted funct the facts do not declare is a provenance problem, not an
        # optimization one: the program is using an instruction nobody has vouched for.
        "undeclared_emitted": undeclared,
        "licence": (
            "an unused declared instruction is an OPPORTUNITY, not a defect -- the "
            "program may be correct without it. It says the compiler never reaches for "
            "a capability the hardware offers, which is where a structural lever hides."
        ),
    }


def _funct_code(value: Any) -> int:
    """One declared funct code as an int, accepting the spellings a fact table uses.

    A table may carry its codes as ints or as strings, and a string may be decimal or prefixed
    hexadecimal. Both are parsed to an int and COMPARED AS DATA -- no spelling is matched.
    """
    if isinstance(value, bool):
        raise ValueError("a boolean is not a funct code")
    if isinstance(value, int):
        return value
    return int(str(value).strip(), 0)


def capability_utilization_for_target(artifact_text: str, *, target: str) -> dict[str, Any]:
    """:func:`capability_utilization` with the declared table DERIVED from ``target``'s own facts.

    THE ONE PRODUCER. Both phases that measure declared-versus-emitted instruction use call this:
    the capsule-bench ISA tool broker and the whole-model emission analysis the optimization agent
    reads. A second copy would drift, and the half that drifted would be the half that reports a
    clean number.

    Both inputs the measurement needs -- the instruction table and the custom major opcode -- come
    from the target's own RTL facts (``funct_decode_table``), never from a literal here. When the
    facts are unavailable the table comes back empty, and an empty table would report every emitted
    instruction as undeclared and NOTHING as unused: a clean answer produced by not looking. So that
    case is recorded as ``UNKNOWN`` with the reason, and the caller can say the measurement did not
    run rather than that the compiler used everything the machine offers.
    """
    from merlin.kernels.decode.rocc import funct_table_for

    table = funct_table_for(target)
    names = table.get("names") if isinstance(table, Mapping) else None
    opcode = table.get("custom_opcode") if isinstance(table, Mapping) else None
    missing = [
        what
        for what, value in (("funct_decode_table.names", names), ("funct_decode_table.custom_opcode", opcode))
        if not value
    ]

    def unknown(reason: str) -> dict[str, Any]:
        return {
            "schema": "isa_capability_utilization_v1",
            "status": "UNKNOWN",
            "reason": reason,
            "declared_count": None,
            "used_count": None,
            "unused": [],
            "undeclared_emitted": [],
        }

    if missing:
        return unknown(
            f"{', '.join(missing)} could not be derived from this target's RTL facts, "
            f"so declared-versus-emitted instruction use was NOT measured. An empty "
            f"declared table would report full utilization, which is why this is a "
            f"stated UNKNOWN rather than a clean result."
        )
    try:
        declared = {_funct_code(code): str(name) for code, name in dict(names).items()}
        custom_opcode = _funct_code(opcode)
    except (AttributeError, TypeError, ValueError) as exc:
        return unknown(
            f"this target's derived funct_decode_table carries codes that are not integers "
            f"({type(exc).__name__}), so declared-versus-emitted instruction use was NOT measured."
        )
    return {
        **capability_utilization(artifact_text, declared_functs=declared, custom_opcode=custom_opcode),
        "status": "measured",
    }


def field_utilization(
    observed: Mapping[str, Any], *, declared_fields: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Declared instruction FIELDS a program never sets away from their identity value.

    An unused funct is a whole capability skipped; an unused FIELD is subtler and just as
    expensive. The instruction is emitted, so every funct-level check is satisfied, but a
    capability inside it sits at its default and the work it would have done happens somewhere
    else -- usually on the host, usually invisibly.

    Measured on one target's configuration instructions: nine declared fields were never set away
    from identity. Among them a DMA load scale (the hardware multiplies during the load, so an
    input rescale is free on the way in), a pixel-repeat count (one loaded row replayed to several
    destination rows), and two operand transposes (each input readable transposed at no cost, which
    is what makes an operand orientation a free choice rather than a relayout). Nothing in the
    emitted program reported their absence, because the instructions carrying them WERE emitted.

    `identity` gives the value at which a field does nothing, and it is REQUIRED per field: a
    default this module invented would decide what counts as "used", which is the whole finding.
    A declared field with no identity is reported as unknown rather than assumed unused.
    """
    declared = dict(declared_fields)
    unknown = sorted(name for name in declared if name not in identity)
    used, unused = [], []
    for name in sorted(declared):
        if name in unknown:
            continue
        seen = observed.get(name)
        # A field is USED when the program ever sets it away from the value that means "do
        # nothing". Presence alone is not use: an attribute emitted at its identity is exactly
        # the case this instrument exists to surface.
        values = seen if isinstance(seen, (list, tuple, set)) else ([seen] if seen is not None else [])
        if any(value != identity[name] for value in values):
            used.append(name)
        else:
            unused.append({"field": name, "identity": identity[name], "purpose": declared[name]})
    return {
        "schema": "isa_field_utilization_v1",
        "declared_count": len(declared),
        "used_count": len(used),
        "used": used,
        "unused": unused,
        "identity_not_declared": unknown,
        "licence": (
            "a field left at its identity is a capability the instruction carried and the "
            "compiler declined; the work it would have done is happening elsewhere. This "
            "is an OPPORTUNITY, not a defect -- but unlike an unused instruction, nothing "
            "else in the emitted program reveals it."
        ),
    }
