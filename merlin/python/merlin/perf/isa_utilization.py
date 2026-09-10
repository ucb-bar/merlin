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

__all__ = ["ASM_MARKER", "emitted_functs", "capability_utilization"]

#: The attribute an emitted LLVM-dialect inline-assembly operation carries its template in.
ASM_MARKER = 'asm_string = "'

#: Position of the funct field in a `.insn r` template's comma-separated operand list
#: (`.insn r <opcode>, <funct3>, <funct7>, ...`). Named so the parse is a stated convention
#: rather than an anonymous index.
_FUNCT_FIELD = 2
_OPCODE_FIELD = 0


def _templates(artifact_text: str) -> list[str]:
    """Every inline-assembly template in the artifact, in order.

    Parsed structurally (split on the attribute marker and the closing quote) rather than by
    pattern: a template that spells its operands differently must still be seen, and a missed
    one silently under-reports what the program uses.
    """
    out: list[str] = []
    for chunk in artifact_text.split(ASM_MARKER)[1:]:
        end = chunk.find('"')
        if end >= 0:
            out.append(chunk[:end])
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
        if not head or head[0] != ".insn":
            continue
        try:
            opcode = int(head[-1], 0)
            funct = int(fields[_FUNCT_FIELD], 0)
        except ValueError:
            continue
        if opcode == custom_opcode:
            counts[funct] += 1
    return counts


def capability_utilization(artifact_text: str, *, declared_functs: Mapping[int, str],
                           custom_opcode: int) -> dict[str, Any]:
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
        "licence": ("an unused declared instruction is an OPPORTUNITY, not a defect -- the "
                    "program may be correct without it. It says the compiler never reaches for "
                    "a capability the hardware offers, which is where a structural lever hides."),
    }
