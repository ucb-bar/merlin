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
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

# ---- isa_utilization ---------------------------------------------------------------------------

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
    from merlin.analysis.observation.rocc_commands import funct_table_for

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


# ---- capability_refusal ------------------------------------------------------------------------
# Why a capability the target offers was REFUSED, per site, aggregated by clause.
#
# A capability selector is a sequence of guard clauses over target facts and operation semantics,
# and it returns both a verdict and the clause that decided it. Compilers routinely keep the verdict
# and discard the clause -- `eligible, _reason = select(...)` -- which throws away the only
# machine-readable statement of what would have to change. Measured on a whole-model ResNet-50: all
# 53 convolutions were refused the device-side convolution sequencer, the selector named the deciding
# clause every time, and every one of those 53 strings was dropped at the call site. Downstream, the
# emitted program simply contains no such instruction, and the absence looks like a choice nobody
# made rather than a refusal with a stated cause.
#
# `isa_utilization` can say "this declared instruction is never emitted". Only the refusal ledger can
# say WHY, and whether the cause is a contract this compiler chose (an output dtype, a tensor layout)
# or a property of the operation it cannot change.
#
# THE SHORT-CIRCUIT CAVEAT, which is the whole reason this is subtle. A guard sequence returns on its
# FIRST failing clause, so a census counts first refusals and says nothing about the clauses never
# reached. Removing the top clause does not admit those sites; it reveals the next clause. Measured
# here: the sites refused for a full-width output dtype had their layout clause never evaluated, so
# the dtype fix -- which was made, and did satisfy that clause -- exposed the layout requirement
# underneath rather than admitting anything. A reader who treats the top clause as "the" blocker will
# predict admission and be wrong. :func:`census` therefore reports `first_refusal_only: True` and
# carries that caveat in the payload, and :func:`unblocking_sequence` reports the clauses as an
# ORDERED CASCADE rather than a set of independent fixes.
#
# Target-neutral by construction: clause names are opaque strings chosen by the selector, site labels
# are opaque strings chosen by the caller. This module never interprets either, and never names a
# target, capability, instruction or clause of its own.

#: The conventional clause name a selector returns when it ADMITS a site. Callers may use any
#: string; passing it here keeps an admitted site out of the refusal census.
SELECTED = "selected"


class RefusalSite(NamedTuple):
    """One site's verdict for one capability.

    `site` and `clause` are opaque. `detail` carries whatever the caller wants a reader to see
    (geometry, dtypes, the tensor layouts) so a clause count can be turned back into the concrete
    thing that has to change.
    """

    site: str
    admitted: bool
    clause: str
    detail: Mapping[str, Any] | None = None
    #: EVERY clause this site fails, when the selector was asked for all of them instead of stopping
    #: at the first. ``None`` means the caller reported a first refusal only, which is the short-circuit
    #: shape the module docstring warns about. Supplying it is what lets :func:`census` report the
    #: caveat as a MEASURED property of its input rather than a blanket assumption — measured on the
    #: recorded ResNet-50 emission, all 53 convolutions reported one blocker and were failing three.
    clauses: tuple[str, ...] | None = None


def _rows(sites: Iterable[RefusalSite]) -> list[RefusalSite]:
    out: list[RefusalSite] = []
    for row in sites:
        if isinstance(row, RefusalSite):
            out.append(row)
            continue
        # Accept the selector's own (bool, str) shape paired with a label, so a caller does not
        # have to restate what it already has.
        site, admitted, clause = row[0], bool(row[1]), str(row[2])
        detail = row[3] if len(row) > 3 else None
        clauses = tuple(str(c) for c in row[4]) if len(row) > 4 and row[4] is not None else None
        out.append(RefusalSite(str(site), admitted, clause, detail, clauses))
    return out


def _blocking(row: RefusalSite) -> tuple[str, ...]:
    """Every clause this site is known to fail — its whole stack when it supplied one, else the single
    clause that decided it. A site that supplied a stack but omitted its own deciding clause still
    counts that clause, so the two shapes cannot disagree about what blocked it."""
    if row.clauses is None:
        return (row.clause,)
    return row.clauses if row.clause in row.clauses else (row.clause, *row.clauses)


def census(capability: str, sites: Iterable[RefusalSite]) -> dict[str, Any]:
    """Aggregate per-site verdicts for one capability into a clause census.

    Whether the census is of FIRST refusals is a property of what the caller supplied, not an
    assumption: a site that reports its whole failing stack (``RefusalSite.clauses``) is counted under
    every clause blocking it, and ``first_refusal_only`` is true only while some refused site did not.
    ``clause_depth`` carries the shape -- all-ones for a first-refusal census, and for a complete one
    the number that says how far clearing the top clause would actually get.
    """
    rows = _rows(sites)
    admitted = [r for r in rows if r.admitted]
    refused = [r for r in rows if not r.admitted]
    # MEASURED, not assumed: the caveat holds only while some refused site reported a first refusal
    # only. A census every one of whose sites supplied its whole stack is complete, and saying
    # otherwise would understate what is known — which is the opposite error, but still an error.
    partial = [r for r in refused if r.clauses is None]
    counts: Counter[str] = Counter()
    for r in refused:
        counts.update(set(_blocking(r)))
    # Rank by how many sites a clause decided, then by name so equal counts are stable.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "schema": "capability_refusal_census_v1",
        "capability": capability,
        "sites_total": len(rows),
        "sites_admitted": len(admitted),
        "sites_refused": len(refused),
        "admitted_sites": [r.site for r in admitted],
        "clauses": [
            {
                "clause": clause,
                "sites": count,
                "share_of_refused": round(count / len(refused), 6) if refused else 0.0,
                "example_sites": [r.site for r in refused if clause in _blocking(r)][:5],
                "example_detail": next((dict(r.detail) for r in refused if clause in _blocking(r) and r.detail), None),
            }
            for clause, count in ordered
        ],
        "first_refusal_only": bool(partial),
        # How deep each refused site's KNOWN stack is. A census of first refusals is all-ones by
        # construction; anything else is the shape a complete stack has, and it is the number that
        # tells a reader how far "fix the top clause" would actually get.
        "clause_depth": dict(sorted(Counter(len(set(_blocking(r))) for r in refused).items())),
        "sites_with_complete_stack": len(refused) - len(partial),
        "caveat": (
            (
                "a guard sequence returns on its FIRST failing clause, so these counts are of "
                "first refusals and say nothing about clauses never reached. Removing the top "
                "clause reveals the next one; it does not admit these sites."
            )
            if partial
            else (
                "every refused site reported EVERY clause it fails, so these counts are complete: "
                "a site appears under each clause blocking it, the shares therefore sum above 1, "
                "and a clause cleared here does not reveal a hidden one underneath."
            )
        ),
    }


def unblocking_sequence(censuses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ordered cascade a caller would have to clear, across one or more capabilities.

    Deliberately NOT a set of independent fixes: the clauses are returned in refusal-count order
    with an explicit statement that each is only known to be the blocker while the ones above it
    stand.
    """
    steps: list[dict[str, Any]] = []
    for entry in censuses:
        clauses = entry.get("clauses") or ()
        # A census whose sites each reported their WHOLE stack is not a cascade: nothing here is
        # conditional on a clause above it, and every listed clause has to be cleared before any of
        # those sites is admitted. Saying "walk this in order" about such a census would invite the
        # same wrong prediction from the other direction.
        cascade = bool(entry.get("first_refusal_only", True))
        for rank, row in enumerate(clauses):
            steps.append(
                {
                    "capability": entry.get("capability"),
                    "clause": row.get("clause"),
                    "sites": row.get("sites"),
                    "rank_within_capability": rank,
                    "known_blocker_only_while": (
                        ([c.get("clause") for c in clauses[:rank]] or None) if cascade else None
                    ),
                    "all_must_clear": not cascade,
                }
            )
    steps.sort(key=lambda s: (-(s["sites"] or 0), str(s["capability"]), str(s["clause"])))
    return {
        "schema": "capability_unblocking_sequence_v1",
        "steps": steps,
        "reading": (
            "a step with `known_blocker_only_while` is the deciding clause for that many sites GIVEN "
            "those clauses still stand: clearing it re-runs the selector and may surface a clause that "
            "was never evaluated, so that part is a cascade to walk, not a list to divide up. A step "
            "marked `all_must_clear` came from a census whose sites each reported every clause they "
            "fail, so nothing is hidden underneath it and every such clause must be cleared before any "
            "of its sites is admitted."
        ),
    }


__all__ = [
    "ASM_MARKER",
    "RefusalSite",
    "SELECTED",
    "capability_utilization",
    "capability_utilization_for_target",
    "census",
    "emitted_functs",
    "field_utilization",
    "unblocking_sequence",
]
