"""Can a performance family's emitter actually target the machines its gate admits?

A family is admitted by a DOUBLE gate: its `performance.gate.traits` must all be satisfied on the
target, and its `performance.emitter.status` must be `existing`. Both are checked, independently, and
neither asks the question between them -- whether the emitter can emit for THAT target.

⚠️ MEASURED, AND IT IS NOT HYPOTHETICAL. `PC`'s traits are satisfied on exactly one target, gemmini
(`independent_engine_ports` and `explicit_completion` both derived True from its own facts), and
gemmini is the one target `workload_gen` cannot encode a kernel for -- it ships no ISA definition,
being a RoCC command-buffer machine rather than a self-hosted-ISA one. The only target the emitter CAN
encode, atlas, has `explicit_completion` unestablished. So the family is unsatisfiable as declared, and
before this it read as merely "the emitter is not wired yet": a reader would keep trying to build the
thing that was already built.

The distinction the two emitter kinds turn on:

``merlin.targetgen.corpus_spec.build``
    A capsule BUILDER. It emits the interface grammar and needs no ISA, so it reaches every target with
    a corpus binding -- which is why PK, PF and PL run on gemmini and atlas.

``merlin.perf.workload_gen.plan_matmul``
    A machine-level PROGRAM generator. It encodes real instructions against the target's derived field
    maps, so it needs a self-hosted ISA definition. Measured: of this repo's targets only atlas has one.

`reachability` returns the whole matrix rather than a verdict, because the interesting states are
per-(family, target) and collapsing them loses which half is missing.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Emitter entry-point prefixes that need a self-hosted ISA definition to encode anything. Matched on
#: the declared MODULE rather than on a family name, so a new family reusing the program generator is
#: classified without being listed.
_NEEDS_ISA = ("merlin.perf.workload_gen",)


@dataclass(frozen=True)
class Reach:
    """Whether ``entry`` can emit for ``target``, and why not when it cannot."""

    entry: str
    target: str
    can_emit: bool
    needs_isa: bool
    reason: str = ""


def needs_isa_definition(entry: str) -> bool:
    """True when this emitter encodes real instructions and so needs the target's own ISA."""
    return str(entry).startswith(_NEEDS_ISA)


def can_emit(entry: str, target: str) -> Reach:
    """Can the emitter named by ``entry`` produce a program for ``target``?

    A capsule builder always can -- it emits the interface grammar, which every target's corpus
    binding accepts. A program generator can only where the ISA resolves, and the reason it does not
    is carried through rather than reduced to False, because "no ISA definition" and "no compute array"
    are different repairs.
    """
    if not needs_isa_definition(entry):
        return Reach(entry=entry, target=target, can_emit=True, needs_isa=False,
                     reason="capsule builder: emits the interface grammar, no ISA needed")
    try:
        from merlin.perf.workload_gen import machine_facts
        machine_facts(target)
    except Exception as exc:                       # noqa: BLE001 -- the reason is the payload
        return Reach(entry=entry, target=target, can_emit=False, needs_isa=True,
                     reason=f"{type(exc).__name__}: {str(exc)[:160]}")
    return Reach(entry=entry, target=target, can_emit=True, needs_isa=True,
                 reason="the target's ISA definition resolves, so instructions can be encoded")


def family_reach(families: dict, targets) -> dict:
    """``{family: {target: Reach}}`` for ``{family: emitter entry}`` over ``targets``."""
    return {fam: {t: can_emit(entry, t) for t in targets} for fam, entry in families.items()}


def unreachable_where_admitted(family_gates: dict, families: dict) -> dict[str, tuple[str, ...]]:
    """Families whose gate is satisfied ONLY on targets their emitter cannot reach.

    ``family_gates`` maps a family to the targets whose traits satisfy it. A family with no satisfying
    target at all is NOT reported here -- that is the trait gate's own business and it already says so
    with evidence. What this names is the contradiction the two gates cannot see between them: the
    claim is admissible somewhere, and nowhere it is admissible can it be emitted.
    """
    out: dict[str, tuple[str, ...]] = {}
    for fam, satisfying in family_gates.items():
        satisfying = tuple(satisfying)
        if not satisfying:
            continue                               # no admitting target: the trait gate's story
        entry = families.get(fam)
        if entry is None:
            continue
        blocked = tuple(t for t in satisfying if not can_emit(entry, t).can_emit)
        if len(blocked) == len(satisfying):
            out[fam] = blocked
    return out
