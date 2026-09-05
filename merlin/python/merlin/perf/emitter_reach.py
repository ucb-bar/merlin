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

⚠️ THAT MEASUREMENT HAS SINCE EXPIRED, and the paragraph above is kept because it is why PC is declared
the way it is, not because it still holds. Re-measured 2026-09-05: ``machine_facts`` now RESOLVES for
both the systolic RoCC target and the self-hosted-ISA one, and fails only for the SIMT target. So the
premise that made PC unsatisfiable -- the one target admitting it being the one target with no ISA --
is no longer true, and PC's repointing may now be over-constrained rather than required. Whether to
move it back is a declaration decision and is deliberately NOT made here; what this note prevents is
the next reader taking a stale measurement as a live one. The test asserting the old fact fails on this
tree for that reason and not because of the classification change below.

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

#: Emitter entry-point prefixes that work off the frozen INTERFACE GRAMMAR -- either producing it
#: (`corpus_spec.build` emits the interface) or consuming it (`command_stream_gen` permutes the ordered
#: command list a capsule already parses to). Neither encodes a machine instruction, so both reach
#: every target that has a corpus binding.
#:
#: The classification is by what the emitter CONSUMES, which is the property that decides reach; a
#: per-entry list would drift as families are added, and matching on the module keeps a new family
#: reusing either archetype classified without an edit here.
#:
#: ⚠️ THIS IS AN ALLOWLIST AND MUST STAY ONE. It used to be the `else` branch of the ISA test, which
#: made "reaches every target, no ISA needed" the answer for every entry that merely was not the
#: program generator. Measured on the entries this repo actually declares, that returned True for the
#: barrier pair, both `new:` placeholders that name no module at all, and a deliberately nonsensical
#: string -- each with a REASON asserting a property nobody had checked. An emitter kind nobody has
#: classified is now `established=False`, not a yes.
_GRAMMAR_EMITTERS = ("merlin.targetgen.corpus_spec.build", "merlin.perf.command_stream_gen")

#: Emitter entry points that drive the TARGET'S OWN driver emitter through a knob on its signature
#: (see `merlin.perf.barrier_arms.pair_from_emitter`, whose `knob=` names a parameter belonging to the
#: emitter, not to that module). Whether a given target's emitter accepts the knob is a fact about
#: that target's backend, which does not live on the import path -- `merlin/targets` is not a package
#: -- so this module cannot resolve it and does not pretend to.
_DRIVES_TARGET_EMITTER = ("merlin.perf.barrier_arms.pair_from_emitter",)


@dataclass(frozen=True)
class Reach:
    """Whether ``entry`` can emit for ``target``, and why not when it cannot."""

    entry: str
    target: str
    can_emit: bool
    needs_isa: bool
    reason: str = ""
    #: Whether this answer was DETERMINED. ``False`` means the kind of emitter was not recognised or
    #: the fact lives somewhere this module cannot read -- ``can_emit`` is then ``False`` because an
    #: unestablished reach is not a reach, NOT because anything was shown to be impossible. Collapsing
    #: the two is how "nobody classified this" comes to read as "measured, and it works everywhere".
    established: bool = True


def needs_isa_definition(entry: str) -> bool:
    """True when this emitter encodes real instructions and so needs the target's own ISA."""
    return str(entry).startswith(_NEEDS_ISA)


def drives_target_emitter(entry: str) -> bool:
    """True when this emitter calls the TARGET'S own driver emitter through a knob on its signature."""
    return str(entry).startswith(_DRIVES_TARGET_EMITTER)


def is_grammar_emitter(entry: str) -> bool:
    """True only for entries KNOWN to produce or consume the interface grammar, never by default."""
    return str(entry).startswith(_GRAMMAR_EMITTERS)


#: Kept as the older spelling of :func:`is_grammar_emitter` for callers that used it.
is_capsule_builder = is_grammar_emitter


def can_emit(entry: str, target: str) -> Reach:
    """Can the emitter named by ``entry`` produce a program for ``target``?

    A capsule builder always can -- it emits the interface grammar, which every target's corpus
    binding accepts. A program generator can only where the ISA resolves, and the reason it does not
    is carried through rather than reduced to False, because "no ISA definition" and "no compute array"
    are different repairs. Every OTHER kind is unestablished: ``can_emit`` is False and
    ``established`` is False, so a reader can tell "shown not to reach" from "never classified".
    """
    if is_grammar_emitter(entry):
        return Reach(entry=entry, target=target, can_emit=True, needs_isa=False,
                     reason="works off the interface grammar, not machine instructions: no ISA needed")
    if drives_target_emitter(entry):
        return Reach(
            entry=entry, target=target, can_emit=False, needs_isa=False, established=False,
            reason=("drives the target's own driver emitter through a knob on its signature; whether "
                    "this target's emitter accepts that knob is a fact about its backend, which is not "
                    "on the import path, so reach is NOT established here"))
    if not needs_isa_definition(entry):
        return Reach(
            entry=entry, target=target, can_emit=False, needs_isa=False, established=False,
            reason=(f"unclassified emitter kind {entry!r}: not a known capsule builder, program "
                    "generator, or target-emitter driver, so nothing here has checked that it can "
                    "emit for this target"))
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
        reach = {t: can_emit(entry, t) for t in satisfying}
        # ESTABLISHED non-reach only. An unclassified emitter kind is not evidence that the family
        # cannot be emitted -- reporting it here would turn "nobody checked" into a named defect, the
        # mirror of the permissive default this module used to carry. `unestablished_reach` names
        # those instead, so neither state is silent and neither borrows the other's meaning.
        if any(not r.established for r in reach.values()):
            continue
        blocked = tuple(t for t, r in reach.items() if not r.can_emit)
        if len(blocked) == len(satisfying):
            out[fam] = blocked
    return out


def unestablished_reach(families: dict, targets) -> dict[str, str]:
    """``{family: reason}`` for families whose emitter kind this module cannot classify.

    Separate from :func:`unreachable_where_admitted` on purpose. A family here is not broken and is
    not known to work; it is UNMEASURED, and the repair is to classify the emitter kind (or to teach
    this module how to resolve it) rather than to go looking for a missing backend feature.
    """
    out: dict[str, str] = {}
    for fam, entry in families.items():
        for t in targets:
            r = can_emit(entry, t)
            if not r.established:
                out[fam] = r.reason
                break
    return out
