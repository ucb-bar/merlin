"""The ENGINE model — the one place that says which CCA facet a compute datapath is described by.

WHY THIS EXISTS. A target is not one kind of machine. Measured on the atlas corpus (180 hand-written
expert kernels, bucketed by the roles its DERIVED ``IsaModel`` assigns): 26 drive only the matrix
array, **50 drive only the vector engine**, 35 drive both, and 26 drive neither (scalar/DMA/control).
Filing atlas as "an NPU" and giving it one facet would leave 62% of its engine-driving kernels — the
whole ``smolvla_*`` softmax/silu/gelu/rope set, which is most of what a real model spends its
non-GEMM time on — undescribed. Radiance is the same shape: SIMT *and* an MX mesh.

So: **a target is a SET of engines**, and this module maps each engine to the facet that describes it.

WHAT THIS IS NOT. It is not a new taxonomy. ``targetgen.compute_units.KINDS`` already is one, already
models a target as a set of composing units, and already separates the datapath (``kind``) from how
software drives it (``endpoint_kind``). This module adopts that vocabulary rather than inventing a
parallel one — the whole point is to end up with FEWER vocabularies, not more. The mapping below is
the only new fact, and it is deliberately many-to-one: ``systolic`` and ``spatial`` are different
datapaths (stationary-weight wavefront vs rank-1 outer-product accumulate, which is why they are
separate kinds and carry separate RTL fact families) but they are described by the SAME facet, because
what a lifter reads out of the assembly — tile edge, dataflow, accumulator residency — is the same
question for both. The distinction survives where it belongs: as a VALUE of ``spatial.dataflow``.

HOW A TARGET'S SET IS OBTAINED. Derived, never declared twice. Today that derivation reads the
target's own contract (``compute_units``); the role census that the asm lifter needs will feed the
same function, so roles do double duty — one derivation, two consumers — instead of becoming a second
list that has to agree with this one. A target that declares nothing yields an EMPTY set, which is
honest: it means nobody has said what silicon this is, and callers must fail closed rather than
assume a default engine.

Consumed by ``kernels.cca_contract`` (which facet family an axis belongs to) and by
``runtime.backends.base`` (which derives its coarse ``TargetClass`` from this rather than maintaining
a third vocabulary in parallel).
"""
from __future__ import annotations

from typing import Iterable

#: Engine kind -> the CCA facet that describes it, or ``None`` for a kind that is not an accelerator
#: engine at all. Keys are exactly ``targetgen.compute_units.KINDS``; that is enforced by
#: :func:`check_covers_kinds` rather than trusted, so a kind added there cannot silently arrive here
#: with no facet and be treated as "describes nothing".
#:
#: ``scalar`` maps to ``None`` deliberately. A scalar core is not a datapath the CCA describes the
#: INSIDE of — it is the code AROUND the loop, which is ``EnvelopeFacet``'s subject, and that facet is
#: engine-agnostic (every target has one). Mapping it to ``envelope`` would wrongly imply a target
#: with a scalar core has an engine-scoped facet nothing else does.
ENGINE_FACET: dict[str, str | None] = {
    "systolic": "spatial",   # stationary-weight wavefront
    "spatial": "spatial",    # rank-1 outer-product accumulate into a tile — same facet, different dataflow
    "simt": "simt",          # threads of control: warps, divergence, barriers, managed scratchpad
    "vector": "vector",      # element-parallel WITHIN one thread of control: VL/LMUL/SEW, tail policy
    "scalar": None,          # not an engine — see above
}

#: The facets that are ENGINE-SCOPED (populated only for a target that has the engine). Every other
#: facet — compute, memory, envelope, coverage, and the dispatch/layout facets — is engine-agnostic
#: and may be populated for any target. Derived from the mapping so the two cannot disagree.
ENGINE_FACETS: frozenset[str] = frozenset(f for f in ENGINE_FACET.values() if f)


#: Facets that are meaningful for ANY target, whatever engines it has. Stated explicitly rather than
#: inferred as "everything not in ENGINE_FACETS", so that adding a facet to the CCA forces a decision:
#: :func:`check_facets_are_classified` fails on a facet in neither set. A facet nobody has classified
#: would otherwise default to agnostic and be populated for targets whose silicon cannot exhibit it.
AGNOSTIC_FACETS: frozenset[str] = frozenset({
    "compute",    # op, contraction form, accumulator, widening, reduction, epilogue
    "memory",     # access pattern, panel reuse, operand broadcast
    "envelope",   # what is emitted AROUND the loop -- including a scalar core's contribution
    "coverage",   # whole-model claim; model-scoped rather than per-kernel
    "dataflow",   # movement, awaiting its fold into memory/dispatch (see cca.DataflowFacet)
})


def check_facets_are_classified(facets: Iterable[str]) -> None:
    """Raise unless every CCA facet is declared engine-scoped or engine-agnostic, and not both."""
    known = set(facets)
    if both := sorted(ENGINE_FACETS & AGNOSTIC_FACETS):
        raise KeyError(f"facet(s) {both} are declared both engine-scoped and agnostic; a facet is "
                       f"populated either because a target HAS an engine or regardless of it")
    if unclassified := sorted(known - ENGINE_FACETS - AGNOSTIC_FACETS):
        raise KeyError(f"CCA facet(s) {unclassified} are neither engine-scoped nor agnostic; add each "
                       f"to ENGINE_FACET (via the engine it describes) or to AGNOSTIC_FACETS")
    if phantom := sorted((ENGINE_FACETS | AGNOSTIC_FACETS) - known):
        raise KeyError(f"facet(s) {phantom} are classified here but are not in the CCA schema")


def check_covers_kinds(kinds: Iterable[str]) -> None:
    """Raise unless every compute-unit kind has a facet decision recorded here.

    Fail-closed on BOTH sides. A kind with no entry would silently describe nothing; an entry for a
    kind that no longer exists is a stale mapping that will outlive whatever it referred to.
    """
    declared, known = set(ENGINE_FACET), set(kinds)
    if missing := sorted(known - declared):
        raise KeyError(f"compute-unit kind(s) {missing} have no engine->facet mapping; add them to "
                       f"ENGINE_FACET (map to None if the kind is not an accelerator engine)")
    if stale := sorted(declared - known):
        raise KeyError(f"ENGINE_FACET maps {stale}, which are not compute-unit kinds any more; "
                       f"remove them rather than leaving a mapping for something that cannot occur")


def facet_for(kind: str) -> str | None:
    """The CCA facet describing engine ``kind``, or None when the kind is not an engine.

    Raises for an unknown kind rather than returning None, because "no facet" and "I have never heard
    of this" are different answers and only one of them is safe to act on.
    """
    try:
        return ENGINE_FACET[kind]
    except KeyError:
        raise KeyError(f"unknown compute-unit kind {kind!r}; known: {sorted(ENGINE_FACET)}") from None


def engines_of_units(units) -> frozenset[str]:
    """The engine set of a list of ``targetgen.compute_units.ComputeUnit``.

    The SET, not a primary. Collapsing it to one kind is what makes a hybrid describe the wrong half of
    itself: a SIMT cluster containing a matrix PE reports only ``simt``, and an NPU with a vector
    engine beside its array reports only the array.

    Composition needs no special handling here, and deliberately gets none. ``contains`` references a
    unit BY NAME, so a contained unit must itself be declared to be referenced at all — it is already
    an element of ``units``. Walking ``contains`` as well would add nothing, and a walk that followed
    only one level (as an obvious implementation does) would quietly miss a grandchild while looking
    like it handled nesting. Taking every declared unit's kind is both simpler and transitively
    correct.
    """
    return frozenset(u.kind for u in units)


def engines_for(target: str) -> frozenset[str]:
    """The engine set a target declares, derived from its own contract. Empty when it declares none.

    An empty set is a real answer meaning "this target has not said what silicon it is", NOT a default.
    Callers that need an engine must fail closed on it; see ``facet_families_for``.
    """
    try:
        from merlin.targetgen import target_registry as _tr
        from merlin.targetgen.compute_units import compute_units
    except ImportError:                      # targetgen not installed in this sandbox
        return frozenset()
    try:
        contract = _tr.load_contract(target)
    except (FileNotFoundError, KeyError):
        # NO CONTRACT is a real answer -- nobody has declared this target -- and empty is correct for
        # it. Deliberately NOT a bare `except`: a malformed contract, an unreadable file or a bad
        # `compute_units` block must propagate, because silently answering "no engines" for a target
        # whose declaration is BROKEN reads identically to one that declares none, and the two want
        # opposite responses.
        return frozenset()
    return engines_of_units(compute_units(contract or {}))


def facet_families_for(target: str) -> frozenset[str]:
    """The engine-scoped CCA facets a target's silicon warrants, derived from its engine set."""
    return frozenset(f for f in (facet_for(k) for k in engines_for(target)) if f)


# ---------------------------------------------------------------------------------------------
# The coarse backend class, derived rather than restated
# ---------------------------------------------------------------------------------------------
#
# ``runtime.backends.base.TargetClass`` (cpu/gpu/npu) is a THIRD vocabulary for the same subject, and
# ``targetgen.families`` asserts in prose that the compute-unit kinds are "aligned with" it. Five kinds
# cannot align with three tokens by inspection, and a comment cannot be checked -- so the relationship
# lives here, as one total map, and a test enforces it.
#
# Note what this does NOT do: it does not remove the enum. A backend genuinely declares its class at
# registration, and that is a fact about the backend, not something to compute away. What is derived is
# the RELATIONSHIP -- given a target's engines, which class it is -- so the two can no longer disagree.
TARGET_CLASS_OF_ENGINE: dict[str, str] = {
    "systolic": "npu",   # tensor accelerator
    "spatial": "npu",    # ditto: a different array datapath, the same coarse class of silicon
    "simt": "gpu",
    "vector": "cpu",     # a lane engine is a CPU feature, not a separate device
    "scalar": "cpu",
}

#: Which engine decides the class when a target has several. A SIMT cluster that CONTAINS a matrix PE
#: is a GPU with a tensor unit, not an NPU -- so the outermost (containing) engine wins, which is
#: exactly what ``targetgen.target_experiment._primary_kind`` computes.
#:
#: Worth stating plainly, because it is the one place a "primary engine" is the RIGHT question: asking
#: which single class of device this is has a single answer. It is the wrong question for deciding
#: which FACETS or which RTL FACTS a target has, where collapsing the set describes only half of a
#: hybrid -- and that is the collapse to fix, not this one.
_CLASS_PRECEDENCE: tuple[str, ...] = ("simt", "systolic", "spatial", "vector", "scalar")


def target_class_for(engines: Iterable[str]) -> str | None:
    """The coarse backend class (``cpu``/``gpu``/``npu``) for an engine set, or None if it is empty.

    None means "this target has not declared what silicon it is" and must not be defaulted: guessing
    ``cpu`` for an undeclared accelerator is how a result gets attributed to the wrong class of device.
    """
    have = set(engines)
    for kind in _CLASS_PRECEDENCE:
        if kind in have:
            return TARGET_CLASS_OF_ENGINE[kind]
    return None


def check_class_map_is_total(kinds: Iterable[str], classes: Iterable[str]) -> None:
    """Raise unless every kind maps to a class and every class is reachable from some kind.

    The second half matters as much as the first: a class no engine can produce is a token the runtime
    can accept and the compiler can never derive, which is precisely the drift this map replaces.
    """
    known, declared = set(kinds), set(TARGET_CLASS_OF_ENGINE)
    if missing := sorted(known - declared):
        raise KeyError(f"compute-unit kind(s) {missing} map to no TargetClass; add them to "
                       f"TARGET_CLASS_OF_ENGINE")
    if stale := sorted(declared - known):
        raise KeyError(f"TARGET_CLASS_OF_ENGINE maps {stale}, which are not compute-unit kinds")
    produced, valid = set(TARGET_CLASS_OF_ENGINE.values()), set(classes)
    if bad := sorted(produced - valid):
        raise KeyError(f"TARGET_CLASS_OF_ENGINE produces {bad}, which are not TargetClass values")
    if unreachable := sorted(valid - produced):
        raise KeyError(f"TargetClass value(s) {unreachable} are reachable from no compute-unit kind, "
                       f"so nothing could ever derive them")
    if missing_prec := sorted(known - set(_CLASS_PRECEDENCE)):
        raise KeyError(f"kind(s) {missing_prec} have no precedence rank, so a hybrid containing one "
                       f"would have an order-dependent class")
