"""A capsule may not be required to issue an instruction its operation does not perform.

``corpus_spec._classes_source`` resolves three regimes. Two of them ask the op what it owes; the third
-- the RoCC ``encoding.semantic_class`` path -- accepted ``op``, ``epilogue`` and ``movement`` and
returned the same weight-stationary systolic sequence regardless. So on a RoCC target every capsule
owed the systolic multiply, including a ``movement`` capsule that performs no arithmetic at all.

Measured before the fix: six tracked capsules (``isa/FT00``, ``isa/FT01``, the two ``isa/SY_movement_*``
and two HIDDEN ``HM0*``) required PRELOAD and COMPUTE_PRELOADED of a pure tile copy. That error is in
the direction running the capsule cannot catch -- a backend that moves the tile correctly and issues no
multiply is marked non-conformant -- and ``build_movement`` passes ``movement=True`` precisely to
prevent it.

The two directions are tested separately because each has its own failure mode: dropping too little
fabricates a demand, and dropping too much silently excuses a real one.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import semantic_families as SF
from merlin.targetgen.corpora import descriptor_path
from merlin.targetgen.target_experiment import load_target_experiment

sys.path.insert(0, str(merlin_dir() / "contract" / "capsules"))


def _binding(target: str):
    import generate_corpus as GC
    prof = GC.load_profile(target)
    te = load_target_experiment(descriptor_path(target))
    return CS.derive_binding(te, prof.get("datapath") or {})


def _rocc_targets():
    """Targets whose classes come from an ``encoding.semantic_class`` map and resolve to something."""
    out = []
    for t in ("gemmini", "mx_gemmini"):
        try:
            b = _binding(t)
            if b.classes_for(op="matmul", output_dtype=b.cap_dtype(b.accum_dtype)):
                out.append(t)
        except Exception:                          # noqa: BLE001 -- an unresolvable target is not a case
            continue
    return out


@pytest.fixture(scope="module")
def rocc():
    targets = _rocc_targets()
    if not targets:
        pytest.skip("no target resolves a RoCC encoding class list")
    return targets


def _contraction_classes(classes) -> set[str]:
    """Which of these class names the SHARED class vocabulary files under the contraction."""
    return {c for c in classes if SF.from_isa_class(c) == "contraction"}


def test_a_non_contraction_op_does_not_owe_the_contraction_classes(rocc):
    """An elementwise/movement capsule must not be required to multiply."""
    for target in rocc:
        b = _binding(target)
        odt = b.cap_dtype(b.accum_dtype)
        for op in ("bias_add", "movement", "rmsnorm"):
            prims = SF.primitives_of(SF.from_op(op) or "")
            assert prims and "contraction" not in prims, f"{op} is the wrong fixture for this test"
            got = b.classes_for(op=op, output_dtype=odt, movement=(op == "movement"))
            assert not _contraction_classes(got), (
                f"{target}: a {op!r} capsule is required to issue "
                f"{sorted(_contraction_classes(got))}, which belong to the contraction it does not "
                f"perform; a conformant backend would be marked non-conformant")


def test_a_contraction_still_owes_the_full_sequence(rocc):
    """The over-correction mirror: the fix must not excuse a real multiply."""
    for target in rocc:
        b = _binding(target)
        odt = b.cap_dtype(b.accum_dtype)
        for op in ("matmul", "linear", "fused_matmul_bias"):
            got = b.classes_for(op=op, output_dtype=odt)
            assert _contraction_classes(got), (
                f"{target}: a {op!r} capsule owes NO contraction class, so a backend that never "
                f"multiplies would satisfy it")


def test_a_non_contraction_op_still_owes_its_data_motion(rocc):
    """Only the contraction classes are dropped -- the op's operands still have to move.

    An empty requirement is satisfied by emitting nothing at all, so replacing a fabricated demand
    with no demand would trade one unfalsifiable capsule for another.
    """
    for target in rocc:
        b = _binding(target)
        got = b.classes_for(op="movement", output_dtype=b.cap_dtype(b.accum_dtype), movement=True)
        assert got, f"{target}: a movement capsule owes nothing at all"


def test_a_class_the_shared_vocabulary_cannot_place_is_kept(rocc):
    """Fail closed: an unrecognised class name is not evidence that it is unnecessary.

    The plumbing classes (CONFIG*, FLUSH) are exactly this case -- ``_ISA_CLASS_FAMILY`` maps them to
    no family on purpose -- so they must survive into a non-contraction op's demand rather than being
    dropped along with the multiply.
    """
    for target in rocc:
        b = _binding(target)
        odt = b.cap_dtype(b.accum_dtype)
        full = b.classes_for(op="matmul", output_dtype=odt)
        unplaceable = [c for c in full if SF.from_isa_class(c) is None]
        if not unplaceable:
            continue
        got = b.classes_for(op="movement", output_dtype=odt, movement=True)
        assert set(unplaceable) <= set(got), (
            f"{target}: classes {sorted(set(unplaceable) - set(got))} were dropped from a movement "
            f"capsule, but the shared vocabulary files them under no family -- silently dropping what "
            f"cannot be classified is the failure the parsing rule forbids")


def test_a_union_of_per_op_demands_is_in_the_targets_issue_order(rocc):
    """A whole-model capstone unions its ops' class lists; the result must be issuable.

    The union is built in first-encounter order, which only looked right while every op returned the
    identical full sequence. Once a movement op correctly owes fewer classes, the op the model happens
    to mention first decides the order -- and a model whose first eligible op was a copy reported its
    store class BEFORE its multiply.
    """
    from merlin.targetgen.capsule_source import _in_issue_order

    for target in rocc:
        b = _binding(target)
        odt = b.cap_dtype(b.accum_dtype)
        reference = list(b.classes_for(op="matmul", output_dtype=odt))
        if len(reference) < 3:
            continue
        # exactly the situation the capstone hit: the movement demand seen first, then the contraction
        first = list(b.classes_for(op="movement", output_dtype=odt, movement=True))
        union = first + [c for c in reference if c not in first]
        ordered = _in_issue_order(union, b, odt)
        assert ordered == [c for c in reference if c in set(union)], (
            f"{target}: the union {ordered} is not in the target's own issue order {reference}")
