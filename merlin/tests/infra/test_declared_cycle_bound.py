"""A capsule's one slot for a cycle ceiling, and what it is allowed to put in it.

``performance.cost.projected_cycles`` is the field a cost plane reads to find the ceiling a capsule
declares, and such a reader needs an integer: a word resolves to no ceiling, so the assessment comes
back incomplete and the plane can reject a physically impossible count but never a merely slow one.

Measured before this: 113 of 595 capsules carry the field, all 113 write ``derived_at_preflight``, no
capsule writes an integer, and no code anywhere reads that string -- no preflight pass resolves it
and no branch matches on it. The field also had no schema type, so it reached disk through
``additionalProperties: true`` and a typo in it was indistinguishable from a declaration.

Two gates, deliberately: :mod:`merlin.perf.cycle_bound` owns the vocabulary and the check, and
``capsule.schema.json`` states the same thing to every consumer that validates a capsule without
importing Python. The tests below assert the two agree, because a value one refuses and the other
admits is worse than no gate at all.

These tests do not give any capsule a bound. They make the ABSENCE of one a declaration rather than
an accident, which is the precondition for a capsule ever owing a number.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.perf import cycle_bound as CB
from merlin.targetgen.contract.schemas import ContractViolation, load_schema, validate_capsule


def _generator():
    from merlin_experiments.phase0 import profiles

    return profiles


def _capsules():
    root = repo_root() / "merlin" / "contract" / "capsules"
    for path in sorted(root.rglob("capsule.yaml")):
        doc = yaml.safe_load(path.read_text())
        if isinstance(doc, dict):
            yield path, doc


def _with_cycles(value):
    return {"cost": {"projected_cycles": value}}


# ---------------------------------------------------------------------------------------------------
# the declaration side
# ---------------------------------------------------------------------------------------------------


def test_an_integer_ceiling_is_now_expressible() -> None:
    """The point of the change. Before it, the schema modelled no `performance` block at all, so this
    said nothing either way -- the field would have accepted the integer and the string 'banana'
    alike."""
    schema = load_schema("capsule")["properties"]["performance"]
    options = schema["properties"]["cost"]["properties"]["projected_cycles"]["anyOf"]
    kinds = {opt["type"] for opt in options}
    assert kinds == {"integer", "string"}
    integer = next(opt for opt in options if opt["type"] == "integer")
    assert integer["minimum"] == 1, "a ceiling of zero is not a bound any correct run can meet"


@pytest.mark.parametrize("value", [1, 1024, 4_000_000])
def test_a_positive_count_validates(value) -> None:
    CB.validate(value, owner="T")


@pytest.mark.parametrize("value", [0, -1, -4096])
def test_a_ceiling_no_run_could_meet_is_refused(value) -> None:
    """These would have passed the generic non-emptiness check the block already had: ``0`` is not
    None, not an empty string and not an empty container."""
    with pytest.raises(ValueError, match="not"):
        CB.validate(value, owner="T")


@pytest.mark.parametrize("value", [None, 1.5, True, [], {}, "4096", "derived at preflight", "soon"])
def test_anything_that_is_neither_a_count_nor_a_declared_reason_is_refused(value) -> None:
    """Including the near-misses. ``"4096"`` is the one that matters: a cycle count as a STRING reads
    as a bound to a human and resolves to no ceiling in the reader, which is the exact failure mode
    the whole field already had."""
    with pytest.raises(ValueError, match="projected_cycles"):
        CB.validate(value, owner="T")


def test_every_no_bound_word_states_why_there_is_no_bound() -> None:
    """The vocabulary is closed so that an unrecognised word fails instead of silently meaning 'no
    ceiling'. An entry with no reason would put the word back to being a bare token."""
    words = CB.NO_CYCLE_BOUND
    assert words, "an empty vocabulary would make every word a refusal, including the corpus's own"
    for word, why in words.items():
        CB.validate(word, owner="T")
        assert isinstance(why, str) and len(why.strip()) > 20, f"{word!r} declares no reason"


# ---------------------------------------------------------------------------------------------------
# the corpus as it stands
# ---------------------------------------------------------------------------------------------------


def test_the_corpus_still_validates_and_the_census_is_what_it_was() -> None:
    """Backward compatibility, stated as the measurement it came from. If this count moves, either a
    capsule gained a bound (good, and this number should be updated deliberately) or the corpus was
    regenerated into a shape nobody reviewed."""
    declared, wordy, numeric = 0, 0, 0
    for path, capsule in _capsules():
        validate_capsule(capsule)  # raises ContractViolation naming the file
        value = ((capsule.get("performance") or {}).get("cost") or {}).get("projected_cycles")
        if value is None:
            continue
        declared += 1
        if isinstance(value, int) and not isinstance(value, bool):
            numeric += 1
        else:
            wordy += 1
            assert value in CB.NO_CYCLE_BOUND, f"{path}: undeclared no-bound word {value!r}"
    assert declared == wordy + numeric
    assert declared > 0, "no capsule carries the field at all -- this test would be vacuous"


def test_the_schema_and_the_vocabulary_refuse_the_same_values() -> None:
    """The two gates must agree. A value one refuses and the other admits would pass on whichever path
    consults only the permissive gate -- and the corpus's one hand-authored performance block never
    goes through generation at all."""
    gen = CB
    base = {
        "name": "T", "kind": "layer", "source_role": "derived_sweep", "label": "public",
        "operation": {"op": "matmul", "attributes": {}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
        "expected": {"instruction_classes": []}, "required_oracle_tiers": ["L2"],
    }  # fmt: skip
    for good in (4096, *sorted(gen.NO_CYCLE_BOUND)):
        validate_capsule({**copy.deepcopy(base), "performance": _with_cycles(good)})
    for bad in (0, -1, "4096", "soon"):
        with pytest.raises(ContractViolation):
            validate_capsule({**copy.deepcopy(base), "performance": _with_cycles(bad)})
        with pytest.raises(ValueError):
            gen.validate(bad, owner="T")


def test_the_shared_template_is_still_the_only_place_a_perf_block_is_authored() -> None:
    """The guard this change had to respect. A per-capsule cycle number is a target fact, and the
    obvious way to declare one -- a perf entry in a target's own profile -- is refused, because
    onboarding a target must not fork the experiment it is compared under. So the field's vocabulary
    widened and its HOME did not."""
    template = repo_root() / "experiments/templates/phase0/performance.yaml"
    offender = {"capsules": [{"name": "X", "cat": "perf", "performance": {"cost": {}}}]}
    misplaced = _generator()._target_local_perf_declarations(offender)
    assert misplaced, "a target-local perf entry must still be detected"
    assert template.is_file()


def test_reading_a_bound_tells_absent_from_unparseable() -> None:
    """``declared_cycles`` answers None for a capsule that declares no bound AND for one that carries
    no performance block, because both owe no number. It RAISES on a malformed declaration: a capsule
    whose field nobody can parse has not declared the absence of a bound, and reading it as None would
    reinstate exactly the permissive default this module removes."""
    assert CB.declared_cycles({"name": "A"}) is None
    assert CB.declared_cycles({"name": "B", "performance": {}}) is None
    for word in CB.NO_CYCLE_BOUND:
        assert CB.declared_cycles({"name": "C", "performance": _with_cycles(word)}) is None
    assert CB.declared_cycles({"name": "D", "performance": _with_cycles(4096)}) == 4096
    for bad in (0, "4096", "soon"):
        with pytest.raises(CB.CycleBoundError):
            CB.declared_cycles({"name": "E", "performance": _with_cycles(bad)})


def test_the_corpus_is_readable_through_the_same_accessor() -> None:
    """Every capsule on disk parses through ``declared_cycles`` without raising -- so the vocabulary
    describes what the corpus actually contains rather than what it ought to."""
    assert all(CB.declared_cycles(capsule) is None for _path, capsule in _capsules())
