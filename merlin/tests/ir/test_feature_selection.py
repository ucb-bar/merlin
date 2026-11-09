"""Generic feature selection constraints apply after implication closure."""

import pytest

from merlin.llvmlower import impr_features as F


@pytest.fixture
def feature(monkeypatch):
    entries = dict(F._REGISTRY)
    monkeypatch.setattr(F._FEATURES, "entries", entries)
    monkeypatch.setattr(F, "_REGISTRY", entries)

    def add(name, **kwargs):
        return F.register(F.ImprFeature(name, "PASS", "selection test", **kwargs)).name

    return add


def test_default_metadata_keeps_empty_and_implication_behavior(feature):
    base = feature("test_selection_base")
    wrapper = feature("test_selection_wrapper", implies=frozenset({base}))
    assert F.get(base).alternative_group is None
    assert F.get(base).requires_exactly_one_of == frozenset()
    assert F.normalize(None) == F.normalize([]) == frozenset()
    assert F.normalize([wrapper]) == frozenset({wrapper, base})
    passes = ["canonicalize"]
    assert F.apply_pipeline(passes, F.normalize([])) is passes
    assert F.apply_pipeline(passes, F.normalize([wrapper])) == passes


def test_alternatives_allow_one_per_group_and_unrelated_features(feature):
    a = feature("test_selection_a", alternative_group="test_schedule")
    b = feature("test_selection_b", alternative_group="test_schedule")
    other = feature("test_selection_other", alternative_group="test_other")
    plain = feature("test_selection_plain")
    assert F.normalize([a, other, plain]) == frozenset({a, other, plain})
    with pytest.raises(ValueError, match="alternatives.*test_schedule"):
        F.normalize([b, a])


@pytest.mark.parametrize("selected", [(), ("a",), ("b",), ("a", "b")])
def test_exactly_one_prerequisite_is_a_count_constraint(feature, selected):
    # No alternative group: the prerequisite rule itself must reject two choices.
    a = feature("test_selection_a")
    b = feature("test_selection_b")
    dependent = feature("test_selection_dependent", requires_exactly_one_of=frozenset({a, b}))
    names = [{"a": a, "b": b}[n] for n in selected]
    if len(selected) == 1:
        assert F.normalize([dependent, *names]) == frozenset({dependent, *names})
    else:
        with pytest.raises(ValueError, match="requires exactly one"):
            F.normalize([dependent, *names])


def test_implies_closure_satisfies_prerequisite_before_validation(feature):
    choice = feature("test_selection_choice")
    wrapper = feature("test_selection_wrapper", implies=frozenset({choice}))
    dependent = feature("test_selection_dependent", requires_exactly_one_of=frozenset({choice}))
    assert F.normalize([dependent, wrapper]) == frozenset({dependent, wrapper, choice})


def test_implied_alternative_conflict_is_rejected(feature):
    a = feature("test_selection_a", alternative_group="test_schedule")
    b = feature("test_selection_b", alternative_group="test_schedule")
    wrapper = feature("test_selection_wrapper", implies=frozenset({b}))
    with pytest.raises(ValueError, match="alternatives"):
        F.normalize([a, wrapper])


def test_implied_prerequisite_conflict_is_rejected(feature):
    a = feature("test_selection_a")
    b = feature("test_selection_b")
    wrapper = feature("test_selection_wrapper", implies=frozenset({b}))
    dependent = feature("test_selection_dependent", requires_exactly_one_of=frozenset({a, b}))
    with pytest.raises(ValueError, match="requires exactly one"):
        F.normalize([a, wrapper, dependent])


def test_unknown_implied_name_is_still_rejected(feature):
    wrapper = feature("test_selection_wrapper", implies=frozenset({"test_selection_missing"}))
    with pytest.raises(KeyError, match="unknown impr feature"):
        F.normalize([wrapper])
