from __future__ import annotations

from copy import deepcopy

from merlin.targetgen.core_aten_eqsat import quotient_observational_equivalents


def _case(identifier: str, assignment: dict[str, str], obligation: str, *, expected: int = 1):
    return {
        "case_id": identifier,
        "overload": "aten.test.default",
        "schema": "test",
        "rng_seed": 0,
        "comparison": "torch_close",
        "comparison_parameters": {"rtol": 0.0, "atol": 0.0, "equal_nan": True},
        "arguments": {"args": [], "kwargs": {}},
        "post_arguments": {"args": [], "kwargs": {}},
        "mutated_arguments": [],
        "output_input_aliases": [],
        "expected": expected,
        "expected_sha256": "test",
        "partition_assignment": assignment,
        "covered_obligations": [obligation],
    }


def test_eqsat_quotient_unions_labels_of_identical_executions() -> None:
    candidates = {
        "b": _case("b", {"values": "ordinary"}, "ordinary"),
        "a": _case("a", {"values": "sign_mix"}, "sign_mix"),
    }
    reduced, audit = quotient_observational_equivalents(candidates)
    assert list(reduced) == ["a"]
    assert reduced["a"]["covered_obligations"] == ["ordinary", "sign_mix"]
    assert reduced["a"]["equivalent_partition_assignments"] == [
        {"values": "ordinary"},
        {"values": "sign_mix"},
    ]
    assert audit["engine"] == "xdsl-eqsat"
    assert audit["eliminated_candidate_count"] == 1
    assert audit["nontrivial_eclass_count"] == 1


def test_eqsat_quotient_is_deterministic_under_reordered_input() -> None:
    a = _case("a", {"dtype": "float32"}, "f32")
    b = _case("b", {"dtype": "float"}, "float")
    first = quotient_observational_equivalents({"a": a, "b": b})
    second = quotient_observational_equivalents({"b": deepcopy(b), "a": deepcopy(a)})
    assert first == second


def test_eqsat_never_merges_different_observable_results() -> None:
    reduced, audit = quotient_observational_equivalents(
        {
            "a": _case("a", {"values": "one"}, "one", expected=1),
            "b": _case("b", {"values": "two"}, "two", expected=2),
        }
    )
    assert set(reduced) == {"a", "b"}
    assert audit["eliminated_candidate_count"] == 0
