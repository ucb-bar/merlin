from __future__ import annotations

import importlib.util
import json

import pytest

from merlin.targetgen import core_aten_cover as cover
from merlin.targetgen import core_aten_samples as samples

requires_z3 = pytest.mark.skipif(importlib.util.find_spec("z3") is None, reason="requires z3 (merlin verify extra)")
requires_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")


@requires_z3
def test_exact_coverage_of_a_small_synthetic_universe():
    result = cover.exact_minimum_cover(
        ["aten.a.default", "aten.b.default"],
        {"a": ["aten.a.default"], "b": ["aten.b.default"]},
    )
    assert result.status == "optimal"
    assert result.selected_capsules == ["a", "b"]
    assert result.selected_union_matches_denominator
    assert result.minimum_cardinality_lower_bound == 2
    assert result.minimum_cardinality_upper_bound == 2
    assert result.objective_bounds_closed


@requires_z3
def test_redundant_capsules_are_omitted():
    result = cover.exact_minimum_cover(
        ["aten.a.default", "aten.b.default"],
        {
            "all": ["aten.a.default", "aten.b.default"],
            "only_a": ["aten.a.default"],
            "only_b": ["aten.b.default"],
        },
    )
    assert result.selected_capsules == ["all"]


@requires_z3
def test_a_multi_operator_capsule_reduces_the_solution_size():
    without = cover.exact_minimum_cover(
        ["aten.a.default", "aten.b.default", "aten.c.default"],
        {"a": ["aten.a.default"], "b": ["aten.b.default"], "c": ["aten.c.default"]},
    )
    with_multi = cover.exact_minimum_cover(
        ["aten.a.default", "aten.b.default", "aten.c.default"],
        {
            "a": ["aten.a.default"],
            "b": ["aten.b.default"],
            "c": ["aten.c.default"],
            "ab": ["aten.a.default", "aten.b.default"],
        },
    )
    assert without.selected_count == 3
    assert with_multi.selected_count == 2


@requires_z3
def test_selection_is_deterministic_under_reordered_input():
    universe = ["aten.a.default", "aten.b.default"]
    forward = {"z": universe, "a": universe}
    reverse = dict(reversed(list(forward.items())))
    assert cover.exact_minimum_cover(universe, forward).selected_capsules == ["a"]
    assert cover.exact_minimum_cover(list(reversed(universe)), reverse).selected_capsules == ["a"]


def test_an_uncovered_operator_is_reported_before_optimization():
    result = cover.exact_minimum_cover(
        ["aten.a.default", "aten.missing.default"],
        {"a": ["aten.a.default"]},
    )
    assert result.status == "uncovered"
    assert result.solver_version == "not_run"
    assert result.minimum_cardinality_lower_bound is None
    assert result.minimum_cardinality_upper_bound is None
    assert not result.objective_bounds_closed
    assert result.uncovered_overloads == ["aten.missing.default"]
    assert result.selected_capsules == []


@requires_z3
def test_partial_mode_proves_the_minimum_for_only_the_coverable_universe():
    result = cover.exact_minimum_cover(
        ["aten.a.default", "aten.missing.default"],
        {"a": ["aten.a.default"]},
        allow_partial=True,
    )
    assert result.status == "partial_optimal"
    assert result.selected_capsules == ["a"]
    assert result.selected_union_matches_coverable
    assert not result.selected_union_matches_denominator


def test_overloads_remain_distinct():
    result = cover.exact_minimum_cover(
        ["aten.add.Scalar", "aten.add.Tensor"],
        {"tensor_only": ["aten.add.Tensor"]},
    )
    assert result.uncovered_overloads == ["aten.add.Scalar"]
    assert result.coverable_count == 1


@requires_z3
def test_final_selected_union_is_independently_validated():
    universe = ["aten.a.default", "aten.b.default", "aten.c.default"]
    candidates = {
        "ab": ["aten.a.default", "aten.b.default"],
        "bc": ["aten.b.default", "aten.c.default"],
    }
    result = cover.exact_minimum_cover(universe, candidates)
    selected_union = set().union(*(set(candidates[name]) for name in result.selected_capsules))
    assert selected_union == set(universe)
    assert result.selected_union == sorted(universe)
    assert result.selected_union_matches_denominator


class _UnknownZ3:
    unknown = object()
    sat = object()
    unsat = object()


class _UnknownSolver:
    def check(self):
        return _UnknownZ3.unknown

    def reason_unknown(self):
        return "synthetic timeout"


class _ErrorSolver:
    def check(self):
        raise RuntimeError("synthetic solver failure")


def test_solver_unknown_fails_closed():
    with pytest.raises(cover.SolverFailure, match="unknown.*synthetic timeout"):
        cover._check(_UnknownSolver(), _UnknownZ3, "test query")


def test_solver_error_fails_closed():
    with pytest.raises(cover.SolverFailure, match="synthetic solver failure"):
        cover._check(_ErrorSolver(), _UnknownZ3, "test query")


def test_denominator_digest_binds_overload_boundaries():
    document = cover.denominator_document(
        {
            "torch": "test",
            "n_core": 2,
            "ops": ["aten.add.Scalar", "aten.add.Tensor"],
        }
    )
    assert document["overload_count"] == 2
    assert len(document["overload_sha256"]) == 64


def test_matrix_uses_exact_provenance_and_excludes_tagless_capsules(tmp_path):
    tagged = tmp_path / "tagged"
    tagged.mkdir()
    (tagged / "capsule.yaml").write_text("name: tagged\n", encoding="utf-8")
    (tagged / "capsule.linalg.mlir").write_text(
        '%0 = "test.op"() {prov.aten = "aten.add.Tensor"} : () -> ()\n', encoding="utf-8"
    )
    tagless = tmp_path / "tagless"
    tagless.mkdir()
    (tagless / "capsule.yaml").write_text("name: tagless\n", encoding="utf-8")
    (tagless / "capsule.interface.mlir").write_text("builtin.module {}\n", encoding="utf-8")

    matrix = cover.capsule_operator_matrix([tmp_path], ["aten.add.Scalar", "aten.add.Tensor"])
    assert matrix["capsules"] == {"tagged": ["aten.add.Tensor"], "tagless": []}
    assert matrix["eligible_candidates"] == ["tagged"]
    assert matrix["provenance"]["tagless"]["status"] == "ineligible_no_provenance"


def test_matrix_fails_closed_when_two_capture_files_disagree(tmp_path):
    capsule = tmp_path / "ambiguous"
    capsule.mkdir()
    (capsule / "capture.json").write_text(json.dumps({"status": "captured"}), encoding="utf-8")
    (capsule / "a.mlir").write_text('prov.aten = "aten.add.Scalar"\n', encoding="utf-8")
    (capsule / "b.mlir").write_text('prov.aten = "aten.add.Tensor"\n', encoding="utf-8")
    with pytest.raises(cover.CoverageFailure, match="ambiguous"):
        cover.capsule_operator_matrix([tmp_path], ["aten.add.Scalar", "aten.add.Tensor"])


@requires_torch
def test_every_grouped_gap_filler_is_a_valid_eager_pytorch_program():
    for name, (loader_source, requested) in sorted(samples.SAMPLE_LOADERS.items()):
        namespace = {"__name__": f"core_aten_sample_{name}"}
        exec(compile(loader_source, f"<{name}>", "exec"), namespace)  # noqa: S102 - controlled source
        model, inputs = namespace["get_model_and_inputs"]()
        result = model(*inputs)
        assert result is not None
        assert requested
        assert len(requested) == len(set(requested))
        assert all(op.startswith("aten.") for op in requested)
