"""Native fixture selection boundaries, without SDK or tool execution."""

import copy
import json
from types import SimpleNamespace

import original_scalar_conversion_fixtures as F
import pytest
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_scalar_conversion as V
from test_original_integer_scalar_binary_sources import tensor_binding
from test_original_scalar_binary_plan import budget as source_budget
from test_original_scalar_binary_sources import declarations

from merlin.targetgen.frontend_original_call import call_contracts


def test_existing_fixture_default_keeps_v1_selection(monkeypatch):
    calls = []
    factory = object()
    selected = object()

    def prepare(original, *, version=1):
        calls.append((original, version))
        return selected

    monkeypatch.setattr(F, "prepare_native_originals", prepare)
    assert F.native_originals.__wrapped__(factory) is selected
    assert calls == [(factory, 1)]


def test_integer_fixture_requires_explicit_v2_selection(monkeypatch):
    calls = []
    factory = object()

    def prepare(original, *, version=1):
        calls.append((original, version))

    monkeypatch.setattr(F, "prepare_native_originals", prepare)
    F.integer_native_originals.__wrapped__(factory)
    assert calls == [(factory, 2)]


@pytest.mark.parametrize("version", [0, 3, True, 2.0])
def test_unsupported_fixture_version_refuses_before_source_creation(version):
    with pytest.raises(ValueError, match="explicit supported version"):
        F.prepare_native_originals(object(), version=version)


def test_v2_requires_explicit_getter_compiler_before_source_creation(monkeypatch):
    for name in (
        "MERLIN_TEST_TORCH_PYTHON",
        "MERLIN_TEST_M2M_ROOT",
        "MERLIN_TEST_M2M_COMMIT",
        "MERLIN_TEST_OPERATOR_DECLARATIONS",
        "MERLIN_TEST_TORCH_SOURCE_ROOT",
        "MERLIN_TEST_FIRTOOL",
        "MERLIN_TEST_MLIR_OPT",
    ):
        monkeypatch.setenv(name, "/explicit-selection-unused-before-refusal")
    monkeypatch.delenv("MERLIN_TEST_TENSOR_ARGUMENT_COMPILER", raising=False)
    with pytest.raises(pytest.skip.Exception, match="explicit public/native source and tool"):
        F.prepare_native_originals(object(), version=2)


def test_v1_budget_and_all_prior_limits_remain_unchanged():
    prior = {
        "max_source_bytes": 2000000,
        "max_nesting": 64,
        "max_operations": 30,
        "max_tensor_elements": 1000,
        "max_members": 100,
        "max_total_tensor_elements": 10000,
        "max_total_source_bytes": 100000000,
        "timeout_s": 180,
    }
    assert F.conversion_budget() == F.conversion_budget(version=1) == prior
    selected = F.conversion_budget(version=2)
    assert {key: selected[key] for key in prior if key != "max_total_source_bytes"} == {
        key: value for key, value in prior.items() if key != "max_total_source_bytes"
    }
    assert selected["max_promotion_tensor_elements"] == 1000
    assert selected["max_total_promotion_tensor_elements"] == 10000


def complete_declared_sources(monkeypatch, tmp_path):
    """Mock only native schema/default seams; execute the ordinary factories."""
    documents = [declarations(target, scalar) for target, scalar in F.DECLARED_CALLS]
    paths = [str(tmp_path / f"graph-{index}.json") for index in range(len(documents))]
    basis = SimpleNamespace(
        graph_sources=[SimpleNamespace(path=path) for path in paths],
        declaration_json=json.dumps({"members": [{"id": f"independent-{index}"} for index in range(len(paths))]}),
    )
    observed, schema_members = [], []
    for index, path in enumerate(paths):
        directory = tmp_path / "sources" / str(index)
        directory.mkdir(parents=True)
        observed.append(
            {"graph_path": path, "request": {}, "observation": str(directory / "observation.json"), "invocation": {}}
        )
        literal = F.DECLARED_CALLS[index][1]
        schema_members.append(
            {
                "graph_path": path,
                "tensor_bindings": [tensor_binding(call_contracts(*documents[index])[0])]
                if type(literal) is int
                else [],
            }
        )
    monkeypatch.setattr(C.D, "observe_members", lambda **kwargs: copy.deepcopy(observed))
    monkeypatch.setattr(
        C.D, "verify_member", lambda row, **kwargs: copy.deepcopy(documents[paths.index(row["graph_path"])])
    )
    record = C.observe(
        schema_record={"schema": "merlin.independent_operator_schema_intake.v2", "members": schema_members},
        basis=basis,
        numerical_semantics={"original_numerical_policy_pending": True},
        budget=source_budget(),
        destination=tmp_path / "sources",
        version=7,
    )
    return record, basis


def test_v2_reserves_complete_six_call_eighteen_slot_products_before_native_conversion(monkeypatch, tmp_path):
    record, basis = complete_declared_sources(monkeypatch, tmp_path)

    def forbidden_execution(*args, **kwargs):
        raise AssertionError("preflight must not execute a native conversion")

    monkeypatch.setattr(V.I, "run", forbidden_execution)
    selected = F.conversion_budget(version=2)
    members, totals = V.required_members(record, basis=basis, budget=selected)
    assert len(members) == 18 and all(source is not None for _, source in members)
    assert [(row["original_member_id"], row["cohort"], row["extent"]) for row, _ in members] == [
        (f"independent-{index}", cohort, extent) for index in range(6) for cohort, extent in C.required_source_cohorts()
    ]
    assert totals["tensor_elements"] == 240 and totals["promotion_tensor_elements"] == 63
    assert totals["source_bytes"] == 3897 and totals["reserved_product_bytes"] == 110000000
    assert selected["max_total_source_bytes"] == 146000000
    assert totals["source_bytes"] + totals["reserved_product_bytes"] <= selected["max_total_source_bytes"]
    assert "original_numerical_policy_and_reference" in V._UNKNOWN
    assert "mandatory_coverage_and_phase_admission" in V._UNKNOWN
    for maximum in (100000000, totals["source_bytes"] + totals["reserved_product_bytes"] - 1):
        with pytest.raises(ValueError, match="complete product reservation exceeds its aggregate byte budget"):
            V.required_members(record, basis=basis, budget={**selected, "max_total_source_bytes": maximum})
