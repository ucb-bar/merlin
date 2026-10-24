from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from merlin.targetgen.core_aten_bounded import (
    BoundedCoverageProfile,
    _case_id,
    _obligations,
    bounded_suite_digest,
)
from merlin.targetgen.core_aten_cases import case_document, core_aten_cases, eager_case_observation
from merlin.targetgen.core_aten_cover import overload_digest
from merlin.targetgen.core_aten_eqsat import quotient_observational_equivalents
from merlin.targetgen.core_aten_eval import (
    BoundedEagerSuiteAdapter,
    EagerSuiteAdapter,
    JsonCommandSuiteAdapter,
    evaluate_bounded_core_aten_suite,
    evaluate_core_aten_suite,
)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dtype,values,wrong", [("int64", [1000000], [1000001]), ("bool", [True], [False])])
def test_nonfloating_tensor_values_are_exact(dtype, values, wrong):
    from merlin.targetgen.core_aten_eval import _assert_document_close

    expected = dict(
        kind="tensor", dtype=dtype, shape=[1], stride=[1], storage_offset=0, requires_grad=False, values=values
    )
    _assert_document_close(expected, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(AssertionError):
        _assert_document_close(expected, {**expected, "values": wrong}, rtol=1e-5, atol=1e-6)


def _corpus(*names: str) -> dict:
    cases = [case_document(core_aten_cases()[name]) for name in sorted(names)]
    return {
        "schema_version": 1,
        "pytorch_version": torch.__version__,
        "overload_count": len(cases),
        "case_count": len(cases),
        "complete": True,
        "cases": cases,
    }


@dataclass
class RecordingHardwareAdapter:
    name: str = "recording-hardware"
    calls: int = 0

    def execute(self, corpus):
        self.calls += 1
        results = {}
        for case in corpus["cases"]:
            observation = eager_case_observation(case)
            observation["execution_kind"] = "hardware"
            observation["evidence"] = {
                "device": "unit-test-device",
                "decomposed": True,
                "receipt": f"unit-test::{case['overload']}",
            }
            results[case["overload"]] = observation
        return results

    def verify_hardware_receipt(self, case, observation):
        return observation["evidence"]["receipt"] == f"unit-test::{case['overload']}"


def test_one_function_call_runs_the_complete_suite_once() -> None:
    adapter = RecordingHardwareAdapter()
    result = evaluate_core_aten_suite(adapter)
    assert adapter.calls == 1
    assert result["status"] == "pass"
    assert result["passed_count"] == result["case_count"] == 193


def test_decomposition_and_provenance_do_not_affect_semantic_pass() -> None:
    adapter = RecordingHardwareAdapter()
    result = evaluate_core_aten_suite(adapter, corpus=_corpus("aten.add.Tensor"), validate_live_denominator=False)
    assert result["cases"]["aten.add.Tensor"]["status"] == "pass"
    assert result["cases"]["aten.add.Tensor"]["evidence"]["decomposed"] is True


def test_requested_hardware_rejects_host_execution() -> None:
    result = evaluate_core_aten_suite(
        EagerSuiteAdapter(),
        corpus=_corpus("aten.add.Tensor"),
        required_execution_kind="hardware",
        validate_live_denominator=False,
    )
    assert result["status"] == "fail"
    assert result["status_counts"] == {"execution_kind_mismatch": 1}


def _crash(corpus):
    raise RuntimeError("queue unavailable")


@pytest.mark.parametrize(
    "name,execute,overloads,counts",
    [
        pytest.param("missing", lambda corpus: {}, ["aten.add.Tensor"], {"missing_result": 1}, id="missing_result"),
        pytest.param(
            "malformed",
            lambda corpus: {corpus["cases"][0]["overload"]: ["not", "a", "mapping"]},
            ["aten.add.Tensor"],
            {"invalid_result": 1},
            id="malformed_observation",
        ),
        pytest.param(
            "crashing", _crash, ["aten.add.Scalar", "aten.add.Tensor"], {"adapter_error": 2}, id="batch_adapter_crash"
        ),
    ],
)
def test_missing_malformed_or_crashed_results_fail_closed(name, execute, overloads, counts) -> None:
    adapter = SimpleNamespace(name=name, execute=execute)
    result = evaluate_core_aten_suite(adapter, corpus=_corpus(*overloads), validate_live_denominator=False)
    assert result["status_counts"] == counts


def test_compiler_failure_is_preserved() -> None:
    failure = {"status": "compile_failure", "reason": "unsupported lowering", "evidence": {"stage": "merlin"}}
    adapter = SimpleNamespace(name="compile-failure", execute=lambda corpus: {corpus["cases"][0]["overload"]: failure})
    result = evaluate_core_aten_suite(adapter, corpus=_corpus("aten.add.Tensor"), validate_live_denominator=False)
    verdict = result["cases"]["aten.add.Tensor"]
    assert verdict["status"] == "compile_failure"
    assert verdict["reason"] == "unsupported lowering"


def test_output_mutation_and_alias_mismatches_fail() -> None:
    corpus = _corpus("aten.alias.default", "aten.resize_.default")
    adapter = RecordingHardwareAdapter()
    observations = adapter.execute(corpus)
    observations["aten.alias.default"]["output_input_aliases"] = []
    observations["aten.resize_.default"]["mutated_arguments"] = []
    adapter = SimpleNamespace(name="fixed", execute=lambda corpus: observations)
    result = evaluate_core_aten_suite(adapter, corpus=corpus, validate_live_denominator=False)
    assert result["status_counts"] == {"semantic_mismatch": 2}


def test_json_command_adapter_invokes_one_external_batch(tmp_path) -> None:
    corpus = _corpus("aten.add.Tensor")
    observation = eager_case_observation(corpus["cases"][0])
    observation["execution_kind"] = "hardware"
    observation["evidence"]["receipt"] = "external-test-receipt"
    response_json = json.dumps({"schema_version": 1, "results": {"aten.add.Tensor": observation}}, sort_keys=True)
    runner = tmp_path / "runner.py"
    runner.write_text(
        """\
import json
import pathlib
import sys

request = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert request["pytorch_version"]
assert request["complete"] is True
assert [case["overload"] for case in request["cases"]] == ["aten.add.Tensor"]
pathlib.Path(sys.argv[2]).write_text(RESPONSE + "\\n", encoding="utf-8")
""".replace("RESPONSE", repr(response_json)),
        encoding="utf-8",
    )
    adapter = JsonCommandSuiteAdapter(
        argv=(sys.executable, str(runner), "{request}", "{response}"),
        name="external-hardware",
        receipt_verifier=lambda _case, observed: observed["evidence"]["receipt"] == "external-test-receipt",
    )

    result = evaluate_core_aten_suite(adapter, corpus=corpus, validate_live_denominator=False)

    assert result["status"] == "pass"
    assert result["status_counts"] == {"pass": 1}


def test_default_api_rejects_a_declared_complete_subset() -> None:
    with pytest.raises(ValueError, match="live Core ATen denominator"):
        evaluate_core_aten_suite(EagerSuiteAdapter(), corpus=_corpus("aten.add.Tensor"))


@pytest.fixture
def bounded_unit_corpus():
    assignment = {
        "control:alpha": "2",
        "dtype": "float32",
        "layout": "contiguous",
        "shape": "rectangular",
        "values": "ordinary",
    }
    case_id = _case_id("aten.add.Tensor", assignment)
    obligations = sorted(_obligations("aten.add.Tensor", assignment))
    case = {
        **case_document(core_aten_cases()["aten.add.Tensor"]),
        "case_id": case_id,
        "partition_assignment": assignment,
        "covered_obligations": obligations,
    }
    quotient, eqsat = quotient_observational_equivalents({case_id: case})
    case = quotient[case_id]
    # One candidate covers nonempty obligations, so both exact bounds are one.
    cover = dict(
        status="optimal",
        selected_capsules=[case_id],
        objective_value=1,
        minimum_cardinality_lower_bound=1,
        minimum_cardinality_upper_bound=1,
        objective_bounds_closed=True,
        solver="unit singleton proof",
        solver_version="1",
    )
    profile = BoundedCoverageProfile()
    corpus = {
        "schema_version": 1,
        "scope": "bounded unit test",
        "pytorch_version": torch.__version__,
        "profile": profile.to_dict(),
        "profile_sha256": profile.sha256,
        "generator_sha256": "a" * 64,
        "engines": {
            "set_cover": cover["solver"],
            "set_cover_version": cover["solver_version"],
            "minimum_certificate": "unit-test certificate",
            "observational_quotient": "xdsl-eqsat",
            "observational_quotient_version": eqsat["xdsl_version"],
            "eqsat_rewrite_rules": [],
        },
        "overload_count": 1,
        "denominator_sha256": "unit-test-denominator",
        "selected_count": 1,
        "witnessed_obligation_count": len(obligations),
        "witnessed_obligations": obligations,
        "selected_union_sha256": overload_digest(obligations),
        "complete": True,
        "selected_cases": [case],
        "overloads": {
            "aten.add.Tensor": {
                "baseline": assignment,
                "witnessed_obligations": obligations,
                "selected_case_ids": [case_id],
                "cover": cover,
                "eqsat": eqsat,
                "candidates": {case_id: case},
            }
        },
    }
    corpus["suite_sha256"] = bounded_suite_digest(corpus)
    return corpus


def evaluate_sealed(corpus, adapter=None, **options):
    """Evaluate a sealed unit corpus on the host without the live-denominator and solver re-checks."""
    options = dict(validate_live_denominator=False, validate_solver_certificate=False) | options
    adapter = adapter or BoundedEagerSuiteAdapter()
    return evaluate_bounded_core_aten_suite(adapter, corpus, required_execution_kind="host", **options)


def test_one_call_bounded_evaluator_uses_case_ids(bounded_unit_corpus) -> None:
    corpus = bounded_unit_corpus
    case_id = corpus["selected_cases"][0]["case_id"]
    result = evaluate_sealed(corpus)
    assert result["status"] == "pass"
    assert result["status_counts"] == {"pass": 1}
    tampered = deepcopy(corpus)
    tampered_case = tampered["selected_cases"][0]
    tampered_case["eqsat_eclass"]["observational_fingerprint"] = "0" * 64
    tampered["overloads"]["aten.add.Tensor"]["candidates"][case_id] = tampered_case
    tampered["suite_sha256"] = bounded_suite_digest(tampered)
    with pytest.raises(ValueError, match="eqsat fingerprint mismatch"):
        evaluate_sealed(tampered)
    tampered = deepcopy(corpus)
    tampered["overloads"]["aten.add.Tensor"]["cover"]["minimum_cardinality_lower_bound"] = 0
    tampered["suite_sha256"] = bounded_suite_digest(tampered)
    with pytest.raises(ValueError, match="invalid stored SMT certificate"):
        evaluate_sealed(tampered)
    with pytest.raises(ValueError, match="live Core ATen denominator"):
        evaluate_sealed(corpus, validate_live_denominator=True)


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_bounded_suite_requires_explicit_observed_scope(bounded_unit_corpus, allow, mixed):
    from merlin.targetgen.core_aten_semantics import tensor_paths

    corpus = bounded_unit_corpus
    if mixed:
        # Add a second independently sealed overload, keeping one full observation.
        other = deepcopy(corpus["selected_cases"][0])
        other.update(case_document(core_aten_cases()["aten.sub.Tensor"]))
        assignment = other["partition_assignment"]
        identifier = _case_id("aten.sub.Tensor", assignment)
        other["case_id"] = identifier
        other["covered_obligations"] = sorted(_obligations("aten.sub.Tensor", assignment))
        quotient, eqsat = quotient_observational_equivalents({identifier: other})
        other = quotient[identifier]
        pool = deepcopy(corpus["overloads"]["aten.add.Tensor"])
        pool.update(
            selected_case_ids=[identifier],
            witnessed_obligations=other["covered_obligations"],
            candidates={identifier: other},
            eqsat=eqsat,
        )
        pool["cover"]["selected_capsules"] = [identifier]
        corpus["overloads"]["aten.sub.Tensor"] = pool
        corpus["selected_cases"].append(other)
        corpus["selected_cases"].sort(key=lambda c: c["case_id"])
        obligations = sorted({o for c in corpus["selected_cases"] for o in c["covered_obligations"]})
        corpus.update(
            selected_count=2,
            overload_count=2,
            witnessed_obligations=obligations,
            witnessed_obligation_count=len(obligations),
            selected_union_sha256=overload_digest(obligations),
        )
        corpus["suite_sha256"] = bounded_suite_digest(corpus)

    class ReducedAdapter:
        name = "reduced test adapter"

        def execute(self, suite):
            results = BoundedEagerSuiteAdapter().execute(suite)
            first = results[suite["selected_cases"][0]["case_id"]]
            first.update(semantic_scope="values+mutation+observed-metadata", imposed_by_harness=["requires_grad"])
            for key in ("output", "post_arguments"):
                for _, doc in tensor_paths(first[key], ""):
                    doc["requires_grad"] = None
            return results

    report = evaluate_sealed(corpus, ReducedAdapter(), allow_unobserved_autograd=allow)
    assert report["semantic_scope"] == ("values+mutation+observed-metadata" if allow else "full")
    assert report["status"] == ("pass" if allow else "fail")
    assert report["passed_count"] == (len(corpus["selected_cases"]) if allow else int(mixed))


def test_disk_backed_bounded_verdicts_use_the_same_sealed_checks(bounded_unit_corpus, tmp_path):
    from merlin.targetgen.core_aten_bounded_runner import JsonRecordMapping

    store = JsonRecordMapping(tmp_path)
    report = evaluate_sealed(bounded_unit_corpus, verdict_store=store)
    assert report["cases"] is store
    assert report["status"] == "pass"
    assert report["passed_count"] == len(store)
