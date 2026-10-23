"""One-call semantic evaluation of the complete Core ATen case corpus.

Adapters own compilation, batching, deployment, and execution.  This module owns the invariant that
every denominator overload returns an observable result and that the result matches eager PyTorch.
Decomposition and provenance never affect the verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from merlin.targetgen._aten_opset_worker import core_opset
from merlin.targetgen.core_aten_cases import (
    corpus_document,
    decode_value,
    eager_case_observation,
    resolve_overload,
)
from merlin.targetgen.core_aten_cover import overload_digest
from merlin.targetgen.core_aten_eqsat import observational_fingerprint
from merlin.targetgen.core_aten_overlay import OverlayProvider, validate_overlay


class CoreAtenSuiteAdapter(Protocol):
    """A compiler/target adapter which may batch the entire corpus into one execution."""

    name: str

    def execute(self, corpus: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
        """Execute a complete, versioned corpus and return one observation per exact overload."""


@dataclass(frozen=True)
class EagerSuiteAdapter:
    """Reference adapter used to qualify the harness itself, never as hardware evidence."""

    name: str = "torch-eager"

    def execute(self, corpus: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
        return {str(case["overload"]): eager_case_observation(case) for case in corpus["cases"]}


@dataclass(frozen=True)
class BoundedEagerSuiteAdapter:
    """Reference adapter for qualifying a bounded corpus and its multi-case result protocol."""

    name: str = "torch-eager-bounded"

    def execute(self, corpus: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
        return {str(case["case_id"]): eager_case_observation(case) for case in corpus["selected_cases"]}


@dataclass(frozen=True)
class JsonCommandSuiteAdapter:
    """Portable adapter for any compiler runner implementing the JSON request/response protocol.

    ``argv`` must contain ``{request}`` and ``{response}`` placeholders. The command is invoked once
    for the entire suite, which permits one linked image and one hardware queue lifecycle.
    """

    argv: tuple[str, ...]
    name: str
    timeout_seconds: int = 7200
    cwd: str | Path | None = None
    env: Mapping[str, str] | None = None
    receipt_verifier: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None

    def execute(self, corpus: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
        if not any("{request}" in part for part in self.argv):
            raise ValueError("adapter command must contain a {request} placeholder")
        if not any("{response}" in part for part in self.argv):
            raise ValueError("adapter command must contain a {response} placeholder")
        request = dict(corpus)
        with tempfile.TemporaryDirectory(prefix="core-aten-suite-") as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            response_path = root / "response.json"
            request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            command = [part.format(request=str(request_path), response=str(response_path)) for part in self.argv]
            environment = dict(os.environ)
            environment.update({str(key): str(value) for key, value in (self.env or {}).items()})
            process = subprocess.run(
                command,
                cwd=str(self.cwd) if self.cwd is not None else None,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if process.returncode != 0:
                detail = (process.stderr or process.stdout or "command wrote no diagnostic")[-4000:]
                raise RuntimeError(f"suite adapter exited {process.returncode}: {detail}")
            if not response_path.is_file():
                raise RuntimeError("suite adapter did not write its response document")
            response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("schema_version") != 1 or not isinstance(response.get("results"), dict):
            raise RuntimeError("suite adapter returned an invalid response document")
        return response["results"]

    def verify_hardware_receipt(self, case: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
        if self.receipt_verifier is None:
            return False
        return bool(self.receipt_verifier(case, observation))


def _without_values(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_values(item) for key, item in value.items() if key != "values"}
    return value


def _assert_document_close(expected: Any, observed: Any, *, rtol: float, atol: float) -> None:
    import torch

    if isinstance(expected, dict) and expected.get("kind") == "tensor":
        if not isinstance(observed, dict) or observed.get("kind") != "tensor":
            raise AssertionError(f"expected a tensor document, got {observed!r}")
        for field in ("dtype", "shape", "stride", "storage_offset", "requires_grad"):
            if observed.get(field) != expected.get(field):
                raise AssertionError(
                    f"tensor {field} mismatch: expected {expected.get(field)!r}, got {observed.get(field)!r}"
                )
        actual, reference = decode_value(observed), decode_value(expected)
        if reference.dtype == torch.bool:
            if not torch.equal(actual, reference):
                raise AssertionError("boolean tensor values mismatch")
        else:
            floating = reference.is_floating_point() or reference.is_complex()
            torch.testing.assert_close(
                actual, reference, rtol=rtol if floating else 0, atol=atol if floating else 0, equal_nan=True
            )
        return
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise AssertionError(f"mapping mismatch: expected keys {set(expected)}, got {observed!r}")
        for key in sorted(expected):
            _assert_document_close(expected[key], observed[key], rtol=rtol, atol=atol)
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"sequence mismatch: expected length {len(expected)}, got {observed!r}")
        for expected_item, observed_item in zip(expected, observed):
            _assert_document_close(expected_item, observed_item, rtol=rtol, atol=atol)
        return
    if isinstance(expected, float) and isinstance(observed, (float, int)):
        if math.isnan(expected) and math.isnan(float(observed)):
            return
        if not math.isclose(expected, float(observed), rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"scalar mismatch: expected {expected!r}, got {observed!r}")
        return
    if observed != expected:
        raise AssertionError(f"value mismatch: expected {expected!r}, got {observed!r}")


def _validate_observation(
    case: Mapping[str, Any], observation: Mapping[str, Any], *, allow_unobserved_autograd: bool = False
) -> None:
    required = {
        "status",
        "execution_kind",
        "output",
        "post_arguments",
        "mutated_arguments",
        "output_input_aliases",
    }
    missing = sorted(required - set(observation))
    if missing:
        raise AssertionError(f"observation is incomplete; missing {missing}")
    if observation["status"] != "executed":
        raise AssertionError(f"case was not executed: {observation['status']}")
    scope = observation.get("semantic_scope", "full")
    imposed = observation.get("imposed_by_harness", [])
    if scope == "values+mutation+observed-metadata":
        if not allow_unobserved_autograd or imposed != ["requires_grad"]:
            raise AssertionError("unsupported unobserved boundary fields")
        from merlin.targetgen.core_aten_semantics import tensor_paths

        for key in ("output", "post_arguments"):
            if any(doc.get("requires_grad") is not None for _, doc in tensor_paths(observation[key], "")):
                raise AssertionError("autograd observation disagrees with scope marker")

        def exclude_autograd(value):
            if isinstance(value, dict):
                return {
                    key: None if key == "requires_grad" and value.get("kind") == "tensor" else exclude_autograd(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [exclude_autograd(item) for item in value]
            return value

        case = {
            **case,
            "expected": exclude_autograd(case["expected"]),
            "post_arguments": exclude_autograd(case["post_arguments"]),
        }
    elif scope != "full" or imposed:
        raise AssertionError("unsupported semantic scope")

    reference = observation.get("numeric_reference")
    if reference is not None and reference.get("policy") == "native_host_exact":
        gate = reference.get("gate", {})
        if (
            reference.get("admission_status") != "admitted"
            or gate.get("status") != "pass"
            or gate.get("case_id") != case.get("case_id")
            or not gate.get("results")
            or any(row.get("status") != "pass" for row in gate["results"])
            or gate.get("comparison")
            not in {
                "exact_native_eager_storage_bits",
                "exact_rational_error_not_greater_than_native_per_component",
            }
        ):
            raise AssertionError("native dtype policy lacks its complete independent exactness gate")
        # The exact gate above owns values; sealed eager retains every metadata,
        # alias, mutation and post-argument contract. No expected value is rewritten.
        case = {**case, "comparison": "metadata"}
        reference = None
    if reference is not None:
        gate = reference.get("gate", {})
        if (
            reference.get("policy") != "spec_int8"
            or reference.get("admission_status") != "admitted"
            or gate.get("status") != "pass"
            or not gate.get("results")
            or any(row.get("status") != "pass" for row in gate["results"])
        ):
            raise AssertionError("declared numeric policy lacks a passed exact integer-reference gate")

        def policy_values(original, integer):
            # The policy changes values only. Eager still owns metadata, mutation
            # classification and storage aliases, including all post-call inputs.
            if isinstance(original, dict) and original.get("kind") == "tensor":
                if _without_values(original) != _without_values(integer):
                    raise AssertionError("numeric reference changes boundary metadata")
                return {**original, "values": integer["values"]}
            if isinstance(original, dict):
                if not isinstance(integer, dict) or original.keys() != integer.keys():
                    raise AssertionError("numeric reference changes boundary structure")
                return {key: policy_values(value, integer[key]) for key, value in original.items()}
            if isinstance(original, list):
                if not isinstance(integer, list) or len(original) != len(integer):
                    raise AssertionError("numeric reference changes boundary structure")
                return [policy_values(a, b) for a, b in zip(original, integer, strict=True)]
            return integer

        expected = reference["expected"]
        post = reference["post_arguments"]
        if scope == "values+mutation+observed-metadata":
            expected, post = exclude_autograd(expected), exclude_autograd(post)
        case = {
            **case,
            "expected": policy_values(case["expected"], expected),
            "post_arguments": policy_values(case["post_arguments"], post),
            "comparison_parameters": {"rtol": 0.0, "atol": 0.0},
        }
    parameters = case.get("comparison_parameters") or {}
    rtol = float(parameters.get("rtol", 0.0))
    atol = float(parameters.get("atol", 0.0))
    if case["comparison"] == "metadata":
        if _without_values(observation["output"]) != _without_values(case["expected"]):
            raise AssertionError("output metadata mismatch")
        if _without_values(observation["post_arguments"]) != _without_values(case["post_arguments"]):
            raise AssertionError("post-call argument metadata mismatch")
    elif case["comparison"] == "torch_close":
        _assert_document_close(case["expected"], observation["output"], rtol=rtol, atol=atol)
        _assert_document_close(case["post_arguments"], observation["post_arguments"], rtol=rtol, atol=atol)
    else:
        raise AssertionError(f"unknown comparison policy {case['comparison']!r}")
    if sorted(observation["mutated_arguments"]) != sorted(case["mutated_arguments"]):
        raise AssertionError(
            f"mutation mismatch: expected {case['mutated_arguments']!r}, got {observation['mutated_arguments']!r}"
        )
    if sorted(observation["output_input_aliases"]) != sorted(case["output_input_aliases"]):
        raise AssertionError(
            f"alias mismatch: expected {case['output_input_aliases']!r}, got {observation['output_input_aliases']!r}"
        )


def _validate_execution_evidence(
    adapter: CoreAtenSuiteAdapter,
    case: Mapping[str, Any],
    observation: Mapping[str, Any],
    required_execution_kind: str,
) -> None:
    evidence = observation.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise AssertionError("executed observation did not include a non-empty evidence mapping")
    if required_execution_kind != "hardware":
        return
    if not str(evidence.get("receipt") or "").strip():
        raise AssertionError("hardware observation did not include a non-empty evidence receipt")
    verifier = getattr(adapter, "verify_hardware_receipt", None)
    if not callable(verifier):
        raise AssertionError("hardware adapter does not implement verify_hardware_receipt(case, observation)")
    try:
        verified = bool(verifier(case, observation))
    except Exception as exc:  # noqa: BLE001 -- failed target verification must not earn a pass
        raise AssertionError(f"hardware receipt verification failed: {type(exc).__name__}: {exc}") from exc
    if not verified:
        raise AssertionError("hardware receipt verifier rejected the observation")


def evaluate_core_aten_suite(
    adapter: CoreAtenSuiteAdapter,
    *,
    required_execution_kind: str = "hardware",
    corpus: Mapping[str, Any] | None = None,
    validate_live_denominator: bool = True,
) -> dict[str, Any]:
    """Run and judge every Core ATen case with one adapter invocation.

    The returned report is successful only when every denominator overload executed using the
    requested execution kind and its final observable semantics match the eager oracle.
    """

    opset = core_opset() if corpus is None or validate_live_denominator else None
    canonical_corpus = None
    if opset is not None:
        canonical_corpus = corpus_document(opset["ops"], pytorch_version=opset["torch"])
        canonical_corpus = {**canonical_corpus, "denominator_sha256": opset["sha256"]}
    if corpus is None:
        assert canonical_corpus is not None
        corpus = canonical_corpus
    cases = list(corpus.get("cases") or ())
    expected_names = [str(case["overload"]) for case in cases]
    if (
        corpus.get("schema_version") != 1
        or not corpus.get("complete")
        or len(cases) != int(corpus.get("overload_count", -1))
        or len(cases) != int(corpus.get("case_count", -1))
    ):
        raise ValueError("refusing to evaluate an incomplete Core ATen corpus")
    if expected_names != sorted(set(expected_names)):
        raise ValueError("Core ATen corpus overloads are not sorted and unique")
    if validate_live_denominator:
        assert opset is not None
        assert canonical_corpus is not None
        if expected_names != opset["ops"]:
            raise ValueError("corpus cases do not equal the live Core ATen denominator")
        if corpus.get("pytorch_version") != opset["torch"]:
            raise ValueError("corpus PyTorch version does not equal the live pinned environment")
        if corpus.get("denominator_sha256") != opset["sha256"]:
            raise ValueError("corpus denominator digest does not equal the live Core ATen denominator")
        if cases != canonical_corpus["cases"]:
            raise ValueError("corpus cases or eager oracles do not equal the live canonical suite")

    try:
        raw_results = dict(adapter.execute(corpus))
        adapter_error = None
    except Exception as exc:  # noqa: BLE001 -- a batch crash becomes 193 explicit failed results
        raw_results = {}
        adapter_error = f"{type(exc).__name__}: {exc}"

    expected_set = set(expected_names)
    unexpected = sorted(set(raw_results) - expected_set)
    verdicts: dict[str, dict[str, Any]] = {}
    for case in cases:
        overload = str(case["overload"])
        observation = raw_results.get(overload)
        if observation is None:
            verdicts[overload] = {
                "status": "adapter_error" if adapter_error else "missing_result",
                "reason": adapter_error or "adapter returned no observation",
            }
            continue
        if not isinstance(observation, Mapping):
            verdicts[overload] = {
                "status": "invalid_result",
                "reason": f"adapter observation must be a mapping, got {type(observation).__name__}",
            }
            continue
        if observation.get("status") != "executed":
            verdicts[overload] = {
                "status": str(observation.get("status") or "invalid_result"),
                "reason": str(observation.get("reason") or "case did not execute"),
                "evidence": observation.get("evidence") or {},
                "observation": dict(observation),
            }
            continue
        if observation.get("execution_kind") != required_execution_kind:
            verdicts[overload] = {
                "status": "execution_kind_mismatch",
                "reason": (f"required {required_execution_kind!r}, observed {observation.get('execution_kind')!r}"),
                "evidence": observation.get("evidence") or {},
                "observation": dict(observation),
            }
            continue
        try:
            _validate_execution_evidence(adapter, case, observation, required_execution_kind)
            _validate_observation(case, observation)
        except Exception as exc:  # noqa: BLE001 -- comparison diagnostics belong in the ledger
            verdicts[overload] = {
                "status": "semantic_mismatch",
                "reason": f"{type(exc).__name__}: {exc}",
                "evidence": observation.get("evidence") or {},
                "observation": dict(observation),
            }
        else:
            verdicts[overload] = {
                "status": "pass",
                "evidence": observation.get("evidence") or {},
                "observation": dict(observation),
            }

    counts: dict[str, int] = {}
    for verdict in verdicts.values():
        status = verdict["status"]
        counts[status] = counts.get(status, 0) + 1
    passed = counts.get("pass", 0)
    return {
        "schema_version": 1,
        "scope": "end-to-end semantic execution of every overload tagged torch.Tag.core",
        "pytorch_version": corpus["pytorch_version"],
        "denominator_sha256": corpus.get("denominator_sha256"),
        "adapter": adapter.name,
        "required_execution_kind": required_execution_kind,
        "status": "pass" if passed == len(cases) and not unexpected else "fail",
        "case_count": len(cases),
        "passed_count": passed,
        "failed_count": len(cases) - passed,
        "unexpected_count": len(unexpected),
        "status_counts": dict(sorted(counts.items())),
        "unexpected_results": unexpected,
        "cases": verdicts,
    }


def evaluate_bounded_core_aten_suite(
    adapter: CoreAtenSuiteAdapter,
    corpus: Mapping[str, Any],
    *,
    required_execution_kind: str = "hardware",
    validate_live_denominator: bool = True,
    validate_eager_oracles: bool = True,
    validate_solver_certificate: bool = True,
    allow_unobserved_autograd: bool = False,
    verdict_store=None,
) -> dict[str, Any]:
    """Execute and judge a generated bounded suite with one adapter call.

    Unlike :func:`evaluate_core_aten_suite`, results are keyed by ``case_id`` because an overload
    intentionally appears in multiple semantic partitions. The bounded corpus is generated and
    audited separately; this function refuses incomplete or duplicate inputs and never regenerates
    them in a compiler or target environment.
    """

    from merlin.targetgen.core_aten_bounded import (
        BoundedCoverageProfile,
        _case_id,
        _obligations,
        _validate_assignment_witness,
        bounded_suite_digest,
    )
    from merlin.targetgen.core_aten_cases import decode_arguments
    from merlin.targetgen.core_aten_cover import exact_minimum_cover

    cases = list(corpus.get("selected_cases") or ())
    case_ids = [str(case.get("case_id") or "") for case in cases]
    obligations = list(corpus.get("witnessed_obligations") or ())
    if (
        corpus.get("schema_version") != 1
        or not corpus.get("complete")
        or not cases
        or len(cases) != int(corpus.get("selected_count", -1))
        or case_ids != sorted(set(case_ids))
        or obligations != sorted(set(str(item) for item in obligations))
        or len(obligations) != int(corpus.get("witnessed_obligation_count", -1))
    ):
        raise ValueError("refusing to evaluate an incomplete or non-canonical bounded Core ATen corpus")
    try:
        profile = BoundedCoverageProfile.from_dict(corpus["profile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bounded Core ATen profile is invalid: {exc}") from exc
    if corpus.get("profile_sha256") != profile.sha256:
        raise ValueError("bounded Core ATen profile digest mismatch")
    if corpus.get("suite_sha256") != bounded_suite_digest(corpus):
        raise ValueError("bounded Core ATen suite digest mismatch")
    selected_union = sorted(
        {str(obligation) for case in cases for obligation in (case.get("covered_obligations") or ())}
    )
    if selected_union != obligations or corpus.get("selected_union_sha256") != overload_digest(obligations):
        raise ValueError("bounded Core ATen selected cases do not exactly cover the declared obligations")
    pools = corpus.get("overloads")
    if not isinstance(pools, Mapping):
        raise ValueError("bounded Core ATen corpus has no per-overload candidate pools")
    generator_sha256 = corpus.get("generator_sha256")
    if (
        not isinstance(generator_sha256, str)
        or len(generator_sha256) != 64
        or any(character not in "0123456789abcdef" for character in generator_sha256)
    ):
        raise ValueError("bounded Core ATen corpus has no sealed generator digest")
    engines = corpus.get("engines")
    if not isinstance(engines, Mapping) or engines.get("eqsat_rewrite_rules") != []:
        raise ValueError("bounded Core ATen corpus has invalid reduction-engine metadata")
    global_selected = set(case_ids)
    pool_selected: set[str] = set()
    for overload, pool_value in sorted(pools.items()):
        if not isinstance(pool_value, Mapping):
            raise ValueError(f"bounded candidate pool for {overload} is not a mapping")
        pool = pool_value
        candidates = pool.get("candidates")
        selected_ids = list(pool.get("selected_case_ids") or ())
        pool_obligations = list(pool.get("witnessed_obligations") or ())
        if not isinstance(candidates, Mapping) or selected_ids != sorted(set(selected_ids)):
            raise ValueError(f"bounded candidate pool for {overload} is non-canonical")
        if any(identifier not in candidates for identifier in selected_ids):
            raise ValueError(f"bounded candidate pool for {overload} names a missing selected case")
        stored_cover = pool.get("cover")
        if (
            not isinstance(stored_cover, Mapping)
            or stored_cover.get("status") != "optimal"
            or stored_cover.get("selected_capsules") != selected_ids
            or stored_cover.get("objective_value") != len(selected_ids)
            or stored_cover.get("minimum_cardinality_lower_bound") != len(selected_ids)
            or stored_cover.get("minimum_cardinality_upper_bound") != len(selected_ids)
            or not stored_cover.get("objective_bounds_closed")
        ):
            raise ValueError(f"bounded candidate pool for {overload} has an invalid stored SMT certificate")
        eqsat = pool.get("eqsat")
        if (
            stored_cover.get("solver") != engines.get("set_cover")
            or stored_cover.get("solver_version") != engines.get("set_cover_version")
            or not isinstance(eqsat, Mapping)
            or eqsat.get("xdsl_version") != engines.get("observational_quotient_version")
            or eqsat.get("rewrite_rules") != []
        ):
            raise ValueError(f"bounded candidate pool for {overload} disagrees with reduction engines")
        pool_selected.update(selected_ids)
        if validate_solver_certificate:
            result = exact_minimum_cover(
                pool_obligations,
                {
                    str(identifier): list(candidate.get("covered_obligations") or ())
                    for identifier, candidate in candidates.items()
                },
            )
            if result.status != "optimal" or result.selected_capsules != selected_ids:
                raise ValueError(f"bounded candidate pool for {overload} failed exact cover revalidation")
            if (
                not result.objective_bounds_closed
                or result.minimum_cardinality_lower_bound != result.objective_value
                or result.minimum_cardinality_upper_bound != result.objective_value
            ):
                raise ValueError(f"bounded candidate pool for {overload} failed objective-bound revalidation")
    if pool_selected != global_selected:
        raise ValueError("bounded global selected cases do not equal the per-overload selected sets")
    for case in cases:
        overload = str(case.get("overload"))
        assignment = case.get("partition_assignment")
        if not isinstance(assignment, Mapping):
            raise ValueError(f"bounded Core ATen case has no partition assignment: {case.get('case_id')}")
        normalized_assignment = {str(key): str(value) for key, value in sorted(assignment.items())}
        raw_assignments = case.get("equivalent_partition_assignments") or [assignment]
        if not isinstance(raw_assignments, list) or not raw_assignments:
            raise ValueError(f"bounded Core ATen case has invalid eqsat assignments: {case.get('case_id')}")
        equivalent_assignments = [
            {str(key): str(value) for key, value in sorted(item.items())}
            for item in raw_assignments
            if isinstance(item, Mapping)
        ]
        if len(equivalent_assignments) != len(raw_assignments):
            raise ValueError(f"bounded Core ATen case has malformed eqsat assignments: {case.get('case_id')}")
        if normalized_assignment not in equivalent_assignments:
            raise ValueError("bounded Core ATen primary assignment is absent from its eqsat class")
        assignment_keys = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in equivalent_assignments]
        if assignment_keys != sorted(set(assignment_keys)):
            raise ValueError(f"bounded Core ATen eqsat assignments are non-canonical: {case.get('case_id')}")
        eclass = case.get("eqsat_eclass")
        if not isinstance(eclass, Mapping):
            raise ValueError(f"bounded Core ATen case has no eqsat certificate: {case.get('case_id')}")
        members = list(eclass.get("members") or ())
        if members != sorted(set(str(member) for member in members)) or not members:
            raise ValueError(f"bounded Core ATen eqsat members are non-canonical: {case.get('case_id')}")
        if eclass.get("representative") != case.get("case_id") or case.get("case_id") != min(members):
            raise ValueError(f"bounded Core ATen eqsat representative is invalid: {case.get('case_id')}")
        if len(members) != len(equivalent_assignments):
            raise ValueError(f"bounded Core ATen eqsat member/assignment counts differ: {case.get('case_id')}")
        if eclass.get("observational_fingerprint") != observational_fingerprint(case):
            raise ValueError(f"bounded Core ATen eqsat fingerprint mismatch: {case.get('case_id')}")
        pool = pools.get(overload)
        if not isinstance(pool, Mapping):
            raise ValueError(f"bounded Core ATen case has no candidate pool: {case.get('case_id')}")
        if case.get("case_id") != _case_id(overload, normalized_assignment):
            raise ValueError(f"bounded Core ATen case ID mismatch: {case.get('case_id')}")
        expected_obligations = sorted(
            {
                obligation
                for equivalent_assignment in equivalent_assignments
                for obligation in _obligations(overload, equivalent_assignment)
            }
        )
        if list(case.get("covered_obligations") or ()) != expected_obligations:
            raise ValueError(f"bounded Core ATen obligation labels are invalid: {case.get('case_id')}")
        if pool.get("candidates", {}).get(case["case_id"]) != case:
            raise ValueError(f"bounded selected case differs from its candidate-pool record: {case.get('case_id')}")
        args, kwargs = decode_arguments(case["arguments"])
        for equivalent_assignment in equivalent_assignments:
            _validate_assignment_witness(overload, args, kwargs, equivalent_assignment, pool["baseline"])
        expected_bytes = (
            json.dumps(case.get("expected"), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if case.get("expected_sha256") != hashlib.sha256(expected_bytes).hexdigest():
            raise ValueError(f"bounded Core ATen oracle digest mismatch for {case.get('case_id')}")
        if case.get("schema") != str(resolve_overload(str(case.get("overload")))._schema):
            raise ValueError(f"bounded Core ATen schema mismatch for {case.get('case_id')}")
        if validate_eager_oracles:
            _validate_observation(case, eager_case_observation(case))
    if validate_live_denominator:
        opset = core_opset()
        observed_overloads = sorted({str(case["overload"]) for case in cases})
        if observed_overloads != opset["ops"]:
            raise ValueError("bounded suite cases do not span the live Core ATen denominator")
        if corpus.get("overload_count") != len(opset["ops"]):
            raise ValueError("bounded suite overload count does not equal the live Core ATen denominator")
        if corpus.get("denominator_sha256") != opset["sha256"]:
            raise ValueError("bounded suite denominator digest does not equal the live Core ATen denominator")
        if corpus.get("pytorch_version") != opset["torch"]:
            raise ValueError("bounded suite PyTorch version does not equal the live pinned environment")
    try:
        raw_results = adapter.execute(corpus)
        if not isinstance(raw_results, Mapping):
            raise TypeError("adapter results must be a mapping")
        adapter_error = None
    except Exception as exc:  # noqa: BLE001 -- one batch failure becomes explicit per-case failures
        raw_results = {}
        adapter_error = f"{type(exc).__name__}: {exc}"
    expected = set(case_ids)
    unexpected = sorted(set(raw_results) - expected)
    verdicts = {} if verdict_store is None else verdict_store
    for case in cases:
        identifier = str(case["case_id"])
        observation = raw_results.get(identifier)
        if observation is None:
            verdicts[identifier] = {
                "status": "adapter_error" if adapter_error else "missing_result",
                "reason": adapter_error or "adapter returned no observation",
                "overload": case["overload"],
            }
            continue
        if not isinstance(observation, Mapping):
            verdicts[identifier] = {
                "status": "invalid_result",
                "reason": f"adapter observation must be a mapping, got {type(observation).__name__}",
                "overload": case["overload"],
            }
            continue
        if observation.get("status") != "executed":
            verdicts[identifier] = {
                "status": str(observation.get("status") or "invalid_result"),
                "reason": str(observation.get("reason") or "case did not execute"),
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
                "semantic_scope": observation.get("semantic_scope", "full"),
                "imposed_by_harness": observation.get("imposed_by_harness", []),
            }
            continue
        if observation.get("execution_kind") != required_execution_kind:
            verdicts[identifier] = {
                "status": "execution_kind_mismatch",
                "reason": f"required {required_execution_kind!r}, observed {observation.get('execution_kind')!r}",
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
                "semantic_scope": observation.get("semantic_scope", "full"),
                "imposed_by_harness": observation.get("imposed_by_harness", []),
            }
            continue
        try:
            _validate_execution_evidence(adapter, case, observation, required_execution_kind)
            _validate_observation(case, observation, allow_unobserved_autograd=allow_unobserved_autograd)
        except Exception as exc:  # noqa: BLE001 -- comparisons belong in the per-case ledger
            verdicts[identifier] = {
                "status": "semantic_mismatch",
                "reason": f"{type(exc).__name__}: {exc}",
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
                "semantic_scope": observation.get("semantic_scope", "full"),
                "imposed_by_harness": observation.get("imposed_by_harness", []),
            }
        else:
            verdicts[identifier] = {
                "status": "pass",
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
                "semantic_scope": observation.get("semantic_scope", "full"),
                "imposed_by_harness": observation.get("imposed_by_harness", []),
            }
    counts: dict[str, int] = {}
    for verdict in verdicts.values():
        counts[verdict["status"]] = counts.get(verdict["status"], 0) + 1
    passed = counts.get("pass", 0)
    claimed_scope = "values+mutation+observed-metadata" if allow_unobserved_autograd else "full"
    scope_complete = all(
        v.get("semantic_scope") == "full" or (allow_unobserved_autograd and v.get("semantic_scope") == claimed_scope)
        for v in verdicts.values()
    )
    return {
        "schema_version": 1,
        "semantic_scope": claimed_scope,
        "scope": corpus["scope"],
        "pytorch_version": corpus["pytorch_version"],
        "profile_sha256": corpus["profile_sha256"],
        "adapter": adapter.name,
        "required_execution_kind": required_execution_kind,
        "status": "pass" if passed == len(cases) and not unexpected and scope_complete else "fail",
        "case_count": len(cases),
        "passed_count": passed,
        "failed_count": len(cases) - passed,
        "unexpected_count": len(unexpected),
        "status_counts": dict(sorted(counts.items())),
        "unexpected_results": unexpected,
        "cases": verdicts,
    }


def evaluate_bounded_core_aten_with_overlays(
    adapter: CoreAtenSuiteAdapter,
    portable_corpus: Mapping[str, Any],
    *,
    additive_overlays: tuple[Mapping[str, Any], ...] = (),
    overlay_providers: Mapping[str, OverlayProvider] | None = None,
    required_execution_kind: str = "hardware",
    validate_live_denominator: bool = True,
    validate_eager_oracles: bool = True,
    validate_solver_certificate: bool = True,
) -> dict[str, Any]:
    """Validate the portable suite and additive overlays, then submit their union once.

    The external adapter is invoked exactly once with every selected portable and overlay case.  The
    eager validation call below is local harness qualification, not compiler or target execution.
    Overlay obligations are never merged with the portable set-cover problem, so an overlay cannot
    replace a portable case.
    """

    portable_validation = evaluate_bounded_core_aten_suite(
        BoundedEagerSuiteAdapter(),
        portable_corpus,
        required_execution_kind="host",
        validate_live_denominator=validate_live_denominator,
        validate_eager_oracles=validate_eager_oracles,
        validate_solver_certificate=validate_solver_certificate,
    )
    if portable_validation["status"] != "pass":
        raise ValueError("portable bounded Core ATen corpus failed local qualification")

    overlay_cases: list[dict[str, Any]] = []
    overlay_metadata = []
    for overlay in additive_overlays:
        profile = overlay.get("profile")
        provider = (overlay_providers or {}).get(str(profile.get("name"))) if isinstance(profile, Mapping) else None
        if provider is None:
            raise ValueError(f"unsupported additive overlay profile: {profile!r}")
        cases = validate_overlay(
            provider,
            overlay,
            validate_eager_oracles=validate_eager_oracles,
            validate_solver_certificate=validate_solver_certificate,
        )
        if overlay.get("pytorch_version") != portable_corpus.get("pytorch_version"):
            raise ValueError("additive overlay PyTorch version differs from the portable corpus")
        overlay_cases.extend(cases)
        overlay_metadata.append(
            {
                "name": profile["name"],
                "profile_sha256": overlay["profile_sha256"],
                "overlay_sha256": overlay["overlay_sha256"],
                "selected_count": overlay["selected_count"],
                "obligation_count": overlay["obligation_count"],
            }
        )

    portable_cases = list(portable_corpus["selected_cases"])
    cases = sorted([*portable_cases, *overlay_cases], key=lambda case: str(case["case_id"]))
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("portable and overlay suites contain duplicate case IDs")
    request = {
        "schema_version": 1,
        "scope": "portable bounded Core ATen suite plus independently additive hardware overlays",
        "pytorch_version": portable_corpus["pytorch_version"],
        "portable_suite_sha256": portable_corpus["suite_sha256"],
        "portable_selected_count": len(portable_cases),
        "additive_overlays": overlay_metadata,
        "overlay_selected_count": len(overlay_cases),
        "selected_count": len(cases),
        "selected_cases": cases,
    }
    try:
        raw_results = dict(adapter.execute(request))
        adapter_error = None
    except Exception as exc:  # noqa: BLE001 -- a batch failure must remain visible for every case
        raw_results = {}
        adapter_error = f"{type(exc).__name__}: {exc}"
    expected = set(case_ids)
    unexpected = sorted(set(raw_results) - expected)
    verdicts: dict[str, dict[str, Any]] = {}
    for case in cases:
        identifier = str(case["case_id"])
        observation = raw_results.get(identifier)
        if observation is None:
            verdicts[identifier] = {
                "status": "adapter_error" if adapter_error else "missing_result",
                "reason": adapter_error or "adapter returned no observation",
                "overload": case["overload"],
            }
            continue
        if not isinstance(observation, Mapping):
            verdicts[identifier] = {
                "status": "invalid_result",
                "reason": f"adapter observation must be a mapping, got {type(observation).__name__}",
                "overload": case["overload"],
            }
            continue
        if observation.get("status") != "executed":
            verdicts[identifier] = {
                "status": str(observation.get("status") or "invalid_result"),
                "reason": str(observation.get("reason") or "case did not execute"),
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
            }
            continue
        if observation.get("execution_kind") != required_execution_kind:
            verdicts[identifier] = {
                "status": "execution_kind_mismatch",
                "reason": f"required {required_execution_kind!r}, observed {observation.get('execution_kind')!r}",
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
            }
            continue
        try:
            _validate_execution_evidence(adapter, case, observation, required_execution_kind)
            _validate_observation(case, observation)
        except Exception as exc:  # noqa: BLE001 -- retain one verdict for every requested case
            verdicts[identifier] = {
                "status": "semantic_mismatch",
                "reason": f"{type(exc).__name__}: {exc}",
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
            }
        else:
            verdicts[identifier] = {
                "status": "pass",
                "overload": case["overload"],
                "evidence": observation.get("evidence") or {},
            }
    counts: dict[str, int] = {}
    for verdict in verdicts.values():
        counts[verdict["status"]] = counts.get(verdict["status"], 0) + 1
    passed = counts.get("pass", 0)
    return {
        "schema_version": 1,
        "scope": request["scope"],
        "pytorch_version": request["pytorch_version"],
        "adapter": adapter.name,
        "required_execution_kind": required_execution_kind,
        "status": "pass" if passed == len(cases) and not unexpected else "fail",
        "case_count": len(cases),
        "portable_case_count": len(portable_cases),
        "overlay_case_count": len(overlay_cases),
        "passed_count": passed,
        "failed_count": len(cases) - passed,
        "unexpected_count": len(unexpected),
        "status_counts": dict(sorted(counts.items())),
        "unexpected_results": unexpected,
        "cases": verdicts,
    }
