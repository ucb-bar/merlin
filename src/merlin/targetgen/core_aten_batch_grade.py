"""Grade every Core ATen batch result against the canonical case output documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_DTYPES = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "bf16": np.uint16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.uint8,
}
_EXPECTED_DTYPES = {
    "f32": "float32",
    "f64": "float64",
    "f16": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "i1": "bool",
}


def numeric_policy_counts(verdicts):
    """Count executed result policies; a requested int8 policy grants no result."""
    policies = ("fp32", "int8_w8a8", "spec_int8", "native_host_exact", "original_host", "unqualified_spec_int8")
    numeric = {lane: dict.fromkeys(policies, 0) for lane in ("host", "device", "unassigned")}
    passed = {lane: dict.fromkeys(policies, 0) for lane in numeric}
    for verdict in verdicts:
        lane = verdict.get("lane", "unassigned")
        requested = verdict.get("numeric_policy", "fp32")
        policy = verdict.get("numeric_execution_policy", requested)
        if requested == "spec_int8":
            # Only the grader's declaration, exact-reference and device-evidence
            # checks can mark this execution policy as qualified.
            policy = (
                "original_host"
                if lane == "host"
                else "native_host_exact"
                if policy == "native_host_exact" and verdict.get("host_floor", {}).get("status") == "pass"
                else "spec_int8"
                if policy == "spec_int8" and verdict.get("status") == "pass"
                else "unqualified_spec_int8"
            )
        numeric.setdefault(lane, dict.fromkeys(policies, 0))
        passed.setdefault(lane, dict.fromkeys(policies, 0))
        numeric[lane][policy] = numeric[lane].get(policy, 0) + (verdict.get("numeric_status") == "pass")
        passed[lane][policy] = passed[lane].get(policy, 0) + (verdict.get("status") == "pass")
    return {
        "numeric_passed_by_lane_and_policy": numeric,
        "passed_by_lane_and_policy": passed,
        "spec_int8_passed_count": sum(counts["spec_int8"] for counts in numeric.values()),
        "spec_int8_host_fallback_passed_count": numeric["host"]["original_host"],
    }


def _decode_storage(raw: bytes, dtype: str) -> np.ndarray:
    values = np.frombuffer(raw, dtype=_DTYPES[dtype])
    if dtype == "bf16":
        return (values.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return values


def _complex_values(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get("kind") == "float":
        return float(value["value"])
    if isinstance(value, Mapping) and value.get("kind") == "complex":
        return complex(_complex_values(value["real"]), _complex_values(value["imag"]))
    if isinstance(value, list):
        return [_complex_values(item) for item in value]
    return value


def _leaves(document: Any) -> list[Any]:
    if isinstance(document, Mapping):
        if document.get("kind") == "tuple":
            return [leaf for item in document["items"] for leaf in _leaves(item)]
        if document.get("kind") in ("tensor", "float", "complex"):
            return [document]
        return [leaf for _, item in sorted(document.items()) for leaf in _leaves(item)]
    if isinstance(document, list):
        return [leaf for item in document for leaf in _leaves(item)]
    return [document]


def _storage_equal(actual, reference):
    left = np.ascontiguousarray(actual).view(np.uint8).reshape(-1, actual.dtype.itemsize)
    right = np.ascontiguousarray(reference).view(np.uint8).reshape(-1, reference.dtype.itemsize)
    return np.all(left == right, axis=1).reshape(actual.shape)


def _grade_output(raw: bytes, abi: Mapping[str, Any], expected: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    dtype = str(abi["dtype"])
    shape = tuple(int(dim) for dim in abi["shape"])
    if dtype in {"complex<f32>", "complex<f64>"}:
        complex_dtype = np.complex64 if dtype == "complex<f32>" else np.complex128
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * np.dtype(complex_dtype).itemsize
        if len(raw) != nbytes:
            return {"status": "mismatch", "reason": f"complex result has {len(raw)} bytes; expected {nbytes}"}
        actual = np.frombuffer(raw, dtype=complex_dtype).reshape(shape)
        if isinstance(expected, Mapping) and expected.get("kind") == "complex":
            if shape:
                return {"status": "mismatch", "reason": "canonical complex scalar requires a rank-zero result"}
            expected_value = expected
        else:
            if not isinstance(expected, Mapping) or expected.get("kind") != "tensor":
                return {"status": "ungradable", "reason": "complex result lacks a canonical tensor or scalar document"}
            if expected.get("dtype") != np.dtype(complex_dtype).name or tuple(expected.get("shape", ())) != shape:
                return {"status": "mismatch", "reason": "output dtype or shape differs from canonical complex result"}
            if "values" not in expected:
                if case.get("comparison") != "metadata":
                    return {"status": "ungradable", "reason": "canonical case has no numeric expected values"}
                return {
                    "status": "ungradable",
                    "reason": "metadata-only result requires compiled descriptor observations",
                }
            expected_value = expected["values"]
        try:
            reference = np.asarray(_complex_values(expected_value), dtype=complex_dtype).reshape(shape)
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "ungradable", "reason": f"cannot decode canonical expected value: {exc}"}
        parameters = case.get("comparison_parameters") or {}
        equal = (
            _storage_equal(actual, reference)
            if parameters.get("storage_bits")
            else np.isclose(
                actual,
                reference,
                rtol=float(parameters.get("rtol", 0.0)),
                atol=float(parameters.get("atol", 0.0)),
                equal_nan=bool(parameters.get("equal_nan", False)),
            )
        )
        if bool(np.all(equal)):
            return {"status": "pass", "elements": count}
        first = tuple(int(index) for index in np.argwhere(~equal)[0])
        return {
            "status": "mismatch",
            "elements": count,
            "mismatched_elements": int(np.size(equal) - np.count_nonzero(equal)),
            "first_index": list(first),
            "actual": str(np.asarray(actual[first]).item()),
            "expected": str(np.asarray(reference[first]).item()),
        }
    if dtype not in _DTYPES:
        return {"status": "ungradable", "reason": f"unsupported result ABI dtype {dtype!r}"}
    count = int(np.prod(shape)) if shape else 1
    actual = _decode_storage(raw, dtype)
    if len(actual) != count:
        return {
            "status": "mismatch",
            "reason": f"result has {len(raw)} bytes; expected {count * np.dtype(_DTYPES[dtype]).itemsize}",
        }
    actual = actual.reshape(shape)
    if isinstance(expected, Mapping) and expected.get("kind") == "tensor":
        if expected.get("dtype") != _EXPECTED_DTYPES[dtype] or tuple(expected.get("shape", ())) != shape:
            return {"status": "mismatch", "reason": "output dtype or shape differs from canonical result"}
        if "values" not in expected:
            if case.get("comparison") != "metadata":
                return {"status": "ungradable", "reason": "canonical case has no numeric expected values"}
            return {"status": "ungradable", "reason": "metadata-only result requires compiled descriptor observations"}
        expected_value = expected["values"]
    elif isinstance(expected, Mapping) and expected.get("kind") == "float":
        expected_value = float(expected["value"])
    else:
        expected_value = expected
    try:
        reference_dtype = np.float32 if dtype == "bf16" else _DTYPES[dtype]
        reference = np.asarray(_complex_values(expected_value), dtype=reference_dtype).reshape(shape)
    except (TypeError, ValueError) as exc:
        return {"status": "ungradable", "reason": f"cannot decode canonical expected value: {exc}"}
    parameters = case.get("comparison_parameters") or {}
    if dtype.startswith("f") or dtype == "bf16":
        rtol = float(parameters.get("rtol", 0.0))
        atol = float(parameters.get("atol", 0.0))
        equal = (
            _storage_equal(actual, reference)
            if parameters.get("storage_bits")
            else np.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=bool(parameters.get("equal_nan", False)))
        )
    else:
        equal = np.equal(actual, reference)
    if bool(np.all(equal)):
        return {"status": "pass", "elements": count}
    first = tuple(int(index) for index in np.argwhere(~equal)[0])
    return {
        "status": "mismatch",
        "elements": count,
        "mismatched_elements": int(np.size(equal) - np.count_nonzero(equal)),
        "first_index": list(first),
        "actual": np.asarray(actual[first]).item(),
        "expected": np.asarray(reference[first]).item(),
    }


def grade_core_aten_batch(
    batch_map: Mapping[str, Any],
    output_bytes: Sequence[bytes] | None = None,
    *,
    execution_error: str | None = None,
    output_shapes: Sequence[Sequence[int]] | None = None,
    semantic_readback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete pass/mismatch/unavailable ledger, including noncaptured cases.

    Lossless bytes alone earn numeric-only passes. Exported boundary metadata,
    pre-call snapshots and descriptor alias frames additionally judge observed semantics.
    Fields absent from the compiled ABI are excluded and explicitly reported.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    for record in batch_map["cases"]:
        overload = str(record.get("case_id") or record["overload"])
        if record["status"] != "bundled":
            verdicts[overload] = {"status": record["status"], "reason": record.get("reason", "")}
            continue
        if execution_error is not None or output_bytes is None:
            verdicts[overload] = {"status": "execution_failed", "reason": execution_error or "no hardware output"}
            continue
        indices = record["output_indices"]
        abis = record["output_abi"]
        expected = _leaves(record["case"]["expected"])
        if len(indices) != len(abis) or len(expected) != len(abis):
            verdicts[overload] = {
                "status": "ungradable",
                "reason": "canonical result structure differs from captured result ABI",
            }
            continue
        if any(index >= len(output_bytes) for index in indices):
            verdicts[overload] = {"status": "execution_failed", "reason": "hardware did not emit all result indices"}
            continue
        native_floor = None
        if (
            record.get("numeric_admission", {}).get("host_floor", {}).get("status") == "required"
            and record.get("routing", {}).get("lane") == "device"
        ):
            from merlin.targetgen.core_aten_host_floor import grade_native_bits

            native_floor = grade_native_bits(record, output_bytes, execution_error=execution_error)
        if record.get("numeric_admission", {}).get("quantized_contract"):
            reference = record["numeric_admission"].get("integer_reference", {})
            reference_bytes = reference.get("expected_output_bytes", {})
            covered = bool(reference_bytes) and all(str(i) in reference_bytes for i in indices)
            passed = covered and all(
                i < len(output_bytes) and output_bytes[i].hex() == reference_bytes[str(i)] for i in indices
            )
            native_floor = dict(
                status="pass" if passed else "mismatch",
                comparison="independent_quantized_exact_rational_bits",
                reason="quantized contract",
            )
        details = []
        for index, abi, item in zip(indices, abis, expected):
            if output_shapes is not None:
                shape = list(output_shapes[index])
                physical_shape = [*abi["shape"], 2] if abi["dtype"].startswith("complex<") else abi["shape"]
                declared = abi.get("declared_shape", physical_shape)
                if len(shape) != len(declared) or any(d >= 0 and d != s for d, s in zip(declared, shape)):
                    details.append({"status": "mismatch", "reason": "runtime shape differs from declared result"})
                    continue
                if abi["dtype"].startswith("complex<"):
                    if not shape or shape[-1] != 2:
                        details.append({"status": "mismatch", "reason": "complex result lacks component axis"})
                        continue
                    shape = shape[:-1]
                abi = {**abi, "shape": shape}
            elif any(d < 0 for d in abi.get("declared_shape", abi["shape"])):
                details.append({"status": "ungradable", "reason": "dynamic result lacks runtime extent evidence"})
                continue
            details.append(
                {"status": "pass", "comparison": native_floor["comparison"]}
                if native_floor is not None and native_floor["status"] == "pass"
                else _grade_output(output_bytes[index], abi, item, record["case"])
            )
        if all(item["status"] == "pass" for item in details):
            status = "pass"
        elif any(item["status"] == "mismatch" for item in details):
            status = "mismatch"
        else:
            status = "ungradable"
        verdicts[overload] = {
            "status": status,
            "results": details,
            "output_indices": indices,
            "semantic_scope": "numeric_only",
            "numeric_status": status,
        }
        from merlin.targetgen.core_aten_semantics import validate_full

        full = validate_full({**record, "native_floor_for_values": native_floor}, output_bytes, semantic_readback)
        if full is not None:
            verdicts[overload].update(full)
    for record in batch_map["cases"]:
        identifier = str(record.get("case_id") or record["overload"])
        verdict = verdicts[identifier]
        verdict["numeric_policy"] = record.get("numeric_policy", "fp32")
        if verdict["numeric_policy"] == "int8_w8a8":
            verdict["numeric_qualification"] = "diagnostic, not spec-conformant"
        elif verdict["numeric_policy"] == "spec_int8":
            verdict["numeric_admission"] = record.get("numeric_admission", {"status": "unknown"})
            verdict["numeric_selection"] = record.get("numeric_selection", {})
            admission = record.get("numeric_admission", {})
            if admission.get("quantized_contract"):
                verdict["precision_class"] = "b"
                verdict["quantized_contract"] = admission["quantized_contract"]
            verdict["calibration"] = admission.get(
                "calibration", {"source": "calibration-on-evaluation-input", "performed": False}
            )
            if admission.get("host_floor", {}).get("status") == "required":
                from merlin.targetgen.core_aten_host_floor import grade_native_bits

                floor = grade_native_bits(record, output_bytes, execution_error=execution_error)
                verdict["host_floor"] = {**floor, "case_id": identifier}
                if floor["status"] != "pass":
                    verdict.update(status=floor["status"], numeric_status=floor["status"], reason=floor["reason"])
            reference = admission.get("integer_reference", {})
            if reference.get("status") == "prepared" and output_bytes is not None and execution_error is None:
                comparisons = []
                for index, expected in reference["expected_output_bytes"].items():
                    i = int(index)
                    actual = output_bytes[i] if i < len(output_bytes) else b""
                    comparisons.append(
                        dict(
                            output_index=i,
                            status="pass" if actual.hex() == expected else "mismatch",
                            comparison="independent_quantized_exact_rational_bytes"
                            if admission.get("quantized_contract")
                            else "exact_i32_or_f32_bits_signed_zero_distinct",
                            actual_sha256=__import__("hashlib").sha256(actual).hexdigest(),
                        )
                    )
                exact = all(row["status"] == "pass" for row in comparisons)
                verdict["integer_reference"] = dict(
                    status="pass" if exact else "mismatch", results=comparisons, engine=reference["engine"]
                )
                verdict["numeric_status"] = "pass" if exact else "mismatch"
                if not exact:
                    verdict["status"] = "mismatch"
                    verdict["reason"] = "integer accumulator or final f32 bit-pattern mismatch"
            else:
                verdict["integer_reference"] = {k: v for k, v in reference.items() if k != "expected_output_bytes"}

        verdict["scales"] = record.get("scales", [])
        if record.get("reference_engine"):
            verdict["reference_engine"] = record["reference_engine"]
        if record.get("fp32_case") and output_bytes is not None and execution_error is None:
            from merlin.targetgen.core_aten_numeric import fp32_error

            verdict["fp32_error"] = fp32_error(record, output_bytes)
            if verdict["numeric_policy"] == "int8_w8a8":
                from merlin.targetgen.core_aten_numeric import exact_w8a8_gate

                gate = exact_w8a8_gate(record, output_bytes)
                verdict["w8a8_policy"] = record["w8a8_policy"]
                verdict["correctness_gate"] = gate
                verdict["dispatch"] = record.get("w8a8_dispatch")
                if not gate["passed"]:
                    verdict.update(
                        status="mismatch",
                        numeric_status="mismatch",
                        values_mutation_status="mismatch",
                        reason="raw i32 accumulator or final float storage bits differ from W8A8 reference",
                    )
        routing = dict(record.get("routing") or {"lane": "host", "routed": False, "reason": "scalar mode"})
        verdict.update(
            overload=record["overload"],
            routing=routing,
            routing_reason=routing.get("reason", ""),
            **{key: value for key, value in routing.items() if key != "reason"},
        )
        verdict["numeric_execution_policy"] = verdict["numeric_policy"]
        if verdict["numeric_policy"] == "spec_int8":
            host = routing.get("lane") == "host"
            verdict["numeric_execution_policy"] = "original_host" if host else "unqualified_spec_int8"
            verdict["numeric_qualification"] = (
                "original host fallback; no spec-int8 result" if host else "unqualified spec-int8 execution"
            )
        if routing.get("lane") == "device" and verdict.get("host_floor", {}).get("status") == "pass":
            verdict["numeric_execution_policy"] = "native_host_exact"
            verdict["numeric_qualification"] = "native exactness gate and complete semantic boundary checked"
            if "observation" in verdict:
                verdict["observation"]["numeric_reference"] = dict(
                    policy="native_host_exact", admission_status="admitted", gate=verdict["host_floor"]
                )
        if record.get("source_preparation"):
            verdict["source_preparation"] = record["source_preparation"]
        if routing.get("lane") == "device" and record["status"] == "bundled":
            evidence = routing.get("execution_evidence") or {}
            count = evidence.get("executed_instructions")
            from merlin.targetgen.core_aten_device import verify_execution_evidence

            if (
                routing.get("routed") is not True
                or not routing.get("routed_operations")
                or sum(case["status"] == "bundled" for case in batch_map["cases"]) != 1
                or evidence.get("case_id") != identifier
                or evidence.get("status") != "measured"
                or evidence.get("attribution") != "single_case_shard"
                or type(count) is not int
                or count < 1
                or evidence.get("target") != routing.get("target")
                or not verify_execution_evidence(evidence)
            ):
                verdict.update(
                    status="execution_failed",
                    reason=verdict.get("reason") or "device execution evidence missing or empty",
                    evidence_error="device execution evidence missing or empty",
                )
        if (
            verdict["numeric_policy"] == "spec_int8"
            and routing.get("lane") == "device"
            and verdict["status"] == "pass"
            and verdict.get("integer_reference", {}).get("status") == "pass"
            and record.get("numeric_admission", {}).get("status") == "admitted"
            and (
                record["numeric_admission"].get("quantized_contractions", 0) > 0
                or record["numeric_admission"].get("quantized_non_contraction_sites", 0) > 0
                or record["numeric_admission"].get("native_contractions", 0) > 0
            )
        ):
            # Device evidence was independently verified above; host fallback
            # retains its separate landing-fix accounting.
            verdict["numeric_execution_policy"] = "spec_int8"
            verdict["numeric_qualification"] = "declared capture-time spec-int8; exact integer/f32 gates passed"
            if record["numeric_admission"].get("native_contractions", 0):
                verdict["numeric_qualification"] = (
                    "device path conforms; host int8 narrowing not declared"
                    if record["numeric_admission"].get("host_int8_narrowing")
                    else "conforming"
                )
            if "observation" in verdict and (
                record["numeric_admission"].get("quantized_contractions", 0)
                or record["numeric_admission"].get("quantized_non_contraction_sites", 0)
            ):
                expected_case = record["case"]
                if record["numeric_admission"].get("quantized_contract"):
                    from merlin.targetgen.core_aten_semantics import full_observation

                    # The rational byte oracle owns numeric values. Capture-time
                    # eager/legacy quantized values are a different contract.
                    # Reconstruct only from independently prepared oracle bytes;
                    # the normal semantic validator still checks all metadata,
                    # aliases and mutations against the sealed source case.
                    reference_bytes = record["numeric_admission"]["integer_reference"]["expected_output_bytes"]
                    required = [*record["output_indices"], *record["semantic_boundary"]["post_indices"]]
                    if any(str(i) not in reference_bytes for i in required):
                        verdict.update(status="ungradable", reason="rational oracle lacks complete boundary bytes")
                        continue
                    oracle_raw = [bytes.fromhex(reference_bytes.get(str(i), "")) for i in range(len(output_bytes))]
                    oracle_observation = full_observation(record, oracle_raw, semantic_readback)
                    expected_case = {
                        "expected": oracle_observation["output"],
                        "post_arguments": oracle_observation["post_arguments"],
                    }
                verdict["observation"]["numeric_reference"] = {
                    "policy": "spec_int8",
                    "gate": verdict["integer_reference"],
                    "admission_status": record["numeric_admission"]["status"],
                    "expected": expected_case["expected"],
                    "post_arguments": expected_case["post_arguments"],
                }

    counts: dict[str, int] = {}
    for item in verdicts.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    from merlin.targetgen.core_aten_provenance import batch_provenance

    return {
        "provenance": batch_map.get("provenance") or batch_provenance(),
        "schema_version": 1,
        "case_count": len(verdicts),
        "passed_count": counts.get("pass", 0),
        "status_counts": dict(sorted(counts.items())),
        "cases": verdicts,
        "passed_by_lane": {
            lane: sum(v["status"] == "pass" and v["lane"] == lane for v in verdicts.values())
            for lane in ("host", "device")
        },
        "full_semantic_passed_count": sum(
            v["status"] == "pass" and v.get("semantic_scope") == "full" for v in verdicts.values()
        ),
        "observed_semantic_passed_count": sum(
            v["status"] == "pass" and v.get("semantic_scope") == "values+mutation+observed-metadata"
            for v in verdicts.values()
        ),
        "values_mutation_passed_count": sum(v.get("values_mutation_status") == "pass" for v in verdicts.values()),
        "numeric_only_passed_count": sum(
            v["status"] == "pass" and v.get("semantic_scope") == "numeric_only" for v in verdicts.values()
        ),
        "numeric_passed_count": sum(v.get("numeric_status") == "pass" for v in verdicts.values()),
        **numeric_policy_counts(verdicts.values()),
        "scope": "compiled values, mutation, memref metadata and aliases; imposed fields excluded",
    }
