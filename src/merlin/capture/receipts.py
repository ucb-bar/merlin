"""Admit realized integer/PT2E arithmetic and its original complete source receipts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from merlin.capture.artifacts import M2MUnavailable
from merlin.common.digest import is_sha256

# ---- numerics ----------------------------------------------------------------------------------
# Original captured-output numerical policies and explicit precision recipe projection.

_DEF_ATOL, _DEF_RTOL = 0.03125, 0.02


_COMPILE_DTYPE = {
    "int8": "int8",
    "i8": "int8",
    "fp8_e4m3": "fp8",
    "fp8": "fp8",
    "f32": "fp32",
    "fp32": "fp32",
    "fp16": "fp16",
    "bf16": "fp32",
}


_ACTIVATION_QUANTIZING_SCHEME = {
    "int8": "int8_dyn_act_int8_weight",
    "i8": "int8_dyn_act_int8_weight",
    "fp8_e4m3": "float8_dyn_act_float8_weight",
    "mxfp4": "mx_dyn_act_mx_weight_mx4",
    "mxfp6": "mx_dyn_act_mx_weight_mx6",
    "nvfp4": "nvfp4_dyn_act_nvfp4_weight",
}


_UNQUANTIZED_FORMATS = frozenset({"fp32", "f32", "bf16", "fp16", "f16"})


def _tol(binding) -> tuple[float, float]:
    atol = binding.atol if binding.atol is not None else _DEF_ATOL
    rtol = binding.rtol if binding.rtol is not None else _DEF_RTOL
    return atol, rtol


def _model_output_numeric_policy(output_abi: list[dict], golden_values: list, binding, *, output_names=None) -> dict:
    """Grade in the captured result's element type, not the target's operand type.

    The capture worker serializes even integer tensors through float64 JSON. An
    integer policy is sound only where every serialized golden remains finite,
    integral and exactly representable; the ABI supplies the actual tensor dtype.
    Heterogeneous result dtypes require an explicit complete output roster and
    receive one existing numerical policy per result; legacy calls refuse them.
    """
    if len(output_abi) != len(golden_values):
        raise M2MUnavailable("model output ABI/result cardinality disagrees with its golden")
    dtypes = {str(result.get("dtype") or "") for result in output_abi}
    if len(dtypes) != 1 and output_names is not None:
        if len(output_names) != len(output_abi) or len(set(output_names)) != len(output_names):
            raise M2MUnavailable("typed model output numerical policy needs every unique original result")
        return {
            "schema": "merlin.numeric_policy.per_output.v1",
            "compare": "per_output",
            "outputs": {
                name: _model_output_numeric_policy([abi], [value], binding)
                for name, abi, value in zip(output_names, output_abi, golden_values, strict=True)
            },
        }
    if len(dtypes) != 1:
        raise M2MUnavailable(
            f"model results have different output dtypes {sorted(dtypes)}; one numeric policy cannot describe them"
        )
    (dtype,) = dtypes

    def scalars(value):
        if isinstance(value, (list, tuple)):
            for child in value:
                yield from scalars(child)
        else:
            yield value

    if dtype.startswith("ui") and dtype[2:].isdigit():
        bits = int(dtype[2:])
        if bits < 1:
            raise M2MUnavailable(f"captured model output dtype {dtype!r} has invalid bit width")
        low, high = 0, (1 << bits) - 1
    elif dtype.startswith("i") and dtype[1:].isdigit():
        bits = int(dtype[1:])
        if bits < 1:
            raise M2MUnavailable(f"captured model output dtype {dtype!r} has invalid bit width")
        low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        bits = None
    if bits is not None:
        for value in (v for result in golden_values for v in scalars(result)):
            if (isinstance(value, bool) and bits != 1) or not isinstance(value, (int, float)):
                raise M2MUnavailable(f"captured {dtype} result has a non-integer golden value {value!r}")
            if not math.isfinite(value) or value != int(value) or not low <= value <= high or abs(value) > 1 << 53:
                raise M2MUnavailable(f"captured {dtype} result has an inexact/out-of-range golden value {value!r}")
        return {"compare": "exact_int", "dtype": dtype}
    if dtype in {"f16", "bf16", "f32", "f64"} or dtype.startswith("f8"):
        return {"compare": "tolerance_float", "dtype": dtype, "atol": _tol(binding)[0], "rtol": _tol(binding)[1]}
    raise M2MUnavailable(f"captured model output dtype {dtype!r} has no supported numeric comparison policy")


def _flatten(nested) -> list:
    out: list = []
    if not (isinstance(nested, list) and nested and isinstance(nested[0], list)):
        return list(nested)
    for row in nested:
        out.extend(_flatten(row) if isinstance(row[0], list) else row)
    return out


def compile_dtype(token: str) -> str:
    return _COMPILE_DTYPE.get(token, "fp32")


def activation_quantizing_scheme(fmt: str) -> str | None:
    """The capture scheme a whole-model capsule needs at ``fmt``, or ``None`` when it needs none.

    ``None`` means the format is one the model already runs in (a native float), so there is nothing to
    quantize and the default capture is already the right program.

    Raises for a format that IS quantized but whose activation-quantizing scheme is unknown here.
    Falling back to the weight-only default would be the silent version of the bug this exists to
    prevent: the capture would succeed, contain a float matmul, and certify arithmetic the target does
    not perform.
    """
    tok = str(fmt)
    if tok in _UNQUANTIZED_FORMATS:
        return None
    scheme = _ACTIVATION_QUANTIZING_SCHEME.get(tok)
    if scheme is None:
        raise ValueError(
            f"no activation-quantizing capture scheme is known for format {tok!r}; the weight-only "
            f"default would capture a float matmul over dequantized weights, which is not the "
            f"arithmetic a datapath in this format performs. Add the scheme whose catalog entry "
            f"declares {tok!r} as BOTH its weight_dtype and its activation_dtype"
        )
    return scheme


def derived_recipe(target: str | None, dtype: str) -> dict | None:
    """The target's derived quantization recipe for ``dtype``, or ``None`` when there is none to use.

    ``None`` leaves the caller on its named scheme. That happens when no recipe derives (the reason is
    in the recipe's own ``underivable`` and is reported by the capability-drift gate, not repeated
    here) or when the derived recipe is for a different element format than the capsule asks for.
    """
    if not target:
        return None
    try:
        from merlin.common import target_data as _qf
        from merlin.compiler.quantization import capabilities as _qr

        recipe = _qr.select(_qr.for_target(target))
        if recipe is None or recipe.status != _qr.DERIVED:
            return None
        wanted = _qf.get(str(dtype)).name if _qf.has(str(dtype)) else str(dtype)
        return recipe.to_dict() if recipe.weight.dtype == wanted else None
    except Exception:  # noqa: BLE001 -- a derivation fault leaves the named scheme in force
        return None


# ---- receipts ----------------------------------------------------------------------------------


def _require_integer_contraction(program: str, *, scheme: str | None, recipe: dict | None) -> None:
    """A W8A8 declaration is not evidence of an integer datapath.

    TorchAO may leave an ineligible Linear unchanged (for example, tiny dimensions)
    while model2MLIR still labels the module with the requested scheme. Inspect the
    parsed arithmetic before admitting or caching that capture as an integer test.
    """
    integer_recipe = recipe is not None and all(
        (recipe.get(part) or {}).get("dtype") in {"int8", "i8"} for part in ("activation", "weight")
    )
    if scheme not in {"int8_dyn_act_int8_weight", "int8_static_act_int8_weight"} and not integer_recipe:
        return
    from merlin.compiler.ir import mlir_query as query

    try:
        module = query.parse(program)
        forward = next(
            (
                op
                for op in query.walk(module, "func.func")
                if getattr(getattr(op, "sym_name", None), "data", None) == "forward"
            ),
            None,
        )
        if forward is None:
            raise ValueError("no @forward function")
        for op in query.walk(forward, "linalg.matmul", "linalg.generic"):
            if query.op_name(op) == "linalg.generic" and query.attr_str(op, "prov.op") != "int_matmul":
                continue
            if len(op.operands) < 2 or not op.results:
                continue
            lhs, rhs = (query.type_shape_dtype(value.type)[1] for value in op.operands[:2])
            result = query.type_shape_dtype(op.results[0].type)[1]
            if (lhs, rhs, result) == ("i8", "i8", "i32"):
                return
    except Exception as exc:
        raise M2MUnavailable(f"cannot verify W8A8 capture arithmetic: {exc}") from exc
    raise M2MUnavailable(
        "W8A8 capture has no i8×i8→i32 integer contraction; "
        "the requested torchao scheme may have skipped this model or shape"
    )


def _static_pt2e_model(op: str, *, scheme: str | None, recipe: dict | None, already_quantized: bool) -> bool:
    """Only converted static PT2E models owe the full integerization receipt.

    The dynamic TorchAO and already-materialized graph routes have different
    transformations; in particular an authored integer graph must not be
    described as a PT2E rewrite.
    """
    return (
        op == "model"
        and not already_quantized
        and (
            scheme == "int8_static_act_int8_weight"
            or (
                recipe is not None
                and (recipe.get("activation") or {}).get("mode") == "static"
                and all((recipe.get(part) or {}).get("dtype") in {"int8", "i8"} for part in ("activation", "weight"))
            )
        )
    )


def _require_pt2e_integerization_receipt(
    program: str,
    meta: dict,
    *,
    agreement_tolerance: tuple[float, float] | None,
    capture_root: Path | None = None,
) -> None:
    """Admit a static W8A8 model only if its whole PT2E region was integerized.

    Finding *one* integer matmul in the MLIR proves only an existential fact.
    This receipt additionally binds complete Q/DQ-fed linear/conv/matmul census,
    safe accumulation and a selected numerical comparison to the captured
    program. Missing fields are never treated as zero or as success.
    """
    receipt = meta.get("integerization_receipt")
    if isinstance(receipt, dict) and receipt.get("schema") == "merlin.dynamic_integer_reference.v1":
        from merlin.development.targetgen.applications.inventory import (
            verified_static_integerization,
            verify_capture_receipt,
        )

        if capture_root is None:
            raise M2MUnavailable("dynamic independent reference needs its complete capture directory")
        actual = capture_root / "model.mlir"
        if not actual.is_file() or actual.read_text() != program:
            raise M2MUnavailable("dynamic independent reference program differs from its capture")
        observed = verify_capture_receipt(actual)
        projection = observed.get("capture_integerization")
        if (
            observed.get("status") != "verified_materialized"
            or not isinstance(projection, dict)
            or projection.get("integerization_receipt") != receipt
            or not verified_static_integerization(projection, receipt_sha256=observed.get("receipt_sha256"))
        ):
            raise M2MUnavailable("dynamic independent reference lacks complete bound source/output/census proof")
        return
    if not isinstance(receipt, dict) or receipt.get("schema") != "m2m.pt2e-integerize.v1":
        raise M2MUnavailable("static W8A8 model has no supported integerization receipt")

    def count(value: object, name: str) -> int:
        if type(value) is not int or value < 0:
            raise M2MUnavailable(f"integerization receipt has invalid {name}")
        return value

    seen = count(receipt.get("quantized_contractions_seen"), "quantized contraction census")
    rewritten = count(receipt.get("quantized_contractions_integerized"), "integerized contraction census")
    remaining = count(receipt.get("quantized_contractions_remaining"), "remaining contraction census")
    quantization_stats = meta.get("quantization_stats")
    annotated = count(
        quantization_stats.get("annotated_contractions") if isinstance(quantization_stats, dict) else None,
        "PT2E annotated contraction census",
    )
    if annotated != seen:
        raise M2MUnavailable(
            f"integerization receipt census {seen} disagrees with PT2E's {annotated} annotated contractions"
        )
    if seen < 1 or seen != rewritten + remaining:
        raise M2MUnavailable("integerization receipt has inconsistent quantized contraction census")
    if remaining != 0 or rewritten != seen:
        raise M2MUnavailable(f"integerization receipt has {remaining} quantized contractions remaining")
    by_kind = receipt.get("quantized_by_kind")
    if not isinstance(by_kind, dict) or set(by_kind) != {"linear", "conv2d", "matmul", "unsupported"}:
        raise M2MUnavailable("integerization receipt lacks complete linear/conv2d/matmul/unsupported census")
    kind_totals = [0, 0, 0]
    for kind in ("linear", "conv2d", "matmul", "unsupported"):
        row = by_kind[kind]
        if not isinstance(row, dict):
            raise M2MUnavailable(f"integerization receipt has invalid {kind} census")
        values = [count(row.get(key), f"{kind} {key} census") for key in ("seen", "integerized", "remaining")]
        if values[0] != values[1] + values[2]:
            raise M2MUnavailable(f"integerization receipt has inconsistent {kind} census")
        kind_totals = [total + value for total, value in zip(kind_totals, values)]
    if kind_totals != [seen, rewritten, remaining]:
        raise M2MUnavailable("integerization receipt per-kind census disagrees with its total")
    if by_kind["unsupported"]["seen"]:
        raise M2MUnavailable("integerization receipt contains unsupported quantized contractions")
    refusals = receipt.get("refusals")
    if not isinstance(refusals, list) or refusals:
        raise M2MUnavailable(f"integerization receipt reports refusal(s): {refusals!r}")
    # Worst case signed-int8 product bound; compare using the *same* arithmetic
    # as the upstream integerizer, without assuming a particular accelerator.
    max_k = count(receipt.get("max_reduction_k"), "accumulator reduction K")
    if receipt.get("accumulator_bound_checked") is not True or max_k < 1 or max_k * 128 * 128 > (1 << 31) - 1:
        raise M2MUnavailable("integerization receipt does not prove a safe i32 accumulator")

    engine = (meta.get("recipe") or {}).get("software_numerical_engine")
    if engine not in (None, "integer_reference"):
        raise M2MUnavailable(f"unsupported static W8A8 numerical engine {engine!r}")
    expected = (0.0, 0.0) if engine == "integer_reference" else agreement_tolerance or (_DEF_ATOL, _DEF_RTOL)
    agreement = receipt.get("golden_agreement")
    if not isinstance(agreement, dict) or agreement.get("status") != "passed":
        raise M2MUnavailable("integerization receipt has no passed original-vs-rewrite golden agreement")
    if any(agreement.get(name) != tolerance for name, tolerance in zip(("atol", "rtol"), expected)):
        raise M2MUnavailable("integerization receipt golden agreement tolerance differs from the request")
    if count(agreement.get("samples"), "golden agreement samples") < 1:
        raise M2MUnavailable("integerization receipt has no golden agreement samples")
    outputs = agreement.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise M2MUnavailable("integerization receipt has no golden agreement outputs")
    for result in [agreement, *outputs]:
        if not isinstance(result, dict):
            raise M2MUnavailable("integerization receipt has invalid golden agreement output")
        for metric in ("max_abs", "max_rel"):
            value = result.get(metric)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise M2MUnavailable(f"integerization receipt has invalid golden agreement {metric}")
    if any(row.get("within_tolerance") is not True for row in outputs):
        raise M2MUnavailable("integerization receipt has a golden agreement output outside tolerance")
    if any(row.get("atol") != expected[0] or row.get("rtol") != expected[1] for row in outputs):
        raise M2MUnavailable("integerization receipt has a golden agreement output with different tolerance")
    if engine == "integer_reference":
        if agreement.get("reference") != "pt2e_integer":
            raise M2MUnavailable("integerization receipt did not use the selected independent integer reference")
        source = agreement.get("source")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not is_sha256(source.get("sha256"))
        ):
            raise M2MUnavailable("integerization receipt lacks the integer reference source identity")
        executed = agreement.get("executed_contractions")
        if (
            not isinstance(executed, dict)
            or any(
                type(executed.get(kind)) is not int or executed[kind] != by_kind[kind]["seen"]
                for kind in ("conv2d", "linear", "matmul")
            )
            or any(executed.get(key) != seen for key in ("total", "selected", "observed"))
        ):
            raise M2MUnavailable("independent integer reference contraction census differs from PT2E")
        pointer = agreement.get("output")
        if (
            not isinstance(pointer, dict)
            or pointer.get("path") != "integer-reference.json"
            or not is_sha256(pointer.get("sha256"))
        ):
            raise M2MUnavailable("integerization receipt lacks a bundle-local independent reference artifact")
        if capture_root is None:
            raise M2MUnavailable("independent integer reference artifact has no capture directory")
        reference_path = capture_root / "integer-reference.json"
        if (
            not reference_path.is_file()
            or reference_path.is_symlink()
            or hashlib.sha256(reference_path.read_bytes()).hexdigest() != pointer["sha256"]
        ):
            raise M2MUnavailable("independent integer reference artifact does not match its digest")

    emitted = count(receipt.get("integer_mm_emitted"), "emitted integer matmul count")
    if emitted < rewritten:
        raise M2MUnavailable("integerization receipt emitted fewer integer matmuls than rewritten regions")
    from merlin.compiler.ir import mlir_query as query

    try:
        module = query.parse(program)
        forward = next(
            (
                op
                for op in query.walk(module, "func.func")
                if getattr(getattr(op, "sym_name", None), "data", None) == "forward"
            ),
            None,
        )
        if forward is None:
            raise ValueError("no @forward function")
        structural = 0
        for op in query.walk(forward, "linalg.matmul", "linalg.generic"):
            if query.op_name(op) == "linalg.generic" and query.attr_str(op, "prov.op") != "int_matmul":
                continue
            if len(op.operands) < 2 or not op.results:
                continue
            lhs, rhs = (query.type_shape_dtype(value.type)[1] for value in op.operands[:2])
            result = query.type_shape_dtype(op.results[0].type)[1]
            structural += (lhs, rhs, result) == ("i8", "i8", "i32")
    except Exception as exc:
        raise M2MUnavailable(f"cannot verify integerized model export: {exc}") from exc
    if structural < emitted:
        raise M2MUnavailable(
            f"integerization receipt claims {emitted} integer matmuls, but export contains only {structural}"
        )
    reported_export = count(receipt.get("exported_integer_mm_count"), "exported integer matmul count")
    if reported_export != structural:
        raise M2MUnavailable(
            f"integerization receipt export count {reported_export} disagrees with parsed program {structural}"
        )


def _require_materialized_contraction(program: str, *, dtype: str) -> int:
    """Prove a loader marked already materialized really carries its declared operands.

    This mode skips framework quantization for graphs which already encode their
    numeric operations. It must still fail when the importer drops those operations
    or leaves only host float contractions. The element spelling comes from the
    shared numeric format registry, not a target name.
    """
    from merlin.compiler.ir import mlir_query as query
    from merlin.development.targetgen.corpus.binding import dtype_info

    try:
        _, element, _, integer = dtype_info(dtype)
        if not integer:
            raise ValueError(f"{dtype!r} is not an integer operand format")
        module = query.parse(program)
        forward = next(
            (
                op
                for op in query.walk(module, "func.func")
                if getattr(getattr(op, "sym_name", None), "data", None) == "forward"
            ),
            None,
        )
        if forward is None:
            raise ValueError("no @forward function")
        count = 0
        for op in query.walk(forward, "linalg.matmul", "linalg.generic"):
            if query.op_name(op) == "linalg.generic" and query.attr_str(op, "prov.family") != "contraction":
                continue
            if len(op.operands) < 2 or not op.results:
                continue
            lhs, rhs = (query.type_shape_dtype(value.type)[1] for value in op.operands[:2])
            if (lhs, rhs) == (element, element):
                count += 1
        if count:
            return count
    except Exception as exc:
        raise M2MUnavailable(f"cannot verify already-materialized capture arithmetic: {exc}") from exc
    raise M2MUnavailable(
        f"already-materialized capture has no {element}×{element} contraction; "
        "its declared operand format is absent from the captured program"
    )


# ---- integerization ----------------------------------------------------------------------------
# Validate a complete integer/preserved-floating PT2E contraction partition.
#
# This is frontend evidence, not accelerator or host admission. Preserving a
# non-FP32 Q/DQ contraction never makes it an integer contraction or grants a
# target implementation for that floating operation.


def contraction_partition(receipt: dict) -> dict[str, int]:
    """Reject missing work, unresolved rewrites, or unaccounted precision refusals."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "m2m.pt2e-integerize.v1":
        raise ValueError("unsupported integerization receipt")

    def count(value: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("invalid integerization contraction census")
        return value

    seen = count(receipt.get("quantized_contractions_seen"))
    integerized = count(receipt.get("quantized_contractions_integerized"))
    remaining = count(receipt.get("quantized_contractions_remaining"))
    if seen < 1 or integerized < 1 or seen != integerized + remaining:
        raise ValueError("incomplete integerization contraction census")
    by_kind = receipt.get("quantized_by_kind")
    kinds = {"linear", "conv2d", "matmul", "unsupported"}
    if not isinstance(by_kind, dict) or set(by_kind) != kinds:
        raise ValueError("missing per-kind integerization contraction census")
    totals = [0, 0, 0]
    for kind in kinds:
        row = by_kind[kind]
        if not isinstance(row, dict):
            raise ValueError("invalid per-kind integerization contraction census")
        values = [count(row.get(key)) for key in ("seen", "integerized", "remaining")]
        if values[0] != values[1] + values[2]:
            raise ValueError("inconsistent per-kind integerization contraction census")
        totals = [total + value for total, value in zip(totals, values)]
    if totals != [seen, integerized, remaining] or by_kind["unsupported"]["seen"]:
        raise ValueError("unresolved or inconsistent integerization contraction census")

    refusals = receipt.get("refusals")
    if not isinstance(refusals, list):
        raise ValueError("missing integerization refusal inventory")
    decisions = receipt.get("precision_decisions")
    # Historical complete-integer receipts did not carry a precision ledger.
    # They can never authorize a preserved floating region through this path.
    if decisions is None:
        if remaining or refusals or receipt.get("precision_decision_counts") is not None:
            raise ValueError("remaining contractions lack an exact preserved-precision ledger")
        return {"seen": seen, "integerized": integerized, "preserved": 0}
    if not isinstance(decisions, list) or len(decisions) != seen:
        raise ValueError("incomplete integerization precision ledger")
    counts = {"integerized_i32": 0, "preserve_float_qdq": 0, "unresolved": 0}
    identities: set[tuple[str, str]] = set()
    preserved: set[tuple[str, str]] = set()
    decision_kinds = {kind: [0, 0, 0] for kind in kinds}
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("invalid integerization precision decision")
        kind, node, decision = row.get("kind"), row.get("node"), row.get("decision")
        if (
            not isinstance(kind, str)
            or kind not in kinds
            or not isinstance(node, str)
            or not node
            or not isinstance(decision, str)
            or decision not in counts
        ):
            raise ValueError("invalid integerization precision decision")
        identity = (kind, node)
        if identity in identities:
            raise ValueError("duplicate integerization precision decision")
        identities.add(identity)
        counts[decision] += 1
        decision_kinds[kind][0] += 1
        if decision == "integerized_i32":
            decision_kinds[kind][1] += 1
        elif decision == "preserve_float_qdq":
            if (
                not isinstance(row.get("source_dtype"), str)
                or row["source_dtype"] not in {"torch.bfloat16", "torch.float16"}
                or row.get("required_numeric_semantics") != "dequantize-before-floating-contraction"
            ):
                raise ValueError("preserved Q/DQ lacks exact non-FP32 floating semantics")
            preserved.add(identity)
            decision_kinds[kind][2] += 1
    declared_counts = receipt.get("precision_decision_counts")
    if not isinstance(declared_counts, dict) or set(declared_counts) != set(counts):
        raise ValueError("missing integerization precision partition counts")
    declared_counts = {key: count(value) for key, value in declared_counts.items()}
    if (
        counts != {"integerized_i32": integerized, "preserve_float_qdq": remaining, "unresolved": 0}
        or declared_counts != counts
        or any(
            decision_kinds[kind] != [by_kind[kind][key] for key in ("seen", "integerized", "remaining")]
            for kind in kinds
        )
    ):
        raise ValueError("unresolved or inconsistent integerization precision partition")
    refused: set[tuple[str, str]] = set()
    for row in refusals:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("kind", "node")):
            raise ValueError("invalid integerization refusal")
        identity = (row.get("kind"), row.get("node"))
        if (
            identity not in preserved
            or identity in refused
            or not isinstance(row.get("reason"), str)
            or not row["reason"]
        ):
            raise ValueError("integerization refusal is not an exact preserved-precision decision")
        refused.add(identity)
    if refused != preserved:
        raise ValueError("preserved-precision decisions and refusal inventory differ")
    return {"seen": seen, "integerized": integerized, "preserved": remaining}


_DYNAMIC_TRANSFORMS = frozenset({"static_attention", "dynamic_contraction"})


def _i8_i8_i32(row: dict) -> bool:
    """A traced MLIR op whose first two operands are i8 tensors and whose one result is an i32 tensor."""

    def element(text: object) -> str | None:
        if not isinstance(text, str) or not text.startswith("tensor<") or not text.endswith(">"):
            return None
        return text[len("tensor<") : -1].rsplit("x", 1)[-1]

    operands, results = row.get("operand_types"), row.get("result_types")
    return (
        isinstance(operands, list)
        and len(operands) >= 2
        and [element(t) for t in operands[:2]] == ["i8", "i8"]
        and isinstance(results, list)
        and [element(t) for t in results] == ["i32"]
    )


def traced_integer_owners(trace: dict, program_sha256: str) -> tuple[int, dict[tuple[str, ...], int]]:
    """Owners of every emitted i8 x i8 -> i32 contraction, from the stage's byte-bound frontend trace.

    Each such ``linalg.generic`` must come from exactly one prepared-graph ``aten._int_mm`` node, every
    such node must be emitted exactly once, and the node's transformation names its owner: the PT2E
    integerization (a weight contraction) or a recipe rewrite (a dynamic site, keyed by the original
    nodes of its lineage). Returns (weight contractions, dynamic contractions per lineage).
    """
    from collections import Counter

    mlir = trace.get("mlir") if isinstance(trace, dict) else None
    graphs = trace.get("graphs") if isinstance(trace, dict) else None
    prepared = (graphs or {}).get("prepared") if isinstance(graphs, dict) else None
    if (
        not isinstance(mlir, dict)
        or mlir.get("sha256") != program_sha256
        or not isinstance(mlir.get("operations"), list)
        or not isinstance(prepared, dict)
        or not isinstance(prepared.get("nodes"), list)
    ):
        raise ValueError("dynamic contraction census trace is not bound to the stage program")
    int_mm = {
        node["id"]: node
        for node in prepared["nodes"]
        if isinstance(node, dict) and node.get("target") == "aten._int_mm.default" and isinstance(node.get("id"), str)
    }
    selected_nodes = [
        node for node in prepared["nodes"] if isinstance(node, dict) and node.get("target") == "aten._int_mm.default"
    ]
    if len(selected_nodes) != len(int_mm):
        raise ValueError("traced integer contractions have duplicate or missing node identities")
    seen: Counter = Counter()
    weight, dynamic = 0, Counter()
    for row in mlir["operations"]:
        if not isinstance(row, dict) or row.get("operation") != "linalg.generic" or not _i8_i8_i32(row):
            continue
        sources = row.get("source_node_ids")
        if (
            not isinstance(sources, list)
            or len(sources) != 1
            or not isinstance(sources[0], str)
            or sources[0] not in int_mm
        ):
            raise ValueError("an emitted integer contraction has no unique traced owner")
        node = int_mm[sources[0]]
        seen[sources[0]] += 1
        transform = node.get("transformation")
        if transform == "pt2e_integerization":
            weight += 1
        elif transform in _DYNAMIC_TRANSFORMS:
            origins = node.get("origin_node_ids")
            if not isinstance(origins, list) or any(not isinstance(v, str) for v in origins):
                raise ValueError("a dynamic integer contraction has no original lineage")
            key = tuple(sorted(v for v in origins if v.startswith("g:original:")))
            if not key:
                raise ValueError("a dynamic integer contraction has no original lineage")
            dynamic[key] += 1
        else:
            raise ValueError("an emitted integer contraction has an unknown traced transformation")
    if set(seen) != set(int_mm) or any(count != 1 for count in seen.values()):
        raise ValueError("traced integer contractions are not each emitted exactly once")
    return weight, dict(dynamic)


def _report_only_quality(selection: dict) -> dict:
    """Project an explicit validated numerical selection without publishing recipe details."""
    from merlin.common.digest import is_sha256
    from merlin.compiler.quantization.recipe import digest
    from merlin.development.targetgen.policies.precision import validate

    identities = ("source_recipe_sha256", "source_methodology_sha256", "source_output_path_sha256")
    if (
        not isinstance(selection, dict)
        or set(selection) != {"schema", "policy", *identities}
        or selection.get("schema") != "numerical_recipe_selection_v1"
        or not all(is_sha256(selection.get(key)) for key in identities)
    ):
        raise ValueError("report-only dynamic quality requires an exact numerical recipe selection")
    policy = validate(selection["policy"])
    return {
        "schema": "merlin.dynamic_float_quality.v1",
        "mode": "report_only",
        "qualified": False,
        "selection_sha256": digest(selection),
        "policy_sha256": digest(policy),
        **{key: selection[key] for key in identities},
    }


def dynamic_integerization_count(evidence: dict, *, emitted: int, exported: int, program_sha256: str) -> int:
    """Recheck an answer-free observation of recipe-site and traced-owner accounting.

    The caller binds this observation to materialized capture metadata and trace bytes.
    This is no independent source-execution attestation or target numerical verdict.
    """
    import math
    from collections import Counter

    from merlin.common.digest import is_sha256
    from merlin.compiler.quantization.recipe import site_agreement_of

    if not isinstance(evidence, dict) or evidence.get("schema") not in (
        "merlin.dynamic_integerization.v1",
        "merlin.dynamic_integerization.v2",
    ):
        raise ValueError("invalid dynamic integerization observation")
    if not is_sha256(evidence.get("recipe_sha256")):
        raise ValueError("dynamic integerization observation lacks a recipe identity")
    trace_identity = evidence.get("trace")
    if (
        not isinstance(trace_identity, dict)
        or not is_sha256(trace_identity.get("sha256"))
        or type(trace_identity.get("bytes")) is not int
        or trace_identity["bytes"] <= 0
        or evidence.get("program_sha256") != program_sha256
        or not is_sha256(program_sha256)
    ):
        raise ValueError("dynamic contraction census trace is not bound to the stage program")
    bar = site_agreement_of({"dynamic": {"site_agreement": evidence.get("site_agreement")}})
    if evidence["schema"] == "merlin.dynamic_integerization.v1":
        if bar is None:
            raise ValueError("the recipe declares no per-site agreement bar for its dynamic sites")
    else:
        quality = evidence.get("float_quality")
        identities = {
            "selection_sha256",
            "policy_sha256",
            "source_recipe_sha256",
            "source_methodology_sha256",
            "source_output_path_sha256",
        }
        if (
            bar is not None
            or not isinstance(quality, dict)
            or set(quality) != {"schema", "mode", "qualified", *identities}
            or quality.get("schema") != "merlin.dynamic_float_quality.v1"
            or quality.get("mode") != "report_only"
            or quality.get("qualified") is not False
            or not all(is_sha256(quality.get(key)) for key in identities)
        ):
            raise ValueError("invalid explicit report-only dynamic quality observation")
    families, samples = evidence.get("families"), evidence.get("samples")
    if not isinstance(families, list) or not families or any(not isinstance(f, str) for f in families):
        raise ValueError("invalid selected dynamic contraction families")
    if type(emitted) is not int or emitted < 0 or type(exported) is not int or exported < 1:
        raise ValueError("invalid integer contraction counts")
    weight, roster = evidence.get("weight_contractions"), evidence.get("dynamic_owners")
    if type(weight) is not int or weight < 0 or not isinstance(roster, list):
        raise ValueError("invalid traced integer contraction owners")
    traced = {}
    for row in roster:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("lineage"), list)
            or not row["lineage"]
            or any(not isinstance(v, str) or not v.startswith("g:original:") for v in row["lineage"])
            or type(row.get("calls")) is not int
            or row["calls"] <= 0
        ):
            raise ValueError("invalid traced dynamic contraction owner")
        key = tuple(sorted(row["lineage"]))
        if key in traced:
            raise ValueError("duplicate traced dynamic contraction owner")
        traced[key] = row["calls"]
    block = evidence.get("contractions")

    def need(condition: bool, why: str) -> None:
        if not condition:
            raise ValueError(f"capture dynamic contraction census refused: {why}")

    need(isinstance(block, dict), "invalid dynamic block")
    sites, calls = block.get("sites"), block.get("device_calls")
    need(
        isinstance(sites, list)
        and type(block.get("rewritten")) is int
        and block.get("rewritten") == len(sites)
        and block.get("kept_in_float") == []
        and type(calls) is int
        and calls >= 0,
        "every dynamic site must be rewritten and none kept in floating point",
    )
    need(bool(sites) or (calls == 0 and not traced), "empty dynamic census differs from the traced dynamic owners")
    need(type(samples) is int and samples >= 1, "the recipe agreement records no samples")
    owned: Counter = Counter()
    for site in sites:
        check = site.get("verification") if isinstance(site, dict) else None
        need(isinstance(site, dict), "invalid dynamic site")
        shapes, lineage = site.get("shapes"), site.get("lineage")
        need(
            isinstance(site, dict)
            and site.get("status") == "rewritten"
            and site.get("family") in families
            and type(site.get("device_calls")) is int
            and site["device_calls"] > 0,
            "a dynamic site is not a rewritten contraction of a selected family",
        )
        need(
            isinstance(shapes, list)
            and bool(shapes)
            and isinstance(shapes[0], list)
            and bool(shapes[0])
            and type(shapes[0][-1]) is int
            and 0 < shapes[0][-1]
            and shapes[0][-1] * 128 * 128 <= (1 << 31) - 1,
            "a dynamic site's reduction bound is not proved",
        )
        metrics = ("cosine_min", "cosine_mean", "max_abs_error", "rel_rmse_max")
        need(
            isinstance(check, dict)
            and check.get("schema") == "m2m.dynamic_site_verification.v1"
            and type(check.get("samples")) is int
            and check.get("samples") == samples
            and check.get("finite") is True
            and all(type(check.get(key)) is float and math.isfinite(check[key]) for key in metrics),
            "a dynamic site lacks a finite verification on the recipe agreement samples",
        )
        need(
            check["max_abs_error"] >= 0.0
            and check["rel_rmse_max"] >= 0.0
            and check["cosine_min"] <= check["cosine_mean"],
            "a dynamic site's verification metrics are inconsistent",
        )
        if bar is not None:
            need(
                check["cosine_min"] >= bar["cosine_min"] and check["rel_rmse_max"] <= bar["rel_rmse_max"],
                "a dynamic site is outside the recipe's declared agreement bar",
            )
        key = tuple(sorted(v for v in lineage or () if isinstance(v, str) and v.startswith("g:original:")))
        need(isinstance(lineage, list) and bool(key), "a dynamic site has no original lineage")
        owned[key] += site["device_calls"]
    need(sum(owned.values()) == calls, "dynamic device calls do not sum to the census total")
    need(emitted + calls == exported, "exported integer contractions are not the emitted plus dynamic ones")
    need(weight == emitted and traced == dict(owned), "traced integer contractions differ from their owners")
    return calls


def dynamic_integerization_projection(
    meta: dict, stats: dict, *, trace: dict, trace_identity: dict, program_sha256: str, emitted: int, exported: int
) -> dict:
    """Validate raw recipe/trace identity and project only bounded census evidence.

    Do not publish the private recipe, graph, workload names, source paths or tensor values.
    The receipt owner still has to bind all three artifacts to their exact bytes.
    """
    from merlin.compiler.quantization.recipe import DYNAMIC, digest, site_agreement_of

    recipe = meta.get("recipe")
    if not (isinstance(recipe, dict) and recipe.get("recipe_sha256") == meta.get("recipe_sha256") == digest(recipe)):
        raise ValueError("capture dynamic contraction census refused: the recipe snapshot does not match its digest")
    bar = site_agreement_of(recipe)
    selection = recipe.get("numerical_selection")
    quality = None if selection is None else _report_only_quality(selection)
    if quality is None and bar is None:
        raise ValueError("the recipe declares no per-site agreement bar for its dynamic sites")
    if quality is not None and (bar is not None or recipe.get("software_numerical_engine") != "integer_reference"):
        raise ValueError("report-only dynamic quality requires an unambiguous independent integer-reference recipe")
    block = stats.get("dynamic_contractions")
    if not isinstance(block, dict) or not isinstance(block.get("sites"), list):
        raise ValueError("capture dynamic contraction census refused: invalid dynamic block")
    if not block["sites"] and any(
        type(block.get(key)) is not int or block[key] != 0 for key in ("host_elements", "int8_macs")
    ):
        raise ValueError("capture dynamic contraction census refused: empty domain has invalid work counters")
    sites = []
    for site in block["sites"]:
        if not isinstance(site, dict):
            raise ValueError("capture dynamic contraction census refused: invalid dynamic site")
        # In particular, omit the site node's possibly workload-specific name.
        projected = {key: site.get(key) for key in ("family", "status", "device_calls", "shapes", "lineage")}
        lineage = site.get("lineage")
        projected["lineage"] = (
            [value for value in lineage if isinstance(value, str) and value.startswith("g:original:")]
            if isinstance(lineage, list)
            else None
        )
        verification = site.get("verification")
        projected["verification"] = (
            {
                key: verification.get(key)
                for key in ("schema", "samples", "finite", "cosine_min", "cosine_mean", "max_abs_error", "rel_rmse_max")
            }
            if isinstance(verification, dict)
            else None
        )
        sites.append(projected)
    weight, dynamic = traced_integer_owners(trace, program_sha256)
    agreement = meta.get("recipe_agreement")
    evidence = {
        "schema": "merlin.dynamic_integerization.v1" if quality is None else "merlin.dynamic_integerization.v2",
        **({"float_quality": quality} if quality is not None else {}),
        "recipe_sha256": meta["recipe_sha256"],
        "families": (recipe.get(DYNAMIC) or {}).get("families"),
        "site_agreement": bar,
        "samples": agreement.get("samples") if isinstance(agreement, dict) else None,
        "program_sha256": program_sha256,
        "trace": {key: trace_identity.get(key) for key in ("sha256", "bytes")},
        "weight_contractions": weight,
        "dynamic_owners": [{"lineage": list(key), "calls": calls} for key, calls in sorted(dynamic.items())],
        "contractions": {
            **{key: block.get(key) for key in ("rewritten", "kept_in_float", "device_calls")},
            "sites": sites,
        },
    }
    dynamic_integerization_count(evidence, emitted=emitted, exported=exported, program_sha256=program_sha256)
    return evidence
