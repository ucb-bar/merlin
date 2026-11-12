"""Admit realized integer/PT2E arithmetic and its original complete source receipts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from merlin.capture.artifacts import M2MUnavailable
from merlin.capture.numerics import _DEF_ATOL, _DEF_RTOL
from merlin.common.digest import is_sha256


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
    from merlin.common import mlir_query as query

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
        from merlin.development.targetgen.application_inventory import (
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
    from merlin.common import mlir_query as query

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
    from merlin.common import mlir_query as query
    from merlin.development.targetgen.corpus_spec import dtype_info

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
