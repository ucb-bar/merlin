"""Pure capture datatype and numeric deployment receipt projections."""

from __future__ import annotations


def _mlir_dtype(torch_dtype) -> str:
    """Canonical MLIR element spelling for a captured torch tensor dtype.

    Do not infer this from JSON values: ``_to_native`` deliberately converts tensors through float64,
    so integral-looking values and genuinely integral tensors are indistinguishable after serialization.
    """
    import torch

    spelling = {
        torch.bool: "i1",
        torch.int8: "i8",
        torch.uint8: "ui8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.complex64: "complex<f32>",
        torch.complex128: "complex<f64>",
    }
    for name, mlir in (("float8_e4m3fn", "f8E4M3FN"), ("float8_e5m2", "f8E5M2")):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            spelling[dtype] = mlir
    if torch_dtype not in spelling:
        raise RuntimeError(f"unsupported captured input dtype: {torch_dtype}")
    return spelling[torch_dtype]


def _numeric_deployment(meta: dict, *, staged: bool) -> dict:
    """What a transformed session stage actually deploys, stated in its receipt, not implied.

    ``form`` names the precision path (FP32 staging, then the recipe's W8A8, or either alone). The
    contraction census separates what the recipe selected and integerized, what it selected but had to
    keep as dequantized float (with the source dtype that forced it), and what it never selected. The
    activation x activation matmuls in that last group are an explicit, counted scope note: a static
    weight recipe has no weight to quantize there, so they stay float by design.
    """
    receipt = meta.get("integerization_receipt") or {}
    staging = meta.get("fp32_staging")
    conversion = meta.get("precision_conversion") or {}
    audit = conversion.get("staged_precision_audit") or {}
    quantized = bool(receipt)
    form = (
        ("fp32_staged_then_recipe_w8a8" if staged else "recipe_w8a8")
        if quantized
        else ("fp32_staged" if staged else "untransformed")
    )
    record: dict = {"form": form}
    if staged:
        record["precision_staging"] = {
            "target_dtype": audit.get("target_dtype"),
            "audit_status": audit.get("status"),
            "checked_floating_values": audit.get("checked_floating_values"),
            "non_target_floating_values": audit.get("non_target_floating_values"),
            "graph_dtype_retargeting": conversion.get("graph_dtype_retargeting"),
            "retyped_dtype_decisions": len(conversion.get("dtype_decisions") or ()),
            "original_graph_sha256": conversion.get("original_graph_sha256"),
            "staged_graph_sha256": conversion.get("staged_graph_sha256"),
            "source_state_unchanged": bool(staging),
        }
    if quantized:
        by_kind = receipt.get("quantized_by_kind") or {}
        decisions = receipt.get("precision_decisions") or []
        census = {}
        for kind in ("linear", "conv2d", "matmul"):
            seen = int(receipt.get(f"{kind}_seen") or 0)
            selected = int((by_kind.get(kind) or {}).get("seen") or 0)
            integerized = int((by_kind.get(kind) or {}).get("integerized") or 0)
            census[kind] = {
                "seen": seen,
                "recipe_selected": selected,
                "integerized": integerized,
                "selected_kept_float": selected - integerized,
                "not_selected": seen - selected,
            }
        reasons: dict[str, int] = {}
        for row in decisions:
            if row.get("decision") != "integerized_i32":
                key = f"{row.get('decision')}:{row.get('kind')}:{row.get('source_dtype')}"
                reasons[key] = reasons.get(key, 0) + 1
        record.update(
            contractions=census,
            selected_kept_float_reasons=dict(sorted(reasons.items())),
            integer_contractions=int(receipt.get("exported_integer_mm_count") or 0),
        )
        unselected_matmuls = census["matmul"]["not_selected"]
        if unselected_matmuls:
            record["scope_notes"] = [
                {
                    "kind": "matmul",
                    "count": unselected_matmuls,
                    "reason": "activation x activation matmul (no weight operand); a static weight recipe "
                    "does not quantize it, so it stays float",
                }
            ]
    return record
