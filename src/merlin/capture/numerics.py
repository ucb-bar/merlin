"""Original captured-output numerical policies and explicit precision recipe projection."""

from __future__ import annotations

import math

from merlin.capture.artifacts import M2MUnavailable

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
        from merlin.common import quant_formats as _qf
        from merlin.compiler.quantization import capabilities as _qr

        recipe = _qr.select(_qr.for_target(target))
        if recipe is None or recipe.status != _qr.DERIVED:
            return None
        wanted = _qf.get(str(dtype)).name if _qf.has(str(dtype)) else str(dtype)
        return recipe.to_dict() if recipe.weight.dtype == wanted else None
    except Exception:  # noqa: BLE001 -- a derivation fault leaves the named scheme in force
        return None
