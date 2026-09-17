"""Compiler-owned operand ABI for native block-scaled MX GEMM emission.

The older Muon path receives ``mx_operands`` from a capsule golden and replaces the submitted
artifact with a known-good reference program.  That is useful oracle infrastructure, but it is not a
compiler lowering.  This module defines the independent native boundary: select one semantic GEMM,
quantize ordinary tensor values (or validate explicit model quantization metadata), and bind the
resulting E4M3 codes plus E8M0 scales to the emitted program by digest.

Only the generic mxfp8 GEMM family is admitted in v1.  Every other shape/format/family fails closed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from merlin.targetgen.fp8_codec import fp8_e4m3_encode

ABI_SCHEMA = "semantic_mxfp8_gemm_operand_abi_v1"
NATIVE_FAMILY = "kernels/gemm_mxgemmini"
PROGRAM_MARKER = "// merlin-native-mx-abi-sha256:"
_BLOCK = 32
_E8M0_BIAS = 127
_E8M0_MAX_CODE = 254
# Keep each normalized value in the unit binade (largest E4M3 value there is 1.75).  The selected
# schedule quantizes a narrow running accumulator after every product; filling the full E4M3 range
# (448) can overflow that accumulator before the external E8M0 scale is applied.  This conservative
# compiler policy preserves range in the scale and keeps the validated family numerically closed.
_E4M3_BLOCK_QUANT_TARGET = 1.75
_LEGACY_KEYS = frozenset(
    {
        "mx_operands",
        "canonical_inputs",
        "golden",
        "golden_outputs",
        "reference_kernel",
        "reference_kernel_path",
    }
)


class NativeMxAbiError(ValueError):
    """The semantic buffer cannot be represented by the native MX ABI."""


def is_native_mx_cb(cb: Mapping[str, Any]) -> bool:
    params = cb.get("params")
    return (
        isinstance(params, Mapping)
        and isinstance(params.get("mx_operand_abi"), Mapping)
        and params["mx_operand_abi"].get("schema") == ABI_SCHEMA
    )


def _reject_legacy(cb: Mapping[str, Any]) -> None:
    present = sorted(key for key in _LEGACY_KEYS if key in cb)
    if present:
        raise NativeMxAbiError(f"native MX emission rejects legacy golden/reference operand fields {present}")


def _flat(values: Any) -> list[float]:
    if isinstance(values, Mapping):
        values = values.get("values")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        out: list[float] = []
        for value in values:
            out.extend(_flat(value))
        return out
    if values is None:
        raise NativeMxAbiError("actual tensor has no values")
    value = float(values)
    if not math.isfinite(value):
        raise NativeMxAbiError(f"actual tensor contains non-finite value {value!r}")
    return [value]


def _bytes_field(meta: Mapping[str, Any], plural: str, encoded: str, count: int) -> list[int]:
    raw = meta.get(plural)
    if raw is None and meta.get(encoded) is not None:
        try:
            raw = list(base64.b64decode(str(meta[encoded]), validate=True))
        except Exception as exc:  # noqa: BLE001 - malformed external metadata is one ABI refusal
            raise NativeMxAbiError(f"invalid base64 in quantization metadata field {encoded}") from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise NativeMxAbiError(f"quantization metadata needs {plural} or {encoded}")
    result = [int(value) for value in raw]
    if len(result) != count:
        raise NativeMxAbiError(f"{plural} has {len(result)} entries, expected {count}")
    if any(value < 0 or value > 255 for value in result):
        raise NativeMxAbiError(f"{plural} contains a value outside one byte")
    return result


def _gemm(cb: Mapping[str, Any]) -> dict[str, Any]:
    tensors = cb.get("tensors")
    commands = cb.get("commands")
    if not isinstance(tensors, Mapping) or not isinstance(commands, list):
        raise NativeMxAbiError("native MX GEMM needs command-buffer tensors and commands")
    resident: dict[str, str] = {}
    matmuls: list[Mapping[str, Any]] = []
    commits: list[Mapping[str, Any]] = []
    for command in commands:
        if not isinstance(command, Mapping):
            raise NativeMxAbiError("native MX GEMM command is not a mapping")
        opcode = str(command.get("opcode") or "").upper()
        operands = command.get("operands")
        if not isinstance(operands, Mapping):
            raise NativeMxAbiError(f"{opcode or 'unnamed'} command has no operand mapping")
        if opcode == "RES_PACK":
            resident[str(operands.get("dst"))] = str(operands.get("src"))
        elif opcode in {"MATMUL", "MATMUL_RESIDENT"}:
            matmuls.append(command)
        elif opcode == "COMMIT":
            commits.append(command)
        elif opcode != "EVICT":
            raise NativeMxAbiError(f"native MX GEMM does not admit command {opcode!r}")
    if len(matmuls) != 1 or len(commits) != 1:
        raise NativeMxAbiError("native MX GEMM requires exactly one matmul and one commit")
    mm_ops = matmuls[0]["operands"]
    commit_ops = commits[0]["operands"]
    lhs = mm_ops.get("lhs")
    rhs_handle = mm_ops.get("rhs") or mm_ops.get("weight")
    rhs = resident.get(str(rhs_handle), rhs_handle)
    dst = commit_ops.get("dst")
    if commit_ops.get("src") != mm_ops.get("dst"):
        raise NativeMxAbiError("commit must consume the native MX matmul result")
    if not isinstance(dst, str) or not dst:
        raise NativeMxAbiError("native MX GEMM commit needs a named output")
    if not all(isinstance(name, str) and name in tensors for name in (lhs, rhs)):
        raise NativeMxAbiError("matmul lhs/weight are absent from the tensor table")
    lhs_spec, rhs_spec = tensors[lhs], tensors[rhs]
    if not isinstance(lhs_spec, Mapping) or not isinstance(rhs_spec, Mapping):
        raise NativeMxAbiError("matmul tensor specifications are not mappings")
    operand_dtypes = {str(lhs_spec.get("dtype") or "").lower(), str(rhs_spec.get("dtype") or "").lower()}
    if not operand_dtypes <= {"mxfp8", "f8e4m3fn"}:
        raise NativeMxAbiError(f"native MX ABI v1 requires mxfp8/E4M3 operands, got {sorted(operand_dtypes)}")
    lhs_shape, rhs_shape = lhs_spec.get("shape"), rhs_spec.get("shape")
    if not isinstance(lhs_shape, list) or len(lhs_shape) != 2 or not isinstance(rhs_shape, list) or len(rhs_shape) != 2:
        raise NativeMxAbiError("native MX GEMM requires rank-2 lhs and weight")
    m, k = (int(value) for value in lhs_shape)
    k2, n = (int(value) for value in rhs_shape)
    if k != k2 or min(m, n, k) <= 0:
        raise NativeMxAbiError(f"incompatible GEMM shapes {lhs_shape} and {rhs_shape}")
    if m % 16 or n % 16 or k % _BLOCK:
        raise NativeMxAbiError("generic MX GEMM requires M/N multiples of 16 and K a multiple of 32")
    attrs = commits[0].get("attributes") or {}
    if (
        not isinstance(attrs, Mapping)
        or attrs.get("epilogue", []) not in ([], None)
        or str(attrs.get("output_dtype") or "bf16").lower() not in {"bf16", "bfloat16"}
    ):
        raise NativeMxAbiError("native mxfp8 GEMM v1 admits only an empty epilogue and bf16 output")
    scales: dict[str, str] = {}
    for name, spec in tensors.items():
        if not isinstance(spec, Mapping) or str(spec.get("role") or "").lower() != "scale":
            continue
        scale_of = spec.get("scale_of")
        if isinstance(scale_of, str):
            if scale_of in scales:
                raise NativeMxAbiError(f"multiple scale tensors declare scale_of={scale_of!r}")
            scales[scale_of] = str(name)
    if lhs not in scales or rhs not in scales:
        raise NativeMxAbiError("lhs and weight each need an explicit scale tensor with scale_of")
    for operand, lanes in ((lhs, m), (rhs, n)):
        spec = tensors[scales[operand]]
        if (
            spec.get("shape") != [k // _BLOCK, lanes]
            or int(spec.get("block", 0)) != _BLOCK
            or str(spec.get("dtype") or "").lower() not in {"e8m0", "f8e8m0fnu", "i8"}
        ):
            raise NativeMxAbiError(f"scale tensor {scales[operand]!r} must be [{k // _BLOCK}, {lanes}] E8M0, block 32")
    return {
        "lhs": lhs,
        "rhs": rhs,
        "output": dst,
        "M": m,
        "N": n,
        "K": k,
        "lhs_scale": scales[lhs],
        "rhs_scale": scales[rhs],
    }


def semantic_request(cb: Mapping[str, Any]) -> dict[str, Any]:
    """Selection request for the one native family, derived only from semantic CB facts."""
    geom = _gemm(cb)
    return {"op": "matmul", "dtype": "mxfp8", "shape": {key: geom[key] for key in ("M", "N", "K")}}


def _scale_code(values: Sequence[float]) -> int:
    amax = max((abs(float(value)) for value in values), default=0.0)
    exponent = 0 if amax == 0.0 else math.ceil(math.log2(amax / _E4M3_BLOCK_QUANT_TARGET))
    return min(_E8M0_MAX_CODE, max(0, exponent + _E8M0_BIAS))


def _derive_operand(values: list[float], rows: int, cols: int, *, lhs: bool) -> tuple[list[int], list[list[int]]]:
    groups = cols // _BLOCK if lhs else rows // _BLOCK
    lanes = rows if lhs else cols
    scales = [[0] * lanes for _ in range(groups)]
    codes = [0] * (rows * cols)
    for group in range(groups):
        for lane in range(lanes):
            block_values = (
                [values[lane * cols + group * _BLOCK + offset] for offset in range(_BLOCK)]
                if lhs
                else [values[(group * _BLOCK + offset) * cols + lane] for offset in range(_BLOCK)]
            )
            scale_code = _scale_code(block_values)
            scales[group][lane] = scale_code
            scale = 2.0 ** (scale_code - _E8M0_BIAS)
            for offset, value in enumerate(block_values):
                index = lane * cols + group * _BLOCK + offset if lhs else (group * _BLOCK + offset) * cols + lane
                codes[index] = fp8_e4m3_encode(value / scale)
    return codes, scales


def _metadata_operand(
    meta: Mapping[str, Any], elements: int, groups: int, lanes: int
) -> tuple[list[int], list[list[int]]]:
    if str(meta.get("format") or "").lower() != "mxfp8":
        raise NativeMxAbiError("explicit MX quantization metadata must declare format=mxfp8")
    if int(meta.get("block", 0)) != _BLOCK:
        raise NativeMxAbiError("explicit MX quantization metadata must declare block=32")
    codes = _bytes_field(meta, "codes", "codes_b64", elements)
    flat_scales = _bytes_field(meta, "scale_codes", "scale_codes_b64", groups * lanes)
    if 255 in flat_scales:
        raise NativeMxAbiError("E8M0 NaN scale code 255 is not a legal native operand")
    return codes, [flat_scales[g * lanes : (g + 1) * lanes] for g in range(groups)]


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build_native_mxfp8_gemm_abi(
    cb: Mapping[str, Any],
    *,
    actual_tensors: Mapping[str, Any] | None = None,
    model_quantization: Mapping[str, Mapping[str, Any]] | None = None,
    selected_family: str | None = None,
) -> dict[str, Any]:
    """Build the v1 ABI from actual float tensors or explicit model quantization metadata."""
    _reject_legacy(cb)
    geom = _gemm(cb)
    family = selected_family or (((cb.get("params") or {}).get("kernel_family_selection") or {}).get("selected_family"))
    if family != NATIVE_FAMILY:
        raise NativeMxAbiError(f"native MX ABI v1 requires selected family {NATIVE_FAMILY!r}, got {family!r}")
    actual = actual_tensors or {}
    quant = model_quantization or {}
    if bool(actual) == bool(quant):
        raise NativeMxAbiError("native MX ABI needs exactly one source: actual_tensors or model_quantization")
    m, n, k = geom["M"], geom["N"], geom["K"]
    encoded: dict[str, dict[str, Any]] = {}
    for role, name, rows, cols, lanes, is_lhs in (
        ("lhs", geom["lhs"], m, k, m, True),
        ("weight", geom["rhs"], k, n, n, False),
    ):
        if actual:
            if name not in actual:
                raise NativeMxAbiError(f"actual tensor {name!r} was not supplied")
            values = _flat(actual[name])
            if len(values) != rows * cols:
                raise NativeMxAbiError(f"actual tensor {name!r} has {len(values)} values, expected {rows * cols}")
            codes, scales = _derive_operand(values, rows, cols, lhs=is_lhs)
            source = "compiler_amax_quantized_actual_tensor"
        else:
            meta = quant.get(name)
            if not isinstance(meta, Mapping):
                raise NativeMxAbiError(f"model quantization metadata has no entry for {name!r}")
            codes, scales = _metadata_operand(meta, rows * cols, k // _BLOCK, lanes)
            source = "explicit_model_quantization_metadata"
        encoded[role] = {
            "tensor": name,
            "scale_tensor": geom[f"{role if role == 'lhs' else 'rhs'}_scale"],
            "shape": [rows, cols],
            "codes": codes,
            "scale_codes": scales,
            "source": source,
        }
    abi: dict[str, Any] = {
        "schema": ABI_SCHEMA,
        "family": family,
        "format": "mxfp8",
        "element_encoding": "e4m3",
        "scale_encoding": "e8m0",
        "block": _BLOCK,
        "geometry": {key: geom[key] for key in ("M", "N", "K")},
        "output": geom["output"],
        "operands": encoded,
    }
    abi["abi_sha256"] = _digest(abi)
    return abi


def attach_native_mxfp8_gemm_abi(cb: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    abi = build_native_mxfp8_gemm_abi(cb, **kwargs)
    cb.setdefault("params", {})["mx_operand_abi"] = abi
    return abi


def validate_native_mxfp8_gemm_abi(cb: Mapping[str, Any]) -> dict[str, Any]:
    _reject_legacy(cb)
    params = cb.get("params")
    abi = params.get("mx_operand_abi") if isinstance(params, Mapping) else None
    if not isinstance(abi, Mapping) or abi.get("schema") != ABI_SCHEMA:
        raise NativeMxAbiError("command buffer has no native mxfp8 GEMM operand ABI")
    body = dict(abi)
    digest = body.pop("abi_sha256", None)
    if not isinstance(digest, str) or digest != _digest(body):
        raise NativeMxAbiError("native MX operand ABI digest mismatch")
    geom = _gemm(cb)
    if (
        abi.get("family") != NATIVE_FAMILY
        or abi.get("geometry") != {key: geom[key] for key in ("M", "N", "K")}
        or abi.get("output") != geom["output"]
    ):
        raise NativeMxAbiError("native MX operand ABI disagrees with semantic GEMM")
    expected = {
        "lhs": (geom["lhs"], geom["lhs_scale"], geom["M"] * geom["K"], geom["M"]),
        "weight": (geom["rhs"], geom["rhs_scale"], geom["K"] * geom["N"], geom["N"]),
    }
    operands = abi.get("operands")
    if not isinstance(operands, Mapping):
        raise NativeMxAbiError("native MX operand ABI has no operands mapping")
    for role, (name, scale_name, count, lanes) in expected.items():
        spec = operands.get(role)
        if not isinstance(spec, Mapping) or spec.get("tensor") != name or spec.get("scale_tensor") != scale_name:
            raise NativeMxAbiError(f"native MX {role} binding disagrees with tensor declarations")
        codes = spec.get("codes")
        scales = spec.get("scale_codes")
        if (
            not isinstance(codes, list)
            or len(codes) != count
            or any(not isinstance(value, int) or value < 0 or value > 255 for value in codes)
        ):
            raise NativeMxAbiError(f"native MX {role} codes are malformed")
        if (
            not isinstance(scales, list)
            or len(scales) != geom["K"] // _BLOCK
            or any(not isinstance(row, list) or len(row) != lanes for row in scales)
            or any(not isinstance(value, int) or value < 0 or value > 254 for row in scales for value in row)
        ):
            raise NativeMxAbiError(f"native MX {role} E8M0 scale codes are malformed")
    return dict(abi)


def bind_native_program(cb: Mapping[str, Any], source: str) -> None:
    """Verify that a full native program was emitted from this exact operand ABI."""
    abi = validate_native_mxfp8_gemm_abi(cb)
    marker = PROGRAM_MARKER + str(abi["abi_sha256"])
    if marker not in source:
        raise NativeMxAbiError("native MX program is not bound to this command buffer operand ABI")
    if "mx-reference-kernel" in source or "muon-reference MX placeholder" in source:
        raise NativeMxAbiError("native MX route refuses a legacy reference-kernel artifact")


def emitter_bundle(abi: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the validated public ABI to the low-level MX program renderer's data layout."""
    lhs, weight = abi["operands"]["lhs"], abi["operands"]["weight"]
    geom = abi["geometry"]
    return {
        "fmt": "fp8_e4m3",
        "M": geom["M"],
        "N": geom["N"],
        "K": geom["K"],
        "G": 0,
        "A_bytes": lhs["codes"],
        "B_bytes": weight["codes"],
        "SA": lhs["scale_codes"],
        "SB": weight["scale_codes"],
        "lutA": None,
        "lutB": None,
    }
