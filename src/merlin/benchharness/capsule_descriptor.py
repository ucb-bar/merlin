"""Public, versioned ABI for generic CPU-host capsule descriptors.

This module is deliberately staged into every treatment workspace.  It defines the exact MLIR
dictionary grammar consumed by the trusted grader, stable split-independent enum tables, and six
synthetic conformance fixtures.  It contains no held-out capsules, paper models, or reference data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FAMILY_CODE = {
    "contraction": 1,
    "elementwise_map": 2,
    "reduction": 3,
    "movement_layout": 4,
    "fusion_epilogue": 5,
    "runtime_parallel": 6,
}
DTYPE_CODE = {
    "fp32": 1,
    "int8": 2,
    "int32": 3,
    "w8a8_i32": 4,
    "int8_i32": 5,
}
LAYOUT_CODE = {
    "contiguous": 1,
    "row_row": 2,
    "row_packed_rhs": 3,
    "transposed_rhs": 4,
    "operation_defined": 5,
}
OPERATION_CODE = {
    name: index
    for index, name in enumerate((
        "add", "barrier", "batch_matmul", "clamp", "concatenate",
        "convolution_im2col", "copy", "gelu", "layernorm_components", "matmul",
        "matmul_bias", "matmul_bias_relu", "matmul_requant", "max", "multiply",
        "pack_rhs", "persistent_weight_reuse", "producer_consumer", "relu", "requant",
        "residual_norm", "silu", "single_hart", "softmax_components", "static_partition",
        "strided_slice", "sum", "transpose2d", "unpack",
    ), start=1)
}
SEMANTIC_OPERATION_CODE = {
    "matmul": 1, "batch_matmul": 1, "convolution_im2col": 1,
    "add": 2, "multiply": 3, "relu": 4, "silu": 5, "gelu": 6, "clamp": 7,
    "requant": 8, "sum": 9, "max": 10, "softmax_components": 11,
    "layernorm_components": 12, "copy": 13, "transpose2d": 14, "pack_rhs": 15,
    "unpack": 16, "strided_slice": 17, "concatenate": 18, "matmul_bias": 19,
    "matmul_bias_relu": 20, "matmul_requant": 21, "residual_norm": 22,
    "single_hart": 23, "static_partition": 23, "producer_consumer": 23,
    "barrier": 23, "persistent_weight_reuse": 23,
}
KIND_CODE = {"none": 0, "fp32": 1, "int8": 2, "int32": 3}


def _positive_int(value: object, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def dimensions(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Return the exact ``dim0, dim1, dim2, state0`` descriptor fields."""
    shape = row.get("shape")
    if not isinstance(shape, Mapping):
        raise ValueError("capsule shape must be a mapping")
    family = str(row.get("family", ""))
    if family in {"contraction", "fusion_epilogue"}:
        return (
            _positive_int(shape.get("M"), "shape.M"),
            _positive_int(shape.get("N"), "shape.N"),
            _positive_int(shape.get("K"), "shape.K"),
            0,
        )
    if "length" in shape:
        return _positive_int(shape.get("length"), "shape.length"), 0, 0, 0
    if "working_set_bytes" in shape:
        size = _positive_int(shape.get("working_set_bytes"), "shape.working_set_bytes")
        element_bytes = 4 if row.get("dtype") == "fp32" else 1
        count = size // element_bytes
        if count == 0:
            raise ValueError("working_set_bytes is smaller than one dtype element")
        rows = int(count ** 0.5)
        while rows > 1 and count % rows:
            rows -= 1
        return count, rows, count // rows, count // 2
    if "work_items" in shape:
        state = row.get("state")
        reuse = state.get("reuse_count", 1) if isinstance(state, Mapping) else 1
        return (
            _positive_int(shape.get("work_items"), "shape.work_items"),
            0,
            0,
            _positive_int(reuse, "state.reuse_count"),
        )
    raise ValueError("capsule shape has no supported dimension spelling")


def buffer_plan(row: Mapping[str, Any]) -> dict[str, int | str]:
    """Resolve the exact typed buffer extents carried by the canonical MLIR descriptor."""
    d0, d1, d2, state0 = dimensions(row)
    family = str(row.get("family", ""))
    operation = str(row.get("operation", ""))
    dtype = str(row.get("dtype", ""))
    if operation not in SEMANTIC_OPERATION_CODE:
        raise ValueError(f"unsupported operation {operation!r}")
    plan: dict[str, int | str] = {
        "dim0": d0, "dim1": d1, "dim2": d2, "state0": state0,
        "input0_kind": "none", "input1_kind": "none", "input2_kind": "none",
        "output_kind": "none", "input0_count": 0, "input1_count": 0,
        "input2_count": 0, "output_count": 0,
    }
    if family == "contraction":
        number = "fp32" if dtype == "fp32" else "int8"
        output = "fp32" if dtype == "fp32" else "int32"
        rhs = d2 * d1 if row.get("layout") != "row_packed_rhs" else d2 * ((d1 + 7) // 8) * 8
        plan.update(input0_kind=number, input1_kind=number, output_kind=output,
                    input0_count=d0 * d2, input1_count=rhs, output_count=d0 * d1)
    elif family == "elementwise_map":
        number = {"fp32": "fp32", "int8": "int8", "int32": "int32"}[dtype]
        binary = operation in {"add", "multiply"}
        plan.update(input0_kind=number, input1_kind=number if binary else "none",
                    output_kind=number, input0_count=d0, input1_count=d0 if binary else 0,
                    output_count=d0)
    elif family == "reduction":
        number = "fp32" if dtype == "fp32" else "int8"
        output = "fp32" if dtype == "fp32" else "int32"
        components = operation in {"softmax_components", "layernorm_components"}
        plan.update(input0_kind=number, output_kind=output, input0_count=d0,
                    output_count=2 if components else 1)
    elif family == "movement_layout":
        number = "fp32" if dtype == "fp32" else "int8"
        in0, in1, out = d0, 0, d0
        if operation == "pack_rhs":
            out = d1 * ((d2 + 7) // 8) * 8
        elif operation == "unpack":
            in0 = d1 * ((d2 + 7) // 8) * 8
        elif operation == "strided_slice":
            out = (d0 + 1) // 2
        elif operation == "concatenate":
            in0, in1 = state0, d0 - state0
        plan.update(input0_kind=number, input1_kind=number if in1 else "none",
                    output_kind=number, input0_count=in0, input1_count=in1, output_count=out)
    elif family == "fusion_epilogue":
        number = "fp32" if dtype == "fp32" else "int8"
        output = "fp32" if dtype == "fp32" else "int32"
        third = d0 * d1 if operation == "residual_norm" else d1
        plan.update(input0_kind=number, input1_kind=number, input2_kind=output,
                    output_kind=output, input0_count=d0 * d2, input1_count=d2 * d1,
                    input2_count=third, output_count=d0 * d1)
    elif family == "runtime_parallel":
        plan.update(input0_kind="fp32", input1_kind="fp32", output_kind="fp32",
                    input0_count=d0, input1_count=d0, output_count=d0)
    else:
        raise ValueError(f"unsupported family {family!r}")
    return plan


def render_capsule_mlir(row: Mapping[str, Any]) -> str:
    """Render the one canonical v1 descriptor accepted by the trusted grader."""
    family = str(row.get("family", ""))
    operation = str(row.get("operation", ""))
    dtype = str(row.get("dtype", ""))
    layout = str(row.get("layout", ""))
    if family not in FAMILY_CODE or dtype not in DTYPE_CODE or layout not in LAYOUT_CODE:
        raise ValueError("capsule names values outside the public enum tables")
    if operation not in OPERATION_CODE or operation not in SEMANTIC_OPERATION_CODE:
        raise ValueError("capsule operation is outside the public enum tables")
    digest = str(row.get("sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("capsule sha256 must be 64 lowercase hexadecimal characters")
    harts = _positive_int(row.get("core_count"), "core_count")
    d0, d1, d2, state0 = dimensions(row)
    plan = buffer_plan(row)
    mlir_type = {"none": "i8", "fp32": "f32", "int8": "i8", "int32": "i32"}

    def memref(which: str) -> str:
        return f'memref<{plan[which + "_count"]}x{mlir_type[str(plan[which + "_kind"])]}>'

    return f'''module attributes {{merlin.capsule = {{
  sha256 = "{digest}", family = "{family}", family_code = {FAMILY_CODE[family]} : i32,
  operation = "{operation}", operation_code = {OPERATION_CODE[operation]} : i32,
  semantic_operation_code = {SEMANTIC_OPERATION_CODE[operation]} : i32,
  dtype = "{dtype}", dtype_code = {DTYPE_CODE[dtype]} : i32,
  layout = "{layout}", layout_code = {LAYOUT_CODE[layout]} : i32,
  dim0 = {d0} : i64, dim1 = {d1} : i64, dim2 = {d2} : i64, state0 = {state0} : i64,
  input0_count = {plan["input0_count"]} : i64, input1_count = {plan["input1_count"]} : i64,
  input2_count = {plan["input2_count"]} : i64, output_count = {plan["output_count"]} : i64,
  requested_harts = {harts} : i32
}}}} {{
  func.func private @capsule(%input0: {memref("input0")}, %input1: {memref("input1")},
                             %input2: {memref("input2")}, %output: {memref("output")})
}}
'''


def _fixture_row(family: str, operation: str, dtype: str, layout: str,
                 shape: Mapping[str, int], state: object, core_count: int) -> dict[str, Any]:
    identity = {
        "family": family, "operation": operation, "dtype": dtype,
        "shape": dict(shape), "layout": layout, "state": state, "core_count": core_count,
    }
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {**identity, "sha256": digest}


def conformance_rows() -> tuple[dict[str, Any], ...]:
    """Return one synthetic, non-paper fixture for each generic family."""
    return (
        _fixture_row("contraction", "matmul", "fp32", "row_row",
                     {"M": 2, "N": 3, "K": 4}, "stateless", 1),
        _fixture_row("elementwise_map", "add", "int32", "contiguous",
                     {"length": 9}, "stateless", 1),
        _fixture_row("reduction", "softmax_components", "int8_i32", "contiguous",
                     {"length": 17}, "stateless", 1),
        _fixture_row("movement_layout", "concatenate", "fp32", "operation_defined",
                     {"working_set_bytes": 64}, "stateless", 1),
        _fixture_row("fusion_epilogue", "matmul_bias_relu", "w8a8_i32", "row_row",
                     {"M": 2, "N": 3, "K": 4}, "fused_epilogue", 1),
        _fixture_row("runtime_parallel", "producer_consumer", "fp32", "contiguous",
                     {"work_items": 11}, {"reuse_count": 8}, 4),
    )


def write_conformance_fixtures(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for row in conformance_rows():
        name = f'{row["family"]}-{row["operation"]}.mlir'
        (destination / name).write_text(render_capsule_mlir(row), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args(argv)
    write_conformance_fixtures(args.fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
