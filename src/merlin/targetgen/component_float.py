"""Explicit finite floating semantics and ordinary scalar IR for component tests.

This describes source arithmetic, never hardware admission or a numerical oracle.
"""

from __future__ import annotations

FORMATS = ("f8E4M3FN", "f8E5M2", "bf16", "f16", "f32")
POLICY = {
    "schema": "merlin.component_float.v1",
    "domain": "finite",
    "overflow": "reject",
    "subnormals": "preserve",
    "reduction_order": "index_sequential",
    "product_rounding": "accumulator_format",
}


def validate_policy(policy):
    if policy != POLICY:
        raise ValueError("floating component requires the explicit finite ordered v1 policy")


def validate_node(node, *, result_dtype, accumulator_dtype):
    op = node["op"]
    fields = {"name", "op", "inputs"}
    if op == "convert":
        fields |= {"dtype", "rounding"}
        if node.get("dtype") not in FORMATS or node.get("rounding") != "rne":
            raise ValueError("component conversion requires a supported float destination and RNE")
    elif op in {"matmul", "add", "update"}:
        fields |= {"compute_dtype", "rounding"}
        compute, rounding = node.get("compute_dtype"), node.get("rounding")
        if op == "matmul":
            if compute != accumulator_dtype or rounding != "rne":
                raise ValueError("component matmul requires ordered RNE in the selected accumulator")
        elif compute not in {result_dtype, "f32"} or rounding not in {"rne", "rtz"}:
            raise ValueError("component pointwise arithmetic has unsupported computation/rounding")
        if rounding == "rtz" and (compute, result_dtype) != ("f32", "bf16"):
            raise ValueError("component RTZ currently requires explicit FP32 computation and BF16 storage")
    if set(node) != fields:
        raise ValueError("component floating node has missing or unconsumed semantic fields")


def render_region(node, types, values):
    """Emit a scalar generic whose operations implement every selected rounding step."""
    name, op, actual = node["name"], node["op"], node["actual_inputs"]
    result = types[name]
    lines = []
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    initial = f"%{name}_empty"
    if op == "matmul":
        maps = [
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
        ]
        iterators = '["parallel", "parallel", "reduction"]'
        lines += [
            f"    %{name}_zero = arith.constant 0.0 : {result.dtype}",
            f"    %{name}_init = linalg.fill ins(%{name}_zero : {result.dtype}) "
            f"outs(%{name}_empty : {result.mlir}) -> {result.mlir}",
        ]
        initial = f"%{name}_init"
    else:
        maps, iterators = [identity] * (len(actual) + 1), '["parallel", "parallel"]'
    operands = ", ".join(values[arg] for arg in actual)
    operand_types = ", ".join(types[arg].mlir for arg in actual)
    block_args = [f"%a{i}: {types[arg].dtype}" for i, arg in enumerate(actual)] + [f"%old: {result.dtype}"]
    lines += [
        f"    %{name} = linalg.generic {{indexing_maps = [{', '.join(maps)}], "
        f"iterator_types = {iterators}}} ins({operands} : {operand_types}) "
        f"outs({initial} : {result.mlir}) {{",
        f"      ^bb{node['ordinal']}({', '.join(block_args)}):",
    ]

    def widen(value, dtype, dest, label):
        if dtype == dest:
            return value
        lines.append(f"        %{label} = arith.extf {value} : {dtype} to {dest}")
        return f"%{label}"

    def narrow(value, dtype, dest, rounding):
        if dtype == dest:
            return value
        if rounding == "rtz":
            lines.extend(
                [
                    f"        %bits = arith.bitcast {value} : f32 to i32",
                    "        %shift = arith.constant 16 : i32",
                    "        %upper = arith.shrui %bits, %shift : i32",
                    "        %short = arith.trunci %upper : i32 to i16",
                    "        %rounded = arith.bitcast %short : i16 to bf16",
                ]
            )
        else:
            lines.append(f"        %rounded = arith.truncf {value} : {dtype} to {dest}")
        return "%rounded"

    if op == "convert":
        value = widen("%a0", types[actual[0]].dtype, "f32", "wide")
        value = narrow(value, "f32", result.dtype, node["rounding"])
    elif op == "relu":
        lines += [
            f"        %zero = arith.constant 0.0 : {result.dtype}",
            f"        %result = arith.maximumf %a0, %zero : {result.dtype}",
        ]
        value = "%result"
    else:
        compute = node["compute_dtype"]
        lhs = widen("%a0", types[actual[0]].dtype, compute, "lhs_wide")
        rhs = widen("%a1", types[actual[1]].dtype, compute, "rhs_wide")
        if op == "matmul":
            lines += [
                f"        %product = arith.mulf {lhs}, {rhs} : {compute}",
                f"        %sum = arith.addf %old, %product : {compute}",
            ]
        else:
            lines.append(f"        %sum = arith.addf {lhs}, {rhs} : {compute}")
        value = narrow("%sum", compute, result.dtype, node["rounding"])
    lines += [f"        linalg.yield {value} : {result.dtype}", f"    }} -> {result.mlir}"]
    return lines
