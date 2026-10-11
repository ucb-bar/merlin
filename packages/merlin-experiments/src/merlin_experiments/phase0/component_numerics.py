"""Independent mathematical evaluator of generated integer tensor programs.

This consumes the declared typed program and canonical leaf stimuli, never its
emitted MLIR, device instructions or candidate schedule. Existing golden engines
and their arithmetic remain unchanged.
"""

from __future__ import annotations

from merlin.runtime.tensor import Tensor
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.component_program import analyze


def evaluate(capsule):
    from .component_integer_bounds import preflight_capsule

    preflight_capsule(capsule)
    env = materialize_capsule_leaves(capsule)
    types = capsule["component_program"]
    program = analyze(
        capsule["operation"]["attributes"]["program"],
        operand_dtype=types["selected_storage"]["operand"],
        accumulator_dtype=types["selected_storage"]["accumulator"],
    )
    if program != types:
        raise ValueError("component numerical source differs from builder's typed program")
    for row in program["inputs"]:
        dtype = row["dtype"]
        bits = int(dtype[1:])
        if any(
            type(value) not in (int, float)
            or int(value) != value
            or not -(1 << (bits - 1)) <= value < (1 << (bits - 1))
            for value in env[row["name"]].data
        ):
            raise ValueError("component leaf stimulus exceeds its declared signed storage format")
    for node in program["nodes"]:
        name, op = node["name"], node["op"]
        args = [env[arg] for arg in node["actual_inputs"]]
        if op == "matmul":
            output = args[0].matmul(args[1])
        elif op in {"add", "update"}:
            output = Tensor(
                tuple(node["shape"]),
                [int(a) + int(b) for a, b in zip(args[0].data, args[1].data, strict=True)],
                node["dtype"],
            )
        elif op == "transpose":
            rows, cols = args[0].shape
            output = Tensor(
                (cols, rows),
                [args[0].data[row * cols + col] for col in range(cols) for row in range(rows)],
                node["dtype"],
            )
        elif op == "reduce_sum":
            rows, cols = args[0].shape
            # Mathematical sums followed by the independent signed-width
            # projection below; no emitted loop or contraction is consulted.
            values = (
                [sum(int(args[0].data[row * cols + col]) for row in range(rows)) for col in range(cols)]
                if node["axis"] == 0
                else [sum(int(args[0].data[row * cols + col]) for col in range(cols)) for row in range(rows)]
            )
            output = Tensor(tuple(node["shape"]), values, node["dtype"])
        else:
            output = Tensor(tuple(node["shape"]), list(args[0].data), node["dtype"])
        # Standard unflagged MLIR integer operations have modular width. This
        # exact mathematical reduction followed by modular projection agrees
        # with every partial modular reduction without sharing lowering code.
        bits = int(node["dtype"][1:])
        modulus, sign = 1 << bits, 1 << (bits - 1)
        env[name] = Tensor(
            tuple(node["shape"]), [(int(value) + sign) % modulus - sign for value in output.data], node["dtype"]
        )
    return {row["name"]: env[row["actual_value"]].to_list() for row in program["outputs"]}
