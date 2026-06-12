"""Independent reference recomputation for correctness checking.

The simulator executes the *residency* path (pack a weight once, reuse it). The reference
recomputes each committed output directly from the leaf inputs, bypassing the resident store
entirely. Residency is a performance optimization that must not change results, so the two
must agree -- that equality is the correctness guarantee the execution tests assert.
"""
from __future__ import annotations

from typing import Any

from .commandbuffer import materialize_inputs
from .tensor import Tensor


def reference_outputs(cb: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, list]:
    """Recompute committed outputs from leaf inputs via a naive (non-resident) path."""
    env: dict[str, Tensor] = materialize_inputs(cb, inputs)
    resident_source: dict[str, str] = {}
    matmul_for: dict[str, dict] = {}
    commits: list[dict] = []

    for cmd in cb.get("commands", []):
        op = cmd["opcode"]
        ops = cmd.get("operands", {})
        if op == "RES_PACK":
            resident_source[ops["dst"]] = ops["src"]
        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            matmul_for[ops["dst"]] = cmd
        elif op == "COMMIT":
            commits.append(cmd)

    default_shift = int(cb.get("params", {}).get("requant_shift", 4))
    outputs: dict[str, list] = {}

    for commit in commits:
        ops = commit.get("operands", {})
        attrs = commit.get("attributes", {})
        mm = matmul_for[ops["src"]]
        mops = mm.get("operands", {})
        lhs = env[mops["lhs"]]
        rhs_name = resident_source.get(mops["rhs"], mops["rhs"])  # resolve through the pack
        rhs = env[rhs_name]
        t = lhs.matmul(rhs)
        shift = int(attrs.get("requant_shift", default_shift))
        for stage in attrs.get("epilogue", []):
            if stage in ("bias_add", "bias"):
                bias_name = ops.get("bias")
                if bias_name is not None:
                    t = t.add_bias(env[bias_name])
            elif stage == "requant":
                t = t.requant(shift)
            elif stage == "relu":
                t = t.relu()
        if attrs.get("output_dtype", "i8") == "i8":
            t = t.to_i8()
        outputs[ops["dst"]] = t.to_list()

    return outputs


def outputs_match(a: dict[str, list], b: dict[str, list]) -> bool:
    """Exact equality of two output maps."""
    return a == b
