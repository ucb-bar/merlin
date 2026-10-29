"""Actual verifier-valid memory corruptions cannot authorize compiler delivery."""

import json

import pytest
from buffer_test_inputs import case_root
from xdsl.dialects import arith, func, memref
from xdsl.rewriter import Rewriter

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.verify.compiler_transforms import ALLOCATOR, buffer_check
from merlin.verify.gate_errors import GateRejected, GateUnsupported


@pytest.mark.parametrize("mutation", ["wrong_update", "wrong_initializer"])
def test_actual_memory_mutation_is_refused(tmp_path, tmp_path_factory, pass_gate_tools, mutation):
    base = case_root(tmp_path_factory) / "epoch-cases-v1/two_epochs"
    buffers = [
        (base / name).read_text()
        for name in ("source.mlir", "shape/after.mlir", "memory/middle.mlir", "memory/after.mlir")
    ]
    module = parse_ir(buffers[-1])
    entry = next(op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "kernel")
    if mutation == "wrong_update":
        op = next(op for op in entry.walk() if isinstance(op, arith.AddiOp) and str(op.result.type) == "i32")
        Rewriter.replace_op(op, arith.SubiOp(op.lhs, op.rhs))
    else:
        op = next(op for op in entry.walk() if isinstance(op, memref.StoreOp))
        op.operands = [entry.args[1], *op.operands[1:]]
    module.verify()
    buffers[-1] = ir_text(module)
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "before.mlir").write_text(buffers[0])
    (directory / "after.mlir").write_text(buffers[-1])
    job = {
        "directory": str(directory),
        "before": str(directory / "before.mlir"),
        "after": str(directory / "after.mlir"),
        "params": {
            "intermediates": buffers[1:-1],
            "index_bits": 64,
            "allocator": ALLOCATOR,
            "sdk_bin": str(pass_gate_tools.sdk_bin),
            "kinds": ["shape_view_normalization", "buffer_reuse", "ordered_loop_segmentation"],
        },
    }
    with pytest.raises((GateRejected, GateUnsupported)) as refusal:
        buffer_check(job)
    # Host domain refusals are recorded separately from actual Lean rejection;
    # neither crashes nor infrastructure failures count as successful detection.
    (directory / "mutation-result.json").write_text(
        json.dumps(
            {
                "mutation": mutation,
                "verifier_valid": True,
                "verdict": "not_established" if isinstance(refusal.value, GateRejected) else "unsupported",
                "reason": str(refusal.value),
                "accepted": False,
            },
            indent=2,
        )
    )
