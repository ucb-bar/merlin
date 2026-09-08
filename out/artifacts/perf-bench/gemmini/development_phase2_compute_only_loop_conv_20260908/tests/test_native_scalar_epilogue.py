"""Target-neutral probes for exact, fused host epilogue scalar operations."""
from __future__ import annotations

from pathlib import Path

import pytest

from mlir_oot.gemmini_opt import Pipeline, _print


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "tests/fixtures/pointwise_epilogue_template.mlir").read_text()


def _source(m: int, k: int, n: int) -> str:
    return TEMPLATE.replace("@M@", str(m)).replace("@K@", str(k)).replace("@N@", str(n))


@pytest.mark.parametrize("m,k,n", [
    (1, 3, 1),       # unit axes and one channel
    (15, 17, 7),     # sub-tile output with K and channel tails
    (17, 31, 19),    # tails across every mesh axis
    (2, 5, 33),      # batched rows and a multi-tile channel axis
])
def test_affine_bias_residual_relu_quantize_uses_standard_scalar_ops(
        m: int, k: int, n: int) -> None:
    pipe = Pipeline(_source(m, k, n), enable_source_conv=True).run()
    assert pipe.declined is None
    assert pipe.plan is not None
    assert pipe.artifact is not None
    target = _print(pipe.artifact)
    tasks = pipe.plan.command_buffer["params"]["global_program_plan"]["tasks"]

    assert [task["kind"] for task in tasks] == ["contraction", "host"]
    assert target.count('"llvm.intr.roundeven"') == 1
    assert target.count('"llvm.fptosi"') == 1
    # The chain remains one host traversal: producer pointwise values are lazy
    # and the final quantize writes directly to its boundary buffer.
    storage = pipe.plan.command_buffer["params"]["host_storage"]
    tensor_owners = [row for row in storage["largest_owners"]
                     if row["operation"] == "linalg.generic"]
    assert tensor_owners == []


def test_i1_has_an_explicit_byte_storage_cost() -> None:
    from mlir_oot.lowering.plan import DTYPE_BYTES
    assert DTYPE_BYTES["i1"] == 1
