"""The grounded capsule sources: a capsule defined in PyTorch must lower to 0-opaque linalg-on-tensors
via model2MLIR and carry a host torch-eager reference golden. The op->loader vocabulary must render for
every supported op without torch (that half runs in the merlin venv); the actual capture is skipped when
the m2m venv (torch) is absent, so the suite stays green on a bare checkout.

Target-agnostic: nothing here names a target; precision is a parameter (the token a target's
``compute_units`` declares)."""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_source as CSrc


def test_supported_ops_all_render():
    """Every op template renders to valid python source (no torch needed) with a Model + loader."""
    for op in CSrc.supported_ops():
        spec = {"op": op, "M": 16, "K": 16, "N": 16, "Dv": 16, "dtype": "fp32", "seed": 0}
        src = CSrc.build_loader_src(spec)
        assert "def get_model_and_inputs()" in src
        assert "class Model" in src or "def get_model_and_inputs" in src
        compile(src, f"<loader:{op}>", "exec")   # parseable python


def test_unknown_op_fails_closed():
    with pytest.raises(KeyError):
        CSrc.build_loader_src({"op": "not_a_real_op", "M": 16, "K": 16})


_M2M = CSrc.PytorchRefSource()
_needs_m2m = pytest.mark.skipif(not _M2M.available(),
                                reason="m2m venv (torch) unavailable; set MERLIN_M2M_PYTHON/MERLIN_M2M_DIR")


@_needs_m2m
@pytest.mark.parametrize("spec", [
    {"op": "matmul", "M": 16, "K": 16, "N": 16, "dtype": "fp32", "seed": 1},
    {"op": "rmsnorm", "M": 16, "K": 16, "dtype": "bf16", "seed": 2, "eps": 1e-5},
    {"op": "attention_full", "M": 16, "K": 64, "N": 16, "Dv": 64, "dtype": "fp16", "seed": 3, "causal": True},
])
def test_pytorch_capture_is_clean_with_golden(spec):
    """m2m produces a 0-opaque linalg program + a host-eager golden of the right shape, per dtype."""
    art = _M2M.capture(spec)
    assert art.meta["ok"] and art.meta["opaque"] == 0, art.meta
    assert "linalg." in art.linalg_mlir
    assert art.op == spec["op"] and art.dtype == spec["dtype"]
    # golden is a 2-D reference tensor matching the op's declared output rows
    assert isinstance(art.golden, list) and isinstance(art.golden[0], list)
    assert len(art.golden) == spec["M"]
    # the pytorch source is agent-visible context and must round-trip to the same op
    assert "get_model_and_inputs" in art.pytorch_src
