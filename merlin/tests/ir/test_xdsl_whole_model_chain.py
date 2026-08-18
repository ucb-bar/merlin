"""Whole-model backbone: a multi-layer matmul CHAIN (each layer's output feeds the next)
compiles through the SAME staged pipeline as the MVP and executes correctly on the engine.

This is the whole-model / section compiler entry (`lower_module`): the descent
input -> contract -> schedule -> interface -> target -> runtime -> command buffer must thread
chained intermediates, place every weight resident (not just reused ones), commit in float, and
surface EXACTLY the function's returned tensor — not every intermediate commit.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def _lower_chain(dims, elem="f32"):
    from merlin.xdsl_dialects.lowering import lower_module
    from merlin.xdsl_dialects.lowering.input_workload import build_matmul_chain

    return lower_module(build_matmul_chain(dims=dims, elem=elem))


def test_two_layer_chain_every_stage_verifies():
    res = _lower_chain((8, 16, 12, 6))
    for mod in res.modules():
        mod.verify()


def test_chain_command_buffer_surfaces_only_final_output():
    res = _lower_chain((8, 16, 12, 6))
    cb = res.command_buffer
    # Two chained matmuls => two commits Y0 (intermediate) and Y1 (result). Only Y1 is declared.
    assert cb.get("outputs") == ["Y1"]
    assert cb["tensors"]["Y1"]["role"] == "output"
    dsts = [c["operands"]["dst"] for c in cb["commands"] if c["opcode"] == "COMMIT"]
    assert dsts == ["Y0", "Y1"]
    # The chain is real: layer-2's matmul consumes layer-1's committed output as its LHS.
    mm_lhs = [c["operands"]["lhs"] for c in cb["commands"]
              if c["opcode"] in ("MATMUL", "MATMUL_RESIDENT")]
    assert "Y0" in mm_lhs


def test_chain_executes_and_matches_numpy():
    from merlin.xdsl_dialects.lowering import execute

    res = _lower_chain((8, 16, 12, 6))
    assert execute(res)["correct"] is True

    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 16)).astype(np.float32)
    W1 = rng.standard_normal((16, 12)).astype(np.float32)
    W2 = rng.standard_normal((12, 6)).astype(np.float32)
    # leaf tensor names: activation A0, weights W (layer 1) and W1 (layer 2)
    inj = {"A0": A.tolist(), "W": W1.tolist(), "W1": W2.tolist()}
    run = execute(res, inj)
    got = np.array(next(iter(run["outputs"].values())), dtype=np.float32)
    assert run["correct"] is True
    assert np.allclose(got, A @ W1 @ W2, rtol=1e-4, atol=1e-3)


def test_three_layer_chain_matches_numpy():
    from merlin.xdsl_dialects.lowering import execute

    res = _lower_chain((4, 8, 6, 5, 3))
    cb = res.command_buffer
    assert cb.get("outputs") == ["Y2"]  # three matmuls, only the last is the result
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    Ws = [rng.standard_normal(s).astype(np.float32) for s in [(8, 6), (6, 5), (5, 3)]]
    inj = {"A0": A.tolist(), "W": Ws[0].tolist(), "W1": Ws[1].tolist(), "W2": Ws[2].tolist()}
    run = execute(res, inj)
    got = np.array(next(iter(run["outputs"].values())), dtype=np.float32)
    assert run["correct"] is True
    assert np.allclose(got, A @ Ws[0] @ Ws[1] @ Ws[2], rtol=1e-4, atol=1e-3)


def test_single_layer_still_works():
    """A one-matmul 'chain' (degenerate) compiles and matches — the single-use weight is packed."""
    from merlin.xdsl_dialects.lowering import execute

    res = _lower_chain((4, 5, 3))
    assert res.command_buffer.get("outputs") == ["Y0"]
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 5)).astype(np.float32)
    W = rng.standard_normal((5, 3)).astype(np.float32)
    run = execute(res, {"A0": A.tolist(), "W": W.tolist()})
    got = np.array(next(iter(run["outputs"].values())), dtype=np.float32)
    assert run["correct"] is True
    assert np.allclose(got, A @ W, rtol=1e-4, atol=1e-3)
