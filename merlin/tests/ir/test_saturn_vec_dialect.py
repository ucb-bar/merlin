"""SV2: the isolated saturn_vec xDSL dialect + lowering descends to the certified command buffer.

The vector family has its own dialect (vector.map / vector.reduce), in an isolated package
(artifacts/targets/saturn_vec/hand_v0/), loaded dynamically — not in the core tree. The dialect
verifies, round-trips, and lowers to the target-independent command buffer, which then runs
through the Merlin reference/simulator (and, gated, merlin's MLIR→LLVM host compiler).
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import importlib.util
from pathlib import Path

import pytest

from merlin.runtime import outputs_match, reference_outputs, simulate

REPO = repo_root()
PKG = REPO / "out/artifacts/targets/saturn_vec/hand_v0"

try:
    from merlin.xdsl_dialects import _common
    _HAS_XDSL = _common.HAS_XDSL
except Exception:  # pragma: no cover
    _HAS_XDSL = False


def _load_dialect():
    import sys
    spec = importlib.util.spec_from_file_location("saturn_vec_dialect", PKG / "dialect.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["saturn_vec_dialect"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _HAS_XDSL, reason="xDSL not installed")
@pytest.mark.skipif(not PKG.is_dir(), reason="saturn_vec package not present")
def test_vector_dialect_verifies_and_lowers_to_cb():
    """The vector dialect builds + verifies, and lowers to a cb that the reference/sim agree on."""
    vd = _load_dialect()
    module = vd.build_example(n=64)
    module.verify()
    target_ops = {op.name for op in module.walk()} - {"builtin.module", "func.func", "func.return"}
    assert target_ops == {"saturn_vec.map", "saturn_vec.reduce"}

    cb = vd.lower_to_command_buffer(module)
    assert cb["target"] == "saturn_vec"
    assert {c["opcode"] for c in cb["commands"]} == {"VECTOR_MAP", "VREDUCE"}
    # the lowered cb is a real dot product: s = sum(x * w), and reference == simulator
    sim = simulate(cb)["outputs"]
    ref = reference_outputs(cb)
    assert outputs_match(sim, ref)
    leaves_match = ref == {"s": [sum(a * b for a, b in zip(
        __import__("merlin.runtime.commandbuffer", fromlist=["materialize_inputs"])
        .materialize_inputs(cb)["x"].data,
        __import__("merlin.runtime.commandbuffer", fromlist=["materialize_inputs"])
        .materialize_inputs(cb)["w"].data))]}
    assert leaves_match


@pytest.mark.skipif(not _HAS_XDSL, reason="xDSL not installed")
@pytest.mark.skipif(not PKG.is_dir(), reason="saturn_vec package not present")
def test_vector_dialect_cb_certifies_on_merlin_compiler():
    """The cb lowered from the isolated dialect certifies through merlin's MLIR→LLVM compiler."""
    try:
        from merlin.llvmlower import toolchain
        if not toolchain.available():
            pytest.skip("merlin MLIR→LLVM toolchain unavailable")
    except Exception:
        pytest.skip("toolchain probe failed")
    # saturn_vec was evicted to its own reference package; reach its MLIR submodule via the registry.
    from merlin.runtime.backends import base as _base
    vm = _base.get_backend("saturn_vec").saturn_vec_mlir
    vd = _load_dialect()  # load ONCE — two loads create distinct op classes (isinstance breaks)
    cb = vd.lower_to_command_buffer(vd.build_example(n=64))
    res = vm.run_host(cb)
    assert res["correct"] is True
