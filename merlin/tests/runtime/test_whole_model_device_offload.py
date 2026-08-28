"""Offloading contractions to a device from the whole-model build path.

Until now a device could only be reached by the host INTERPRETER: `kernel_backend="mesh"` is valid
only with `run="host"`, which walks the driver function in Python and ships each layer out of process
to a simulator. Nothing was emitted into a host binary, so no compiled artifact could run a model with
accelerator acceleration at all.

These tests pin the seam that changes that, and -- as importantly -- that it is INERT unless a
placement decision was actually made. A build that silently started moving contractions onto a device
would change what every existing image computes.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.device_offload import load_sidecar

_MODEL = """
module {
  func.func @forward(%a: tensor<16x32xi8>, %b: tensor<32x16xi8>) -> tensor<16x16xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<16x16xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<16x16xi32>) -> tensor<16x16xi32>
    %o = linalg.matmul ins(%a, %b : tensor<16x32xi8>, tensor<32x16xi8>)
                       outs(%f : tensor<16x16xi32>) -> tensor<16x16xi32>
    return %o : tensor<16x16xi32>
  }
}
"""


def _routing(**kw):
    return DeviceRouting(device=kw.pop("device", "gemmini"), package_dir="/nonexistent",
                         operand_dtype="int8", accum_dtype="i32", **kw)


def _prepare(tmp_path, device):
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering
    src = tmp_path / "model.mlir"
    src.write_text(_MODEL, encoding="utf-8")
    prepared, _features = prepare_for_lowering(src, tmp_path, blocking=False, device=device)
    return prepared.read_text(encoding="utf-8"), load_sidecar(tmp_path)


def test_no_routing_leaves_the_model_alone(tmp_path):
    """Every existing caller passes nothing here, and must be byte-identical."""
    text, side = _prepare(tmp_path, None)
    assert "linalg.matmul" in text and "func.call" not in text
    assert not side.get("signatures")


def test_a_routing_without_a_decision_is_still_inert(tmp_path):
    """The placement decision is made elsewhere and passed in. A routing that carries none must not
    cause the build to invent one."""
    text, side = _prepare(tmp_path, _routing())
    assert "linalg.matmul" in text and "func.call" not in text
    assert not side.get("signatures")


def test_a_routing_with_a_decision_moves_the_contraction(tmp_path):
    text, side = _prepare(tmp_path, _routing(select=lambda _s: True))
    assert "linalg.matmul" not in text, "the contraction should have become a call"
    assert text.count("func.call") == 1
    assert len(side.get("signatures") or {}) == 1


def test_the_offloaded_declaration_keeps_its_access_attributes(tmp_path):
    """Without these, one-shot-bufferize copies the weight operand of every routed contraction --
    silently, and at real cost in a shipped model."""
    text, _side = _prepare(tmp_path, _routing(select=lambda _s: True))
    assert text.count("bufferization.access") == 3


def test_a_device_that_declares_no_datapath_moves_nothing(tmp_path):
    """Fail closed: an underivable device offloads nothing rather than assuming a precision."""
    text, side = _prepare(tmp_path, _routing(device="definitely_not_a_target",
                                             select=lambda _s: True))
    assert "linalg.matmul" in text
    assert not side.get("signatures")


def test_the_link_refuses_offloaded_symbols_it_cannot_build(tmp_path):
    """A sidecar with signatures and no routing to build them against would link-error far away, so
    the build says which state it is in instead."""
    import inspect

    from merlin.runtime.backends import spike_model
    src = inspect.getsource(spike_model.build)
    assert "were offloaded but no `device=` routing" in src, (
        "the build must refuse offloaded signatures it has no way to build")
