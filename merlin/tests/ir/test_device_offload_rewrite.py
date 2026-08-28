"""Moving a contraction onto a device, as a call a compiled host program can make.

The matrix-unit path already proved this shape works end to end -- call a private symbol, record the
minted signatures in a sidecar, let host and device objects meet in one archive. What it cannot do is
serve a second device: its symbol stem, its dtype legality and its operand types are literals for one
unit. These tests pin the generalization, and specifically the three ways it could go wrong quietly:

  * a device inheriting ANOTHER device's datapath (the float/integer confusion a literal cannot avoid),
  * two devices minting the same symbol (a duplicate-symbol link error, far from the cause),
  * a contraction moved with no decision behind it (which would duplicate the placement decision and
    then disagree with it).
"""
from __future__ import annotations

import json

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.llvmlower.device_offload import (SIDECAR_NAME, load_sidecar,
                                             rewrite_contractions_to_device, symbol_stem)
from merlin.system.offload import device_dtype_triples

I8_MATMUL = """
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

F32_MATMUL = I8_MATMUL.replace("i8", "f32").replace("i32", "f32").replace(
    "arith.constant 0 : f32", "arith.constant 0.0 : f32")


def _int8_device():
    for name in ("gemmini", "saturn_opu_mxv256d128"):
        if ("i8", "i8", "i32") in device_dtype_triples(name):
            return name
    pytest.skip("no integer-datapath device derivable in this checkout")


def _parse(src):
    return mq.parse(src)


# --------------------------------------------------------------- nothing moves without a decision

def test_no_selector_moves_nothing():
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        r = rewrite_contractions_to_device(m, dev)
        assert r.moved == 0 and r.skipped
        assert mq.op_count(m, "linalg.matmul") == 1, "the module must be untouched"


def test_a_selector_that_declines_everything_moves_nothing():
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        r = rewrite_contractions_to_device(m, dev, select=lambda _s: False)
        assert r.moved == 0 and mq.op_count(m, "linalg.matmul") == 1


# --------------------------------------------------------------- the rewrite itself

def test_a_selected_contraction_becomes_a_call_to_a_private_symbol():
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        r = rewrite_contractions_to_device(m, dev, select=lambda _s: True)
        assert r.moved == 1, f"nothing moved: {r.skipped}"
        assert mq.op_count(m, "linalg.matmul") == 0, "the contraction must be gone"
        assert mq.op_count(m, "func.call") == 1
        sym = r.routed[0].symbol
        assert sym.startswith(symbol_stem(dev))
        assert sym in r.signatures and r.signatures[sym] == (16, 16, 32)
        assert r.routed[0].dtypes == ("i8", "i8", "i32")


def test_the_callee_is_declared_so_the_module_still_verifies():
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        r = rewrite_contractions_to_device(m, dev, select=lambda _s: True)
        decls = [f for f in mq.walk(m, "func.func")]
        assert any(mq.attr_str(f, "sym_name") == r.routed[0].symbol for f in decls), \
            "the call has no declaration; the module would not verify"


# --------------------------------------------------------------- the quiet failures

def test_a_device_does_not_inherit_another_devices_datapath():
    """The failure a hardcoded triple cannot avoid: an integer device taking an f32 contraction."""
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(F32_MATMUL)
        r = rewrite_contractions_to_device(m, dev, select=lambda _s: True)
        assert r.moved == 0, "an f32 contraction must not move onto an integer datapath"
        assert mq.op_count(m, "linalg.matmul") == 1


def test_two_devices_mint_different_symbols():
    """A shared stem collides at link time, and the only diagnostic is a duplicate symbol."""
    assert symbol_stem("alpha") != symbol_stem("beta")
    assert symbol_stem("a-b") == symbol_stem("a_b"), "punctuation must not survive into a symbol"


def test_an_underivable_device_declines_with_a_reason():
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        r = rewrite_contractions_to_device(m, "definitely_not_a_target", select=lambda _s: True)
        assert r.moved == 0
        assert any("no derivable datapath" in why for _, why in r.skipped)
        assert mq.op_count(m, "linalg.matmul") == 1


# --------------------------------------------------------------- the sidecar

def test_the_sidecar_carries_what_the_build_step_needs(tmp_path):
    """The rewrite runs inside the lowering subprocess; the build step that generates the callee runs
    outside it. An in-memory hand-off silently produced an empty signature set."""
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        r = rewrite_contractions_to_device(m, dev, select=lambda _s: True, sidecar_dir=tmp_path)
    assert (tmp_path / SIDECAR_NAME).is_file()
    back = load_sidecar(tmp_path)
    assert back["device"] == dev
    assert back["signatures"][r.routed[0].symbol] == [16, 16, 32]
    assert back["routed"][0]["dtypes"] == ["i8", "i8", "i32"]
    assert json.dumps(back)                       # round-trips as plain JSON


def test_an_absent_sidecar_reads_as_empty_not_an_error():
    assert load_sidecar("/nonexistent/path/for/a/sidecar") == {}
