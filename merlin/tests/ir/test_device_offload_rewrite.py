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


# --------------------------------------------------------------- the on-disk seam a build uses

_TWO_SHAPES = """
module {
  func.func @forward(%a: tensor<16x32xi8>, %b: tensor<32x16xi8>,
                     %c: tensor<8x128xi8>, %d: tensor<128x64xi8>) -> tensor<8x64xi32> {
    %z = arith.constant 0 : i32
    %e1 = tensor.empty() : tensor<16x16xi32>
    %f1 = linalg.fill ins(%z : i32) outs(%e1 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %o1 = linalg.matmul ins(%a, %b : tensor<16x32xi8>, tensor<32x16xi8>)
                        outs(%f1 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %e2 = tensor.empty() : tensor<8x64xi32>
    %f2 = linalg.fill ins(%z : i32) outs(%e2 : tensor<8x64xi32>) -> tensor<8x64xi32>
    %o2 = linalg.matmul ins(%c, %d : tensor<8x128xi8>, tensor<128x64xi8>)
                        outs(%f2 : tensor<8x64xi32>) -> tensor<8x64xi32>
    return %o2 : tensor<8x64xi32>
  }
}
"""


def _rewrite_on_disk(tmp_path, select=lambda _s: True):
    from merlin.llvmlower.device_offload import rewrite_prepared_file
    dev = _int8_device()
    prep = tmp_path / "prepared.mlir"
    prep.write_text(_TWO_SHAPES, encoding="utf-8")
    with IR_LOCK:
        r = rewrite_prepared_file(prep, tmp_path, dev, select=select)
    return r, prep


def test_distinct_extents_get_distinct_callees(tmp_path):
    """MLIR function types are monomorphic, so two shapes cannot share one symbol."""
    r, prep = _rewrite_on_disk(tmp_path)
    assert r.moved == 2 and len(r.signatures) == 2
    assert set(r.signatures.values()) == {(16, 16, 32), (8, 64, 128)}
    text = prep.read_text(encoding="utf-8")
    assert "linalg.matmul" not in text and text.count("func.call") == 2


def test_the_written_module_keeps_its_access_attributes(tmp_path):
    """xDSL prints arg_attrs only for a function WITH a body, so a bodyless declaration loses them
    on the round trip -- and one-shot-bufferize then copies the weight operand of every routed
    contraction. Silent, and a large amount of pointless memcpy in a shipped model."""
    r, prep = _rewrite_on_disk(tmp_path)
    text = prep.read_text(encoding="utf-8")
    assert text.count("bufferization.access") == 3 * len(r.signatures)


def test_the_sidecar_is_written_even_when_nothing_moved(tmp_path):
    """An absent sidecar and an empty one mean different things to a build: 'the rewrite never ran'
    versus 'it ran and routed nothing'."""
    from merlin.llvmlower.device_offload import load_sidecar
    r, _ = _rewrite_on_disk(tmp_path, select=lambda _s: False)
    assert r.moved == 0
    assert load_sidecar(tmp_path).get("signatures") == {}


def test_the_module_is_untouched_when_nothing_moved(tmp_path):
    _r, prep = _rewrite_on_disk(tmp_path, select=lambda _s: False)
    assert prep.read_text(encoding="utf-8").count("linalg.matmul") == 2


# --------------------------------------------------------------- through the runtime dialect

def _text(m):
    from merlin.xdsl_dialects._common import text as to_text
    return to_text(m)


def test_the_offload_is_recorded_as_runtime_ops_before_it_is_realized():
    """Merlin owns a `runtime` dialect that says exactly this -- device.get, a command buffer, an
    append per command, submit -- and no real model passed through it. Now one does."""
    from merlin.llvmlower.device_offload import emit_device_program
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(_TWO_SHAPES)
        chosen = emit_device_program(m, dev, select=lambda _s: True)
        mid = _text(m)
    assert len(chosen) == 2
    assert "runtime.device.get" in mid and "runtime.submit" in mid
    assert mid.count("runtime.command_buffer.append") == 2, "one append per offloaded contraction"


def test_nothing_of_the_dialect_survives_into_the_printed_module():
    """The structural reason this dialect stayed a parallel universe: the module's TEXT goes on to
    upstream mlir-opt, which does not know it. Leaving one op behind fails the whole lowering with an
    error naming a dialect nobody outside this repo has heard of."""
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(_TWO_SHAPES)
        rewrite_contractions_to_device(m, dev, select=lambda _s: True)
        out = _text(m)
    assert "runtime." not in out
    assert out.count("func.call") == 2


def test_passing_through_the_dialect_changes_nothing_about_the_result():
    """The one transport that already worked must pay nothing for the indirection: same calls, same
    signatures, same declarations."""
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(_TWO_SHAPES)
        r = rewrite_contractions_to_device(m, dev, select=lambda _s: True)
        out = _text(m)
    assert r.moved == 2 and len(r.signatures) == 2
    assert set(r.signatures.values()) == {(16, 16, 32), (8, 64, 128)}
    assert "linalg.matmul" not in out and out.count("func.call") == 2
    # The access attributes are NOT asserted here: the printer drops them for a bodyless declaration,
    # which is exactly why the on-disk seam patches them back. That is pinned in
    # `test_the_written_module_keeps_its_access_attributes`, where it is actually observable.


def test_the_lowering_reports_what_it_removed():
    from merlin.llvmlower.device_offload import emit_device_program, lower_device_submits
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        emit_device_program(m, dev, select=lambda _s: True)
        # device.get + create + one append + submit
        assert lower_device_submits(m, dev, transport="host_instruction") == 4
        assert "runtime." not in _text(m)


def test_a_declined_selection_records_no_program():
    from merlin.llvmlower.device_offload import emit_device_program
    dev = _int8_device()
    with IR_LOCK:
        m = _parse(I8_MATMUL)
        assert emit_device_program(m, dev, select=lambda _s: False) == []
        assert "runtime." not in _text(m), "nothing selected means no program to record"
