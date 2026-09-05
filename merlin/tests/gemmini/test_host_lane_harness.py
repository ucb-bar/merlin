"""A program placed entirely on the HOST LANE has a harness, and a float result reaches the console.

Ten shipped capsules (``SY_host_lane_*``, ``SY_host_only_normalization``,
``GN0_layernorm_host_only_bf16_pt``) declare families/dtypes this datapath admits no lowering for, so a
correct compiler routes every region to the CPU lane and emits a CPU-lane program. Until this existed the
right answer was not deliverable:

* there was no harness for a command-less buffer at all -- the tiled renderer REFUSES one
  ("expected RES_PACK(s) + matmuls==commits>=1, got 0/0/0") and the movement renderer needs a movement
  command, so a compiler had to fabricate a matmul it never ran just to get its buffers allocated;
* every destination wider than i8 was declared ``int32_t`` regardless of the dtype's real width, so a
  2-byte bf16 store was read back at a 4-byte stride;
* a float destination was refused outright for having no "integer" spelling.

The fix is one derived rule -- a buffer is laid out and printed at its DECLARED dtype's own storage
width, a float one as its stored bit pattern -- so these tests pin the widths, the refusals, and the
byte-identity of every integer form that already worked.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from merlin.runtime.backends import base as bk

gem = bk.get_backend("gemmini")
CG = sys.modules[gem.__package__ + ".gemmini_codegen_mlir"]
GB = importlib.import_module(gem.__package__ + ".gemmini")   # the backend module behind the package
DIM = 16


def _host_lane_cb(rows=2, cols=4, in_dtype="f32", out_dtype="f32", values=None):
    cb = {"abi_version": "0.1", "target": "gemmini", "backend": "test",
          "tensors": {"arg0": {"shape": [rows, cols], "dtype": in_dtype, "role": "input"},
                      "Y0": {"shape": [rows, cols], "dtype": out_dtype, "role": "output"}},
          "commands": [], "params": {}}
    if values is not None:
        cb["canonical_inputs"] = {"arg0": {"shape": [rows, cols], "values": list(values)}}
    return cb


def _decl(text: str, name: str) -> str:
    hits = [ln.strip() for ln in text.splitlines()
            if ln.strip().startswith("static") and f"T_{name}[" in ln]
    assert len(hits) == 1, f"expected one declaration of T_{name}, got {hits}"
    return hits[0]


# --- the missing harness ---------------------------------------------------------------------------
def test_a_command_less_buffer_now_renders_a_harness():
    """The whole gap: a program with no accelerator command is a placement decision, not a refusal."""
    c = gem.render_harness(_host_lane_cb(), target="gemmini")
    assert "gemmini_kernel((void*)T_arg0, (void*)T_Y0);" in c
    assert 'printf("OUT Y0 2 4")' in c
    assert 'printf("DONE\\n")' in c


def test_the_pointer_abi_is_the_declaration_order():
    """Derived from the buffer's own tensor table, which is what a positional binder on the emitting
    side produces -- not from roles, and not from a guessed convention."""
    cb = _host_lane_cb()
    cb["tensors"]["W"] = {"shape": [2, 4], "dtype": "f32", "role": "weight"}
    c = gem.render_harness(cb, target="gemmini")
    assert "gemmini_kernel((void*)T_arg0, (void*)T_Y0, (void*)T_W);" in c


def test_a_command_less_buffer_with_no_output_is_still_refused():
    """A program that declares nothing to read back is not a host-lane program; it is a buffer with no
    result, and it is refused rather than harnessed into a run that prints no OUT line."""
    cb = _host_lane_cb()
    cb["tensors"]["Y0"]["role"] = "input"
    assert not GB._is_host_lane_cb(cb)
    with pytest.raises(Exception):
        gem.render_harness(cb, target="gemmini")


def test_the_calibration_buffer_is_not_mistaken_for_a_host_lane_program():
    """Zero tensors AND zero commands is the explicit calibration input; it has no output to read back
    and keeps going to the tiled path."""
    assert not GB._is_host_lane_cb({"tensors": {}, "commands": []})
    assert GB._is_host_lane_cb(_host_lane_cb())
    assert not GB._is_host_lane_cb(
        {"tensors": {"Y0": {"shape": [2, 2], "dtype": "i8", "role": "output"}},
         "commands": [{"opcode": "COMMIT", "operands": {"src": "a", "dst": "Y0"}}]})


# --- buffer widths, derived from the declared dtype ------------------------------------------------
@pytest.mark.parametrize("dtype,ctype,elems", [
    ("i8", "elem_t", DIM * DIM), ("i32", "int32_t", DIM * DIM), ("i16", "int16_t", DIM * DIM),
    ("f32", "uint32_t", DIM * DIM), ("bf16", "uint16_t", DIM * DIM), ("f16", "uint16_t", DIM * DIM),
])
def test_the_destination_is_sized_at_the_declared_width(dtype, ctype, elems):
    """A 2-byte result gets a 2-byte container. Declaring every non-i8 output ``int32_t`` is what read
    a bf16 store back at a 4-byte stride -- half the row, and the wrong half."""
    d = _decl(gem.render_harness(_host_lane_cb(out_dtype=dtype), target="gemmini"), "Y0")
    assert f"static {ctype} T_Y0[{elems}]" in d


@pytest.mark.parametrize("dtype", ["i1", "mxfp4", "not_a_dtype"])
def test_a_width_this_harness_cannot_lay_out_is_refused(dtype):
    """FAIL CLOSED. A sub-byte format has no standalone element and an unregistered spelling has no
    width at all; guessing one mis-strides the whole buffer."""
    with pytest.raises(Exception):
        CG.container_for(dtype)


def test_a_float_output_prints_its_pattern_unsigned():
    """Unsigned so a top-bit-set pattern reaches the console as the pattern, not through an
    implementation-defined signed conversion."""
    c = gem.render_harness(_host_lane_cb(out_dtype="f32"), target="gemmini")
    assert 'printf(" %u", (unsigned)T_Y0[' in c
    d = gem.render_harness(_host_lane_cb(out_dtype="i32"), target="gemmini")
    assert 'printf(" %d", (int)T_Y0[' in d


# --- the embedded operands -------------------------------------------------------------------------
def test_a_float_leaf_is_embedded_as_its_code_pattern():
    """The device has to see the operand the golden used. A float value written through ``int(v)``
    into an ``elem_t`` array (what every leaf used to get) truncates it to a byte of nonsense."""
    from merlin.runtime import fp8_formats as ff
    values = [-1.5, 0.25, 3.0, -0.125, 7.5, 2.0, -0.5, 1.0]
    c = gem.render_harness(_host_lane_cb(values=values), target="gemmini",
                           inputs={"arg0": values})
    d = _decl(c, "arg0")
    assert d.startswith("static const uint32_t T_arg0[")
    for code in ff.float_to_codes(values, "f32"):
        assert str(int(code)) in d


def test_an_i8_leaf_is_embedded_exactly_as_before():
    c = gem.render_harness(_host_lane_cb(in_dtype="i8", out_dtype="i8"), target="gemmini")
    assert _decl(c, "arg0").startswith("static const elem_t T_arg0[")


def test_a_rank_1_leaf_is_one_row_not_a_refusal():
    """A layernorm's per-channel weight and bias are rank-1, and a host-lane program carries them as
    ordinary operands. Sizing them through the whole-op path's rank >= 2 rule failed two shipped
    capsules with "needs a positive rank >= 2 shape" -- a harness limit worded as a defect in the
    submission that declared a perfectly ordinary vector."""
    cb = _host_lane_cb(rows=16, cols=32)
    cb["tensors"]["W"] = {"shape": [32], "dtype": "f32", "role": "input"}
    cb["tensors"]["B"] = {"shape": [32], "dtype": "f32", "role": "input"}
    c = gem.render_harness(cb, target="gemmini")
    assert "gemmini_kernel((void*)T_arg0, (void*)T_Y0, (void*)T_W, (void*)T_B);" in c
    # rank-1 [32] is ONE row of 32: the row pitch the kernel indexes with is the column extent, so
    # its elements sit at 0..31 exactly as a leading-dims-multiply-into-rows split puts them.
    assert _decl(c, "W").startswith("static const uint32_t T_W[")


@pytest.mark.parametrize("shape", [[], [0, 4], [4, -1], "notalist"])
def test_a_shape_that_is_not_a_positive_extent_list_is_refused(shape):
    """Still fail-closed on a shape that is not extents at all — only the rank floor was wrong."""
    cb = _host_lane_cb()
    cb["tensors"]["W"] = {"shape": shape, "dtype": "f32", "role": "input"}
    with pytest.raises(Exception, match="non-empty shape of positive extents"):
        gem.render_harness(cb, target="gemmini")


def test_rows_are_padded_to_the_tile_edge():
    """Same layout every other harness here uses, so a kernel written against the declared ABI indexes
    one pitch whichever shape it is handed."""
    c = gem.render_harness(_host_lane_cb(rows=2, cols=4), target="gemmini")
    assert f"T_Y0[{DIM * DIM}]" in c
    assert f"j++) printf(\" %u\", (unsigned)T_Y0[i * {DIM} + j]);" in c


# --- non-regression: the integer paths that already worked -----------------------------------------
def _movement_cb(m, n, out_dtype):
    return {"abi_version": "0.1", "target": "gemmini",
            "tensors": {"X": {"shape": [m, n], "dtype": "i8", "role": "input"},
                        "Y0": {"shape": [m, n], "dtype": out_dtype, "role": "output"}},
            "commands": [{"opcode": "VECTOR_MAP", "operands": {"lhs": "X", "rhs": "X", "dst": "Y0"},
                          "attributes": {"combine": "identity", "activation": [],
                                         "output_dtype": out_dtype}}]}


def test_the_movement_harness_is_unchanged_for_the_shipped_integer_capsules():
    """The two shipped tail capsules' shapes, spelled exactly as they were before the widths became
    derived -- an i8 source in ``elem_t``, an i32 destination in ``int32_t``, printed with ``%d``."""
    for m, n, cells in ((15, 15, 256), (17, 15, 512)):
        c = gem.render_harness(_movement_cb(m, n, "i32"), target="gemmini")
        assert _decl(c, "X").startswith("static const elem_t T_X[")
        assert f"static int32_t T_Y0[{cells}] row_align_acc(1);" in c
        assert f'printf(" %d", (int)T_Y0[i * {DIM} + j]);' in c


def test_a_movement_destination_can_now_be_a_float():
    """The refusal this removes: a float destination had "no readback encoding" only because the
    harness had no container for it, not because the console could not carry the result."""
    cb = _movement_cb(16, 16, "f32")
    cb["tensors"]["X"]["dtype"] = "f32"
    c = gem.render_harness(cb, target="gemmini")
    assert "static uint32_t T_Y0[256] row_align_acc(1);" in c
    assert _decl(c, "X").startswith("static const uint32_t T_X[")
