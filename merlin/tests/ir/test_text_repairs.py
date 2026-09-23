"""The pre-parse text repairs, pinned against the REAL malformed input each one exists for.

Every repair in this file operates on text that does not parse yet — that is why it is
textual rather than structural. Each test therefore quotes an actual line taken from an
artifact on disk (the provenance is in the docstring), so a future rewrite has something
to be byte-identical against rather than a plausible-looking invention.
"""
from __future__ import annotations

from merlin.baselines.buddy import _repair_malformed_select_slices
from merlin.frontends.linalg_mlir import strip_paren_results
from merlin.llvmlower.passes_xdsl import preprocess_text_textual
from merlin.llvmlower.pipeline import _fix_f0x_literals
from merlin.targetgen.rtl.gen_iface_irdl import _fix_string_base_sigil

# --- pipeline: MLIR's `f0x` float literal, which LLVM's textual parser has no rule for ------
# Real lines from out/runs/.../GF5_gelu_bf16_pt/generated/model.ll (a bf16 gelu capsule).
F0X_LL = """  %9 = fmul float f0x3F4C422A, %8
  %31 = fadd float %11, f0x2B8CBCCC
"""


def test_f0x_f32_literal_is_widened_to_a_double_bit_pattern():
    fixed = _fix_f0x_literals(F0X_LL)
    # 0x3F4C422A as an f32 is 0.798..., whose double bit pattern is 0x3FE9884540000000.
    assert "%9 = fmul float 0x3FE9884540000000, %8" in fixed
    assert "f0x" not in fixed


def test_f0x_f64_payload_only_sheds_the_f():
    assert _fix_f0x_literals("double f0x3FF0000000000000") == "double 0x3FF0000000000000"


def test_f0x_with_an_unknown_payload_width_is_left_for_llvm_to_reject():
    """Fail closed: 12 hex digits is neither f32 nor f64, so it is NOT half-rewritten into a
    plausible constant — it reaches the LLVM parser verbatim and fails there."""
    assert _fix_f0x_literals("float f0x3F4C422A3F4C") == "float f0x3F4C422A3F4C"
    # An identifier character before `f0x` means this is the tail of a longer token,
    # not a literal (`%` is not one, so `%f0x…` IS a literal and does get rewritten).
    assert _fix_f0x_literals("named_f0x3F4C422A") == "named_f0x3F4C422A"


# --- linalg_mlir: the parenthesized multi-result linalg terminator xDSL rejects -------------
# Real line from out/runs/rvv/beam/matmul/.../generated/v/model.prepared.mlir.
def test_paren_multi_result_terminator_loses_its_parens():
    assert (strip_paren_results("    } -> (tensor<1x32xf32>, tensor<1x32xi64>)")
            == "    } -> tensor<1x32xf32>, tensor<1x32xi64>")


def test_single_result_terminator_is_untouched():
    line = "    } -> tensor<1x32xf32>"
    assert strip_paren_results(line) == line


def test_nested_parens_after_the_arrow_are_left_alone():
    """Not a shape this normalizer understands — it reaches the parser instead of being mangled."""
    line = "} -> ((tensor<4xf32>))"
    assert strip_paren_results(line) == line


# --- passes_xdsl: model2MLIR's invalid whole-model text ------------------------------------
# Real op from out/artifacts/recaptures/gemma2_2b_int8_full/model.mlir.
DEQUANT = ('    %1121 = "quant_ext.dequantize_per_channel"(%1115, %2, %1120) '
           '<{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize"} : '
           '(tensor<2304x2048xi8>, tensor<2048xf32>, tensor<2048xi32>) -> tensor<2304x2048xf32>')
# Real op shape from a bitvla capture: sizes carry only the RESULT rank, offsets/strides the source's.
RANK_REDUCED_SLICE = ('    %58 = "tensor.extract_slice"(%57) <{static_offsets = '
                      'array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, static_strides = '
                      'array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> '
                      '{prov.op = "select"} : (tensor<1x1x32x32xf32>) -> tensor<32xf32>')


def test_dequant_becomes_a_pure_upstream_linalg_generic():
    out, stats = preprocess_text_textual(DEQUANT)
    assert stats["dequantize_lowered"] == 1
    assert "quant_ext" not in out
    assert "%dq_init_1121 = tensor.empty() : tensor<2304x2048xf32>" in out
    # axis = 1 -> the scale/zero-point operands project onto d1 of a rank-2 output.
    assert "affine_map<(d0, d1) -> (d1)>" in out
    assert "ins(%1115, %2, %1120 : tensor<2304x2048xi8>, tensor<2048xf32>, tensor<2048xi32>)" in out


def test_rank_reduced_extract_slice_sizes_are_padded_to_the_source_rank():
    out, _ = preprocess_text_textual(RANK_REDUCED_SLICE)
    assert "static_sizes = array<i64: 1, 1, 1, 32>" in out
    assert "static_offsets = array<i64: 0, 0, 31, 0>" in out      # untouched
    assert '(tensor<1x1x32x32xf32>) -> tensor<32xf32>' in out     # signature untouched


def test_insert_slice_stride_overrunning_the_destination_is_reset_to_one():
    """m2m's slice_scatter decomposition read `step` from the `end` slot, so a stride of 99
    landed on a destination of extent 14. Only the overrunning stride is reset."""
    op = ('    %396 = "tensor.insert_slice"(%395, %393) <{static_offsets = array<i64: 0, 0>, '
          'static_sizes = array<i64: 1, 14>, static_strides = array<i64: 1, 99>, '
          'operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.op = "x"} : '
          '(tensor<1x14xf32>, tensor<1x14xf32>)')
    out, _ = preprocess_text_textual(op)
    assert "static_strides = array<i64: 1, 1>" in out


def test_well_formed_slices_are_returned_byte_for_byte():
    op = ('    %690 = "tensor.extract_slice"(%689) <{static_offsets = array<i64: 0, 0>, '
          'static_sizes = array<i64: 1, 8>, static_strides = array<i64: 1, 1>}> : '
          '(tensor<1x9xi64>) -> tensor<1x8xi64>')
    assert preprocess_text_textual(op)[0] == op


def test_only_the_first_func_gets_the_c_interface_attribute():
    text = ("func.func @forward(%a: tensor<4xf32>) -> tensor<4xf32> {\n}\n"
            "func.func @other(%a: tensor<4xf32>) -> tensor<4xf32> {\n}\n")
    out, stats = preprocess_text_textual(text)
    assert stats["c_interface_funcs"] == 1
    assert out.count("llvm.emit_c_interface") == 1
    assert out.startswith("func.func @forward(%a: tensor<4xf32>) -> tensor<4xf32> "
                          "attributes {llvm.emit_c_interface} {")


# --- buddy: the m2m aten.select export bug, repaired post-tool and pre-reparse --------------
def test_buddy_reconstructs_rank_r_sizes_for_a_selected_dim():
    """Real op from .../dse_guidance/recaptures/bitvla/model.mlir: offset 31 on dim 2 leaves
    only one element there, so dim 3 (extent 32, offset 0) is the kept dim."""
    op = ('static_offsets = array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, '
          'static_strides = array<i64: 1, 1, 1, 1>, x}> : (tensor<1x1x32x32xf32>) -> tensor<32xf32>')
    fixed, n = _repair_malformed_select_slices(op)
    assert n == 1
    assert "static_sizes = array<i64: 1, 1, 1, 32>" in fixed
    assert "static_strides = array<i64: 1, 1, 1, 1>" in fixed     # closing `>` survives the rewrite


def test_buddy_leaves_a_well_formed_slice_alone():
    op = ('static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 8>, '
          'static_strides = array<i64: 1, 1>, x}> : (tensor<1x8xi64>) -> tensor<1x8xi64>')
    assert _repair_malformed_select_slices(op) == (op, 0)


def test_buddy_leaves_an_unreconstructable_slice_alone():
    """No source dim has extent 99, so there is nothing to safely reconstruct — the op stays
    malformed and the bufferizer rejects it loudly."""
    op = ('static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 99>, '
          'static_strides = array<i64: 1, 1>, x}> : (tensor<1x8xi64>) -> tensor<99xi64>')
    assert _repair_malformed_select_slices(op) == (op, 0)


# --- gen_iface_irdl: the tblgen-to-irdl StrAttr base mlir-opt cannot register ---------------
def test_the_string_attr_base_gets_the_attribute_sigil_not_the_type_sigil():
    """`tblgen-to-irdl` writes StringAttr's base with the TYPE sigil (`!builtin.string`). The IRDL
    runtime resolves `!` through AbstractType and `#` through AbstractAttribute, so the `!`
    spelling does not resolve and the whole dialect fails to load. The repair CORRECTS the sigil;
    it used to relax the whole constraint to `irdl.any`, which then accepted `name = 42 : i64`."""
    raw = '      %5 = irdl.base "!builtin.string" \n      %6 = irdl.any \n'
    assert _fix_string_base_sigil(raw) == '      %5 = irdl.base "#builtin.string" \n      %6 = irdl.any \n'
