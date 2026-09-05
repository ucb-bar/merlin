"""The DIRECT-convolution arm: tile + vectorize the conv form that materializes no im2col.

model2MLIR rewrites every convolution into ``im2col gather + linalg.matmul`` because a contraction is
what a vector schedule matches. Its own element budget (``M2M_IM2COL_MAX_ELEMS``) can divert a conv to
the direct, compound-affine form instead -- which materializes nothing, and which, before this arm,
matched NO arm of the transform schedule and therefore lowered to scalar ``scf`` loops. That made the
budget knob a pure loss and so unusable. These tests pin the three things that were NOT obvious, each
of them measured on deepjscc int8 lowered at ``M2M_IM2COL_MAX_ELEMS=147456``:

1. What the direct form actually IS (4 parallel + 3 reduction dims, a compound-affine activation map),
   and that ``kernels.shapes.contraction_shapes`` cannot see it -- which is why nothing tagged it.
2. That ``transform.structured.vectorize`` REFUSES that op: its indexing maps are not projected
   permutations, and that is a property of the MAP, not the extents (a 1x1 pointwise conv written with
   the same map fails identically). The arm therefore folds the tiled op's unit dims first.
3. That the arm is default-OFF: with the request feature absent the block table -- and every tag and
   schedule derived from it -- is byte-identical.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import perop_blocks as pb

#: A direct 2-D conv exactly as the int8 datapath leaves it (`passes_quant_int.lower_conv_int8` keeps
#: the EXACT maps and iterators of the f32 form): i8 x i8 -> i32, 4 parallel + 3 reduction dims, and an
#: activation map carrying the window term. Geometry is deepjscc's `dec.res_list.0.conv_block.1`.
CONV_MLIR = """\
#in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#w   = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func @forward(%a: tensor<1x64x18x18xi8>, %b: tensor<32x64x3x3xi8>) -> tensor<1x32x16x16xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<1x32x16x16xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<1x32x16x16xi32>) -> tensor<1x32x16x16xi32>
    %r = linalg.generic {indexing_maps = [#in, #w, #out],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel",
                                           "reduction", "reduction", "reduction"]}
         ins(%a, %b : tensor<1x64x18x18xi8>, tensor<32x64x3x3xi8>)
         outs(%f : tensor<1x32x16x16xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %m, %acc : i32
      linalg.yield %s : i32
    } -> tensor<1x32x16x16xi32>
    return %r : tensor<1x32x16x16xi32>
  }
}
"""

#: The schedule text as it stood BEFORE the conv arm, for a table holding one matmul block. The arm
#: must leave this untouched: a conv-free table is every build anyone ships today, and a single
#: character of drift here re-lowers every model.
_FROZEN_ONE_MATMUL_BLOCK = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %b0 = transform.structured.match attributes{merlin.blk_mm_4x16} in %arg0 : (!transform.any_op) -> !transform.any_op
    %b0t, %b0l:2 = transform.structured.tile_using_for %b0 tile_sizes [4, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %b0k, %b0kl = transform.structured.tile_using_for %b0t tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %b0k vector_sizes [4, 16, 1] : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.yield
  }
}
"""


def _mlir_opt():
    """The repo's own LLVM-23 ``mlir-opt``, or None. Used to run the transform interpreter alone --
    the assertion is about what the SCHEDULE does, and that needs no clang, board or model."""
    exe = repo_root() / "third_party" / "llvm-install" / "bin" / "mlir-opt"
    return exe if exe.is_file() else (shutil.which("mlir-opt") or None)


def _run_schedule(tmp_path, module_text: str, schedule_text: str):
    """Run ``schedule_text`` over ``module_text`` through the transform interpreter.

    Returns ``(returncode, stdout, stderr)``. Deliberately NOT raising on failure: two of the tests
    below assert that a schedule FAILS, and they need the message.
    """
    exe = _mlir_opt()
    if exe is None:
        pytest.skip("no mlir-opt (third_party/llvm-install not built)")
    src = tmp_path / "m.mlir"
    sched = tmp_path / "s.mlir"
    src.write_text(module_text, encoding="utf-8")
    sched.write_text(schedule_text, encoding="utf-8")
    proc = subprocess.run(
        [str(exe), str(src), f"--transform-preload-library=transform-library-paths={sched}",
         "--transform-interpreter"],
        capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout, proc.stderr


# --------------------------------------------------------------------------------------------------
# 1. THE FORM. What a direct conv looks like, and why nothing saw it.
# --------------------------------------------------------------------------------------------------

def test_the_direct_conv_is_invisible_to_the_contraction_observer():
    """`contraction_shapes` cannot see a direct conv -- which is exactly why it was never tagged.

    Its test admits a generic with EXACTLY ONE reduction dim; a conv has three (ci, kh, kw). This is
    not a bug in that observer, it is why the conv needed its own reader.
    """
    from merlin.kernels.shapes import contraction_shapes
    assert contraction_shapes(CONV_MLIR) == [], (
        "if the contraction observer starts claiming convs, block_table will price them with the "
        "matmul tile-size vector (3 sizes for a 7-dim op) and the schedule will die")


def test_conv_shapes_reads_the_direct_form_off_the_ir():
    """The reader reports the op's OWN dim order: parallel (N, F, Oh, Ow), reduction (Ci, Kh, Kw)."""
    shapes = pb.conv_shapes(CONV_MLIR)
    assert len(shapes) == 1, f"expected one direct conv, got {shapes}"
    s = shapes[0]
    assert s.op == pb.CONV_CLASS
    assert s.parallel == (1, 32, 16, 16), s
    assert s.reduction == (64, 3, 3), s
    assert s.dtypes == ("i8", "i8", "i32"), s


def test_conv_geometry_solves_the_window_and_rejects_what_is_not_a_conv():
    """The predicate is pure extent arithmetic, so both sides of the tagging can run it.

    ``in = (out - 1) * stride + kernel`` both validates the shape triple and RECOVERS the stride.
    """
    # stride 1: 16 -> 18 through a 3x3 window
    assert pb.conv_geometry([1, 32, 16, 16], [1, 64, 18, 18], [32, 64, 3, 3]) == (1, 1)
    # stride 2: 32 -> 65 through a 3x3 window
    assert pb.conv_geometry([1, 32, 32, 32], [1, 64, 65, 65], [32, 64, 3, 3]) == (2, 2)
    # channel mismatch is not this conv
    assert pb.conv_geometry([1, 32, 16, 16], [1, 63, 18, 18], [32, 64, 3, 3]) is None
    # an input too small for the window
    assert pb.conv_geometry([1, 32, 16, 16], [1, 64, 2, 2], [32, 64, 3, 3]) is None
    # a span that no integer stride explains
    assert pb.conv_geometry([1, 32, 16, 16], [1, 64, 20, 18], [32, 64, 3, 3]) is None
    # not rank 4
    assert pb.conv_geometry([32, 16], [64, 18], [32, 64]) is None


# --------------------------------------------------------------------------------------------------
# 2. DEFAULT-OFF. The whole arm is inert without the request.
# --------------------------------------------------------------------------------------------------

def test_the_arm_prices_nothing_without_the_request_feature():
    assert pb.conv_block_table(CONV_MLIR, (), nr_cap=16) == {}
    assert pb.conv_block_table(CONV_MLIR, ("some_other_feature",), nr_cap=16) == {}
    priced = pb.conv_block_table(CONV_MLIR, (pb.CONV_ARM_FEATURE,), mr_cap=4, nr_cap=16)
    assert priced == {"linalg.conv2d_direct:1x32x16x16:64x3x3": (4, 16)}, priced


def test_a_conv_free_table_emits_the_schedule_it_always_did():
    """Byte-for-byte. Every build that ships today has a conv-free table."""
    got = pb.schedule_text({"linalg.matmul:8x128:344": (4, 16)}, 64)
    assert got == _FROZEN_ONE_MATMUL_BLOCK, got


def test_a_conv_free_table_emits_the_tagger_that_always_tagged():
    """The tagger walks convs now, but with no conv key in the table it tags none of them."""
    src = pb.runner_rewrite_src({"linalg.matmul:8x128:344": (4, 16)})
    assert "merlin.blk_conv" not in src
    compile(src, "<runner>", "exec")     # it is spliced into a script; it must at least parse


# --------------------------------------------------------------------------------------------------
# 3. THE ARM ITSELF. What it emits, and that the emitted schedule really tiles and vectorizes.
# --------------------------------------------------------------------------------------------------

def test_the_conv_arm_tiles_ow_and_f_and_folds_before_vectorizing():
    """The three stages, in the order the measurement forced.

    NR goes on ``ow`` (d3, contiguous and absent from the weight map, so the weight is a broadcast
    scalar); MR on ``f`` (d1). The fold between tiling and vectorizing is not decoration -- see the
    test below, which shows the vectorizer refusing the op without it.
    """
    text = pb.schedule_text({"linalg.conv2d_direct:1x32x16x16:64x3x3": (4, 16)}, 64)
    assert "tile_sizes [1, 4, 1, 16, 0, 0, 0]" in text, text
    assert "tile_sizes [0, 0, 0, 0, 1, 1, 1]" in text, text
    assert "transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices" in text, text
    assert "vector_sizes [4, 16]" in text, text
    # the fold drops the op's tag, so the vectorize must find the op through the annotated nest
    assert 'transform.annotate' in text and pb.conv_nest_tag(4, 16) in text, text
    assert text.index("fold_unit_extent_dims_via_slices") < text.index("vector_sizes [4, 16]"), (
        "the fold must run BEFORE the vectorize, or the vectorize has nothing it can accept")


def test_mr_one_folds_to_a_rank_one_vector():
    """At MR=1 every dim but ``ow`` is unit after tiling, so the folded op is rank 1."""
    text = pb.schedule_text({"linalg.conv2d_direct:1x32x16x16:64x3x3": (1, 16)}, 64)
    assert "vector_sizes [16]" in text, text


def test_the_vectorizer_refuses_the_conv_without_the_fold(tmp_path):
    """MEASURED, and the reason the arm has an extra stage the matmul arm does not.

    A conv's activation map ``(d0, d4, d2 + d5, d3 + d6)`` is not a projected permutation, and
    ``transform.structured.vectorize``'s precondition requires that of every map. Tiling the window
    dims to 1 does not help: the map keeps its ``+ d5`` term whatever the trip count.
    """
    naive = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %c = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %ct, %cl:4 = transform.structured.tile_using_for %c tile_sizes [1, 1, 1, 16, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %ck, %ckl:3 = transform.structured.tile_using_for %ct tile_sizes [0, 0, 0, 0, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %ck vector_sizes [1, 1, 1, 16, 1, 1, 1] : !transform.any_op
    transform.yield
  }
}
"""
    rc, _out, err = _run_schedule(tmp_path, CONV_MLIR, naive)
    assert rc != 0, "if this starts passing, the fold stage may no longer be needed -- re-measure"
    assert "Attempted to vectorize, but failed" in err, err


def test_the_conv_arm_actually_vectorizes_the_conv(tmp_path):
    """End to end on the schedule the arm emits: the innermost body becomes vector ops.

    The shape asserted is the micro-kernel the tile choice was made for -- an activation VECTOR times
    a BROADCAST weight scalar into a resident accumulator, which is the operand shape a widening
    integer MAC wants. Asserted on the IR, not on an instruction count: this is a claim about what the
    schedule produces, not about what it costs.
    """
    # the module needs the tag the tagger would have applied
    tagged = CONV_MLIR.replace(
        "outs(%f : tensor<1x32x16x16xi32>) {",
        "outs(%f : tensor<1x32x16x16xi32>) attrs = {"
        + pb.tag_for(pb.CONV_CLASS, 1, 16) + "} {")
    text = pb.schedule_text({"linalg.conv2d_direct:1x32x16x16:64x3x3": (1, 16)}, 64)
    rc, out, err = _run_schedule(tmp_path, tagged, text)
    assert rc == 0, err
    assert "vector.transfer_read" in out, out[-3000:]
    assert "vector<16xi32>" in out, out[-3000:]
    assert "arith.muli %" in out and "vector<16xi32>" in out
    assert "linalg.generic" not in out, (
        "the conv must be gone from linalg -- if it survives it falls to convert-linalg-to-loops "
        "and lowers scalar, which is the state this arm exists to fix")


def test_the_tagger_tags_a_priced_conv(tmp_path):
    """The tagger itself, run for real -- the generated source is spliced into a script and executed
    in the m2m venv, which is the only place the MLIR python bindings live.

    Asserting on the source text alone is not enough: the conv KEY builder can be present and the
    walk can still never dispatch to it, which is exactly the shape of the bug the priced-vs-tagged
    guard exists to catch.
    """
    from merlin.llvmlower.toolchain import m2m_python
    if not m2m_python().is_file():
        pytest.skip("no model2MLIR venv (MERLIN_M2M_DIR)")
    src = tmp_path / "model.mlir"
    src.write_text(CONV_MLIR, encoding="utf-8")
    table = {"linalg.conv2d_direct:1x32x16x16:64x3x3": (1, 16)}
    out = pb.tag_prepared_mlir(src, table, work=tmp_path)
    text = out.read_text(encoding="utf-8")
    assert pb.tag_for(pb.CONV_CLASS, 1, 16) in text, text[-2000:]


def test_the_tagger_leaves_an_unpriced_conv_alone(tmp_path):
    """With no conv key in the table the same tagger tags nothing -- the default build is untouched."""
    from merlin.llvmlower.toolchain import m2m_python
    if not m2m_python().is_file():
        pytest.skip("no model2MLIR venv (MERLIN_M2M_DIR)")
    src = tmp_path / "model.mlir"
    src.write_text(CONV_MLIR, encoding="utf-8")
    out = pb.tag_prepared_mlir(src, {}, work=tmp_path)
    assert "merlin.blk_" not in out.read_text(encoding="utf-8")


def test_the_tagger_names_an_unpriced_conv_instead_of_ignoring_it():
    """A conv with no block must show up in the agreement report, not vanish.

    The generated tagger walks convs unconditionally; the TABLE decides whether one is tagged. An
    untagged conv lowers scalar, and the whole point of the agreement line is that such an op is
    named rather than silently dropped.
    """
    src = pb.runner_rewrite_src({"linalg.matmul:8x128:344": (4, 16)})
    assert "_merlin_conv_key" in src
    assert "seen_untagged.add" in src
    # one definition of the predicate, spliced from merlin -- never restated
    assert src.count("def conv_geometry") == 1, src[:2000]
