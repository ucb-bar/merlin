"""What counts as a CONTRACTION, and why a rank test is not enough.

Two things read a module and must agree about which ops are contractions:

  * the block POLICY (``kernels.shapes.contraction_shapes`` -> ``llvmlower.perop_blocks.block_table``),
    which runs on the prepared module and prices a register block per geometry; and
  * the TAGGER (``perop_blocks.tag_prepared_mlir``), which runs AFTER
    ``linalg-specialize-generic-ops`` and can only tag what that pass named.

When they disagree, ``perop_blocks.BlockAgreementError`` fails the build -- correctly, because an
untagged contraction matches no schedule arm and silently lowers to scalar loops. MEASURED on smolvla
int8: ``1 contraction(s) were priced by the block policy but not tagged:
['linalg.matmul:1x32:32']``, with the tagger reporting NO untagged geometry, because the op the
policy priced is not a named contraction anywhere in the module.

The op was an ``aten.bucketize`` boundary search. It shares the contraction signature exactly --
``[parallel, parallel, reduction]``, output rank 2, a rank-2 first input -- and the classifier's test
was ranks only, so it was read as a matmul with parallel (1, 32) and a reduction extent of 32 taken
from A's LAST dim. A's last dim is N, not K; the real reduction extent is the 31 of its rank-1
boundary operand. The distinguishing fact is in the INDEXING MAPS, which the test now reads: a
contraction's A operand carries the reduction dim, and carries it last.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.kernels.shapes import contraction_shapes
from merlin.llvmlower import perop_blocks as pb

#: The exact op that failed the smolvla build, reduced to its structure. Note the A map
#: ``(d0, d1, d2) -> (d0, d1)``: it does not mention the reduction dim d2 at all.
BUCKETIZE = """\
#a   = affine_map<(d0, d1, d2) -> (d0, d1)>
#b   = affine_map<(d0, d1, d2) -> (d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @forward(%x: tensor<1x32xf32>, %bnd: tensor<31xf32>) -> tensor<1x32xi64> {
    %z = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %e = tensor.empty() : tensor<1x32xi64>
    %f = linalg.fill ins(%z : i64) outs(%e : tensor<1x32xi64>) -> tensor<1x32xi64>
    %r = linalg.generic {indexing_maps = [#a, #b, #out],
                         iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%x, %bnd : tensor<1x32xf32>, tensor<31xf32>)
         outs(%f : tensor<1x32xi64>) {
    ^bb0(%xv: f32, %bv: f32, %acc: i64):
      %c = arith.cmpf ole, %bv, %xv : f32
      %s = arith.select %c, %one, %z : i64
      %n = arith.addi %acc, %s : i64
      linalg.yield %n : i64
    } -> tensor<1x32xi64>
    return %r : tensor<1x32xi64>
  }
}
"""

#: A REAL contraction at the SAME degenerate M: this is the decode-step shape every autoregressive and
#: flow-matching model produces, and it must still be priced. The fix must reject the bucketize on its
#: MAPS, never on its extents.
DEGENERATE_M_MATMUL = """\
#a   = affine_map<(d0, d1, d2) -> (d0, d2)>
#b   = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @forward(%x: tensor<1x32xi8>, %w: tensor<32x32xi8>) -> tensor<1x32xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<1x32xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<1x32xi32>) -> tensor<1x32xi32>
    %r = linalg.generic {indexing_maps = [#a, #b, #out],
                         iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%x, %w : tensor<1x32xi8>, tensor<32x32xi8>)
         outs(%f : tensor<1x32xi32>) {
    ^bb0(%xv: i8, %wv: i8, %acc: i32):
      %xe = arith.extsi %xv : i8 to i32
      %we = arith.extsi %wv : i8 to i32
      %m = arith.muli %xe, %we : i32
      %s = arith.addi %m, %acc : i32
      linalg.yield %s : i32
    } -> tensor<1x32xi32>
    return %r : tensor<1x32xi32>
  }
}
"""


def _aliases(text: str) -> str:
    """The leading ``#name = affine_map<...>`` lines of a module, so two can be merged into one."""
    return "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))


def _func_body(text: str) -> str:
    """The ``func.func`` block of a single-func module, without its ``module { }`` wrapper."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("func.func"))
    return "\n".join(lines[start:-1])


def _specialized_op_names(text: str) -> list[str]:
    """Op names after ``linalg-specialize-generic-ops`` -- the pass the TAGGER runs behind.

    This is the other side of the disagreement, and running it for real is the point: the assertion
    is that the classifier agrees with THIS pass, not with a restatement of what it is believed to do.
    """
    exe = repo_root() / "third_party" / "llvm-install" / "bin" / "mlir-opt"
    exe = exe if exe.is_file() else shutil.which("mlir-opt")
    if exe is None:
        pytest.skip("no mlir-opt (third_party/llvm-install not built)")
    proc = subprocess.run([str(exe), "--pass-pipeline=builtin.module(func.func("
                           "linalg-specialize-generic-ops))"],
                          input=text, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return [tok for tok in ("linalg.matmul", "linalg.generic") if tok in proc.stdout]


def test_a_bucketize_search_is_not_priced_as_a_contraction():
    """The smolvla failure, reproduced from the geometry alone -- no bundle, no board.

    Before the map check this returned ``linalg.matmul:1x32:32``, which the tagger could never tag.
    """
    got = contraction_shapes(BUCKETIZE)
    assert got == [], (
        f"a boundary search was priced as a contraction: {got}. The tagger runs after "
        f"linalg-specialize-generic-ops, which will not name it, so this geometry is priced and "
        f"un-taggable and BlockAgreementError fails the build.")
    assert pb.block_table(contraction_shapes(BUCKETIZE), nr_cap=16) == {}


def test_a_degenerate_m_contraction_is_still_priced():
    """M=1 is not the defect and must not become the fix. The decode-step shape still prices."""
    got = contraction_shapes(DEGENERATE_M_MATMUL)
    assert len(got) == 1, got
    assert got[0].op == "linalg.matmul"
    assert got[0].parallel == (1, 32) and got[0].reduction == (32,), got[0]
    table = pb.block_table(got, nr_cap=16)
    assert table == {"linalg.matmul:1x32:32": (1, 16)}, table


def test_the_classifier_agrees_with_the_pass_the_tagger_runs_behind():
    """One source of truth, checked against the pass itself rather than against a belief about it."""
    assert _specialized_op_names(DEGENERATE_M_MATMUL) == ["linalg.matmul"], (
        "the real contraction must specialize -- if it does not, pricing it is what is wrong")
    assert _specialized_op_names(BUCKETIZE) == ["linalg.generic"], (
        "the bucketize must NOT specialize -- that is why pricing it is a build failure")


def test_priced_and_specializable_agree_on_a_module_holding_both():
    """The end-to-end invariant: everything the policy prices is something the tagger can reach."""
    # one module holding both funcs (the map aliases are renamed so the two do not collide)
    mm = DEGENERATE_M_MATMUL.replace("@forward", "@mm")
    bk = BUCKETIZE.replace("@forward", "@bk")
    for old_alias, new_alias in (("#a", "#ba"), ("#b", "#bb"), ("#out", "#bout")):
        bk = bk.replace(old_alias + " ", new_alias + " ").replace(old_alias + ",", new_alias + ",") \
               .replace(old_alias + "]", new_alias + "]")
    body = _func_body(mm) + "\n" + _func_body(bk)
    both = _aliases(mm) + _aliases(bk) + "module {\n" + body + "\n}\n"
    priced = contraction_shapes(both)
    assert {s.op for s in priced} == {"linalg.matmul"}, priced
    assert len(priced) == 1, priced
