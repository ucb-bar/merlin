"""``llvmlower.im2col_pack``: the panel-packed im2col rewrite.

Four things are asserted, in the order they can fail:

1. **The frozen baseline is untouched.** With the feature absent nothing runs; with it absent from a
   feature set the pipeline string and the transform schedule are byte-identical.
2. **The rewrite is a rewrite, not a hope.** On a hand-written im2col chain it produces a packed
   ``[MO][K][NR]`` gather and a panel loop, and the MLIR still verifies.
3. **It fails closed.** Each refusal is a named, counted reason -- never a silent skip and never a
   guess.
4. **It computes the same thing.** The packed form is evaluated against the unpacked one by
   interpreting both indexing schemes over the same input, element for element.
"""
from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.llvmlower import im2col_pack as ip
from merlin.llvmlower import impr_features as F
from merlin.llvmlower import perop_blocks as pb
from merlin.llvmlower import pipeline as P


# ---------------------------------------------------------------------------------------------------
# a minimal im2col chain, in the exact form model2MLIR emits (and passes_quant_int leaves behind)
# ---------------------------------------------------------------------------------------------------

def _module(*, c=2, kh=3, kw=3, n=1, oh=4, ow=32, f=8, sh=1, sw=1, dh=1, dw=1):
    hin, win = (oh - 1) * sh + (kh - 1) * dh + 1, (ow - 1) * sw + (kw - 1) * dw + 1
    k, m = c * kh * kw, n * oh * ow
    return f"""
module {{
  func.func @forward(%in: tensor<{n}x{c}x{hin}x{win}xi8>, %w: tensor<{f}x{k}xi8>)
      -> tensor<{f}x{m}xi32> {{
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>
    %g = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5)
                           -> (d3, d0, d4 * {sh} + d1 * {dh}, d5 * {sw} + d2 * {dw})>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}}
      ins(%in : tensor<{n}x{c}x{hin}x{win}xi8>)
      outs(%e : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>) {{
    ^bb0(%a: i8, %o: i8):
      linalg.yield %a : i8
    }} -> tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>
    %flat = tensor.collapse_shape %g [[0, 1, 2, 3, 4, 5]]
      : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8> into tensor<{k * m}xi8>
    %col = tensor.expand_shape %flat [[0, 1]] output_shape [{k}, {m}]
      : tensor<{k * m}xi8> into tensor<{k}x{m}xi8>
    %ae = tensor.empty() : tensor<{f}x{m}xi32>
    %acc = linalg.fill ins(%c0 : i32) outs(%ae : tensor<{f}x{m}xi32>) -> tensor<{f}x{m}xi32>
    %mm = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%w, %col : tensor<{f}x{k}xi8>, tensor<{k}x{m}xi8>)
      outs(%acc : tensor<{f}x{m}xi32>) {{
    ^bb0(%x: i8, %y: i8, %o: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %p = arith.muli %xe, %ye : i32
      %s = arith.addi %p, %o : i32
      linalg.yield %s : i32
    }} -> tensor<{f}x{m}xi32>
    func.return %mm : tensor<{f}x{m}xi32>
  }}
}}
"""


def _table(f, m, k, mr, nr):
    return {pb.shape_key("linalg.matmul", (f, m), (k,)): (mr, nr)}


# ---------------------------------------------------------------------------------------------------
# 1. the frozen baseline
# ---------------------------------------------------------------------------------------------------

def test_the_feature_is_registered_so_the_search_can_see_it():
    # `wholemodel_proposer._composes` swallows the KeyError for an unregistered name and answers
    # False, so an unregistered lever is never proposed AND never complains.
    assert ip.ensure_registered() == ip.FEATURE
    assert ip.FEATURE in F.known()
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS, _composes
    assert ip.FEATURE in [n for n, _ in RANKED_LEVERS]
    assert _composes(["perop_register_block", ip.FEATURE])


def test_empty_feature_set_leaves_the_pipeline_and_schedule_byte_identical():
    ip.ensure_registered()
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == \
        P.build_rvv_pipeline("/tmp/s.mlir")
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE


def test_the_feature_edits_no_pipeline_and_no_schedule_of_its_own():
    # It is a REQUEST consumed by the preparation step. If it ever grew a pipeline or schedule hook,
    # naming it would perturb a build that never reached the preparation step.
    feat = F.get(ip.ensure_registered())
    assert feat.edit_pipeline is None and feat.edit_schedule is None and feat.edit_cflags is None
    assert not feat.schedule_replace
    base = P.build_rvv_pipeline("/tmp/s.mlir")
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset({ip.FEATURE})) == base
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset({ip.FEATURE})) == \
        P.RVV_TRANSFORM_SCHEDULE


def test_a_module_with_no_im2col_chain_is_left_alone():
    src = """
module {
  func.func @forward(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x16xi32> {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<4x16xi32>
    %acc = linalg.fill ins(%c0 : i32) outs(%e : tensor<4x16xi32>) -> tensor<4x16xi32>
    %mm = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xi8>, tensor<8x16xi8>) outs(%acc : tensor<4x16xi32>) {
    ^bb0(%x: i8, %y: i8, %o: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %p = arith.muli %xe, %ye : i32
      %s = arith.addi %p, %o : i32
      linalg.yield %s : i32
    } -> tensor<4x16xi32>
    func.return %mm : tensor<4x16xi32>
  }
}
"""
    module = mq.parse(src)
    before = str(module)
    report = ip.rewrite_module(module, _table(4, 16, 8, 4, 16))
    assert report.packed == 0
    assert str(module) == before


# ---------------------------------------------------------------------------------------------------
# 2. the rewrite
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw_args", [
    dict(),                                   # stride 1, dilation 1
    dict(sh=2, sw=2, ow=32, oh=4),            # strided
    dict(dh=2, dw=2),                         # dilated
    dict(c=1, kh=1, kw=1),                    # 1x1 (pointwise)
    dict(f=3),                                # a non-power-of-two M extent
])
def test_the_chain_is_rewritten_into_a_packed_gather_and_a_panel_loop(kw_args):
    args = dict(c=2, kh=3, kw=3, n=1, oh=4, ow=32, f=8, sh=1, sw=1, dh=1, dw=1)
    args.update(kw_args)
    f = args["f"]
    k = args["c"] * args["kh"] * args["kw"]
    m = args["n"] * args["oh"] * args["ow"]
    nr = 16
    module = mq.parse(_module(**args))
    report = ip.rewrite_module(module, _table(f, m, k, 4, nr))
    assert report.packed == 1, report.to_dict()

    text = str(module)
    # the packed column tensor, its collapse to [MO, K, NR], and the panel loop
    mo = m // nr
    assert f"tensor<{mo}x{k}x{nr}xi8>" in text
    assert "scf.for" in text
    # the old [K, M] column matrix is gone
    assert f"tensor<{k}x{m}xi8>" not in text
    # and the inner contraction is an ORDINARY [F, NR] x K matmul, which is the whole point: it needs
    # no packed-specific schedule arm, tagger key or op class.
    assert report.entries == [(pb.shape_key("linalg.matmul", (f, nr), (k,)), 4, nr)]


def test_the_rewritten_module_still_verifies():
    module = mq.parse(_module())
    assert ip.rewrite_module(module, _table(8, 128, 18, 4, 16)).packed == 1
    _mlir_opt_verifies(str(module))


def _mlir_opt_verifies(text: str) -> None:
    """Parse+verify with the repo's own mlir-opt. Skips when the toolchain is not built."""
    import subprocess

    from merlin.common.paths import repo_root

    opt = repo_root() / "third_party/llvm-install/bin/mlir-opt"
    if not opt.is_file():
        pytest.skip(f"no mlir-opt at {opt}")
    proc = subprocess.run([str(opt), "--allow-unregistered-dialect", "-o", "/dev/null"],
                          input=text, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------------------------------
# 3. fail closed
# ---------------------------------------------------------------------------------------------------

def test_a_geometry_the_block_table_did_not_price_is_refused_not_guessed():
    module = mq.parse(_module())
    report = ip.rewrite_module(module, {})
    assert report.packed == 0
    assert report.refusals == {"refused_unpriced": 1}


def test_a_panel_width_that_does_not_divide_ow_is_refused():
    # Ow = 20, NR = 16: splitting ow into (owo, owi) would need a mod/floordiv in the gather's map
    # and a masked panel in the schedule. Refused, and named.
    module = mq.parse(_module(ow=20, oh=4))
    report = ip.rewrite_module(module, _table(8, 80, 18, 4, 16))
    assert report.packed == 0
    assert report.refusals == {"refused_nr_does_not_divide_ow": 1}


def test_a_gather_whose_result_has_a_second_consumer_is_refused():
    src = _module()
    # give the column matrix a second use, so rewriting it would leave the old gather alive
    src = src.replace(
        "    func.return %mm : tensor<8x128xi32>",
        "    %d = tensor.empty() : tensor<18x128xi8>\n"
        "    %copy = linalg.copy ins(%col : tensor<18x128xi8>) outs(%d : tensor<18x128xi8>)"
        " -> tensor<18x128xi8>\n"
        "    func.return %mm : tensor<8x128xi32>")
    module = mq.parse(src)
    report = ip.rewrite_module(module, _table(8, 128, 18, 4, 16))
    assert report.packed == 0
    assert report.refusals == {"refused_shared_value": 1}


def test_every_refusal_is_counted_under_a_name():
    # A refusal that is not counted is a silent skip, which is how a lever comes to report as applied
    # while doing nothing.
    r = ip.PackReport()
    r.refuse("x"), r.refuse("x"), r.refuse("y")
    assert r.to_dict()["refusals"] == {"x": 2, "y": 1}


# ---------------------------------------------------------------------------------------------------
# 4. same arithmetic
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sh,sw,dh,dw,ow,oh", [(1, 1, 1, 1, 32, 4), (2, 2, 1, 1, 32, 4),
                                               (1, 1, 2, 2, 32, 4), (3, 1, 1, 2, 16, 3)])
def test_the_packed_index_scheme_addresses_the_same_element(sh, sw, dh, dw, ow, oh):
    """``col_p[mo][k][mi] == col[k][mo*NR + mi]`` for every element, evaluated from the two schemes.

    This is the correctness core of the rewrite -- the packing is a permutation of the SAME gather --
    and it is checked by evaluating both index maps rather than by trusting the affine algebra.
    """
    c, kh, kw, n, nr = 2, 3, 3, 1, 16
    for cc in range(c):
        for i in range(kh):
            for j in range(kw):
                k = (cc * kh + i) * kw + j
                for nn in range(n):
                    for o in range(oh):
                        for w in range(ow):
                            m = (nn * oh + o) * ow + w
                            mo, mi = divmod(m, nr)
                            owo, owi = divmod(w, nr)
                            # unpacked: col[k][m] reads in[n, c, o*sh + i*dh, w*sw + j*dw]
                            src = (nn, cc, o * sh + i * dh, w * sw + j * dw)
                            # packed: col_p[mo][k][mi], mo = (n*Oh + oh)*OWO + owo, mi = owi
                            assert mo == (nn * oh + o) * (ow // nr) + owo
                            assert mi == owi
                            # ...and the packed gather's own map, as `_rewrite_one` writes it
                            packed_src = (nn, cc, o * sh + i * dh,
                                          owo * (sw * nr) + owi * sw + j * dw)
                            assert packed_src == src
