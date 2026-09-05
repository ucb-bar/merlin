"""``quantize_before_gather``: quantize the activation, not the matrix it was expanded into.

model2MLIR expands every convolution into ``im2col gather + linalg.matmul`` before merlin sees the
module, so the operand ``lower_contraction_int8`` dynamically quantizes IS the expanded matrix — on
deepjscc's ``enc.net.1`` a 147x4096 f32 im2col matrix, ~41x the 1x3x70x70 activation it was
gathered from. Quantization is elementwise and therefore commutes with a pure gather, but only for
a SINGLE shared scale; the shipped per-parallel-row scale gives one activation element a different
scale in every im2col column it lands in, which is what blocks the commutation.

What is gated here, and why a comment could not do it:

  * **the default does not move** — with the feature off the pass takes no action at all, so the
    shipped int8 datapath cannot drift underneath the lever;
  * **the lever is not inert** — on a real capture it rewrites a specific, asserted number of
    operands and ERASES the f32 expansion. A rewrite that quietly stopped matching (a frontend
    respelling a map, a reshape appearing in the chain) would otherwise look exactly like a model
    with nothing to optimize;
  * **the scale is the gathered max, not the source max** — the arithmetic heart. A stride-2 window
    over a 66-wide padded input never reads column 65, so ``amax(A) >= amax(G(A))``; using the
    source max there would coarsen the scale silently and no shape or op count would show it. The
    covered-box claim is checked against numpy, not against itself;
  * **it refuses rather than approximates** — a computed body, a shared intermediate and a gather
    whose read set has holes each leave the operand on the per-row path, with the reason counted;
  * **``lower_conv_int8`` is dead and says so** — its predicate cannot match anything a model2MLIR
    capture contains, and "0 lowered" is indistinguishable from "nothing to do" without the counters.

NOT gated here, because it is not true: this is not bit-exact against the per-row scheme. It is a
genuine numeric change and is measured against the bundle goldens, not asserted here.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.common.artifacts import recaptures_dir

#: the capture the lever is measured on: every conv in it arrives pre-expanded into im2col+matmul,
#: which is the shape this feature exists for. The test is about the PASS; it skips when absent.
_BUNDLE = "deepjscc_int8_consistent"


def _prepared(bundle):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_xdsl import collapse_overrank_matmul
    from merlin.runtime.dispatch_runtime import _propagate_quant_inner
    module = parse_mlir_file(bundle / "model.mlir")
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)
    return module


def _bundle_or_skip():
    bundle = recaptures_dir() / _BUNDLE
    if not (bundle / "model.mlir").is_file():
        pytest.skip(f"{_BUNDLE} recapture not present")
    return bundle


def _elem_bits(t):
    return getattr(getattr(getattr(t, "element_type", None), "width", None), "data", None)


def _pure_gathers(module):
    from merlin.llvmlower.passes_quant_int import _yields_only_input
    out = []
    for op in module.walk():
        if op.name != "linalg.generic" or not op.body.blocks:
            continue
        if len(op.inputs) == 1 and len(op.results) == 1 and _yields_only_input(op):
            out.append(op)
    return out


# --- the default does not move ------------------------------------------------------------------

def test_feature_off_takes_no_action_at_all():
    """With ``prequant_gather`` unset the pass must not even look: no counters, no rank-0 scale, no
    i8 gather. This is the frozen-baseline invariant in the form a test can hold — the byte-identity
    against the pre-feature lowering is verified separately against the previous revision."""
    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    module = _prepared(_bundle_or_skip())
    report: dict = {}
    lower_contraction_int8(module, report_out=report)
    assert not any(k.startswith("prequant_gather") for k in report), report
    assert all(_elem_bits(g.results[0].type) is None for g in _pure_gathers(module)), \
        "a gather moving i8 means the feature fired with the flag off"


# --- the lever is not inert -----------------------------------------------------------------------

def test_fires_on_im2col_and_erases_the_f32_expansion():
    """On the real capture: the rewrite fires, the gathers it claimed now move i8, and the f32
    expansion is GONE. Erasure is the whole point — leaving the f32 gather in place would keep every
    byte of its traffic and merely add an i8 copy, i.e. the lever would measure as a regression."""
    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    bundle = _bundle_or_skip()
    before = _prepared(bundle)
    n_f32_gathers_before = len(_pure_gathers(before))

    module = _prepared(bundle)
    report: dict = {}
    lower_contraction_int8(module, prequant_gather=True, report_out=report)
    module.verify()

    assert report.get("prequant_gather_rewrites", 0) == 12
    # both abs-max modes are exercised by this one model: 10 gathers read every element of their
    # source, 2 (the stride-2 3x3 convs) read a gap-free box inside it.
    assert report.get("prequant_gather_mode_source_amax", 0) == 10
    assert report.get("prequant_gather_mode_slice_amax", 0) == 2
    assert report.get("prequant_gather_erased_ops", 0) == 48
    # the two partial-coverage gathers must read their abs-max off the covered BOX. Asserted on the
    # emitted op, not just the mode counter: substituting the source max here would still report
    # "slice" while quietly using a coarser scale, and no shape would change.
    assert sum(1 for op in module.walk() if op.name == "tensor.extract_slice") == 2

    after = _pure_gathers(module)
    i8_gathers = [g for g in after if _elem_bits(g.results[0].type) == 8]
    f32_gathers = [g for g in after if _elem_bits(g.results[0].type) is None]
    assert len(i8_gathers) == 12
    # the two survivors are the transposed-conv WEIGHT flips, which are deliberately refused.
    assert len(f32_gathers) == n_f32_gathers_before - 12 == 2
    assert report.get("prequant_gather_refused_unpriceable_index_expr", 0) == 2


def test_the_rewritten_operands_lose_their_per_row_scale():
    """The scale that replaces the per-row one is rank-0 (per-tensor). Without that there is no
    single scale to push through the gather and the commutation does not hold at all."""
    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    module = _prepared(_bundle_or_skip())
    lower_contraction_int8(module, prequant_gather=True)
    def role(op):
        return getattr(op.attributes.get("prov.role"), "data", "")

    rank0_scales = [op for op in module.walk()
                    if op.name == "linalg.generic" and role(op) == "act_scale"
                    and list(op.results[0].type.get_shape()) == []]
    assert len(rank0_scales) == 12
    produced = {op.results[0] for op in rank0_scales}

    # ...and the requant that CONSUMES it must index it with an empty map. A rank-0 tensor read
    # through a per-output-row map is the per-row scheme wearing a per-tensor scale: it reads the
    # wrong element and no shape in the module changes.
    seen = 0
    for op in module.walk():
        if op.name != "linalg.generic" or role(op) != "requant":
            continue
        for v, m in zip(op.inputs, list(op.indexing_maps)):
            if v in produced:
                assert len(m.data.results) == 0, f"per-tensor scale indexed by {m}"
                seen += 1
    assert seen == 12


# --- the scale is the GATHERED max, not the source max ---------------------------------------------

_STRIDED_GATHER = (
    "builtin.module { func.func @forward(%a: tensor<1x1x66x66xf32>) "
    "-> tensor<1x3x3x1x32x32xf32> { "
    "%e = tensor.empty() : tensor<1x3x3x1x32x32xf32> "
    "%g = linalg.generic {indexing_maps = ["
    "affine_map<(d0,d1,d2,d3,d4,d5)->(d3,d0,d4*2+d1,d5*2+d2)>, "
    "affine_map<(d0,d1,d2,d3,d4,d5)->(d0,d1,d2,d3,d4,d5)>], "
    'iterator_types = ["parallel","parallel","parallel","parallel","parallel","parallel"]} '
    "ins(%a : tensor<1x1x66x66xf32>) outs(%e : tensor<1x3x3x1x32x32xf32>) { "
    "^bb(%x: f32, %o: f32): linalg.yield %x : f32 } -> tensor<1x3x3x1x32x32xf32> "
    "func.return %g : tensor<1x3x3x1x32x32xf32> } }")


def test_covered_box_is_exactly_what_the_gather_reads(tmp_path):
    """The correctness heart, checked against numpy rather than against itself.

    A stride-2 3x3 window over a 66-wide input reaches column 64 and never column 65. The box the
    coverage analysis reports must have the same abs-max as the gathered matrix for an activation
    whose largest element sits in the unread column — where the SOURCE max does not."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import _gather_coverage

    src = tmp_path / "gather.mlir"
    src.write_text(_STRIDED_GATHER, encoding="utf-8")
    gather = _pure_gathers(parse_mlir_file(src))[0]
    kind, extents, why = _gather_coverage(gather)
    assert (kind, extents, why) == ("slice", [1, 1, 65, 65], "covered_box")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, 1, 66, 66)).astype(np.float32) * 0.1
    a[0, 0, 65, 65] = 99.0                       # the element the gather never reads
    gathered = np.array(
        [a[0, 0, 2 * i + kh, 2 * j + kw]
         for kh in range(3) for kw in range(3) for i in range(32) for j in range(32)],
        dtype=np.float32)
    box = a[:1, :1, :65, :65]
    assert np.abs(box).max() == pytest.approx(np.abs(gathered).max())
    assert np.abs(a).max() > np.abs(gathered).max(), "the fixture must exercise the difference"


# --- it refuses rather than approximates ----------------------------------------------------------

_MM_ON_GATHER = (
    "builtin.module {{ func.func @forward(%a: tensor<{sk}x{n}xf32>, %w: tensor<{m}x{k}xf32>) "
    "-> tensor<{m}x{n}xf32> {{ "
    "%e = tensor.empty() : tensor<{k}x{n}xf32> "
    "%g = linalg.generic {{indexing_maps = [{imap}, affine_map<(d0,d1)->(d0,d1)>], "
    'iterator_types = ["parallel","parallel"]}} '
    "ins(%a : tensor<{sk}x{n}xf32>) outs(%e : tensor<{k}x{n}xf32>) {{ "
    "^bb(%x: f32, %o: f32): {body} }} -> tensor<{k}x{n}xf32> "
    "%o = tensor.empty() : tensor<{m}x{n}xf32> "
    "%c0 = arith.constant 0.0 : f32 "
    "%f = linalg.fill ins(%c0 : f32) outs(%o : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> "
    "%y = linalg.matmul ins(%w, %g : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) "
    "outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> "
    "func.return %y : tensor<{m}x{n}xf32> }} }}")

_COPY_BODY = "linalg.yield %x : f32"
_COMPUTED_BODY = "%t = arith.mulf %x, %x : f32 linalg.yield %t : f32"
_IDENT = "affine_map<(d0,d1)->(d0,d1)>"
_HOLED = "affine_map<(d0,d1)->(d0*3,d1)>"        # stride 3 over a dim of extent 3*k: reads 1 in 3


def _lower(tmp_path, name, text):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import lower_contraction_int8
    src = tmp_path / f"{name}.mlir"
    src.write_text(text, encoding="utf-8")
    module = parse_mlir_file(src)
    report: dict = {}
    lower_contraction_int8(module, prequant_gather=True, report_out=report)
    module.verify()
    return module, report


def test_fires_on_a_plain_copy_producer(tmp_path):
    """The positive control for the two refusals below: same module, pure-copy body, it fires."""
    _m, report = _lower(tmp_path, "copy", _MM_ON_GATHER.format(
        sk=8, k=8, n=4, m=2, imap=_IDENT, body=_COPY_BODY))
    assert report.get("prequant_gather_rewrites", 0) == 1
    assert report.get("prequant_gather_mode_source_amax", 0) == 1


def test_refuses_a_computed_producer(tmp_path):
    """A body that computes anything does not commute with quantization."""
    _m, report = _lower(tmp_path, "computed", _MM_ON_GATHER.format(
        sk=8, k=8, n=4, m=2, imap=_IDENT, body=_COMPUTED_BODY))
    assert report.get("prequant_gather_rewrites", 0) == 0
    assert report.get("prequant_gather_refused_producer_not_gather", 0) == 1


def test_refuses_a_gather_whose_read_set_has_holes(tmp_path):
    """A stride-3 read over a 3x-tall source touches one row in three, so neither the source max
    nor a bounding box is ``amax(G(A))`` — the operand stays on the per-row path."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import _gather_coverage

    text = _MM_ON_GATHER.format(sk=24, k=8, n=4, m=2, imap=_HOLED, body=_COPY_BODY)
    src = tmp_path / "holed.mlir"
    src.write_text(text, encoding="utf-8")
    assert _gather_coverage(_pure_gathers(parse_mlir_file(src))[0])[0] is None

    _m, report = _lower(tmp_path, "holed2", text)
    assert report.get("prequant_gather_rewrites", 0) == 0
    assert report.get("prequant_gather_refused_strided_holes", 0) == 1


def test_refuses_a_shared_expansion():
    """An expansion read by something else stays materialized, so quantizing before it would only
    ADD an i8 copy. Checked on the real capture, where such sharing actually occurs."""
    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    module = _prepared(_bundle_or_skip())
    report: dict = {}
    lower_contraction_int8(module, prequant_gather=True, report_out=report)
    assert report.get("prequant_gather_refused_shared_value", 0) == 4


# --- lower_conv_int8 is dead, and says so ----------------------------------------------------------

def test_conv_int8_cannot_reach_a_single_conv_in_this_fleet():
    """The pass whose docstring ``contraction_view`` defers convs to has never fired on one.

    Its predicate wants a 2-input generic with a windowed (stride/dilation) indexing map. An
    im2col capture has none: the window lives in the GATHER, which has one input and an empty body.
    The four "compound map" near-misses on this model are broadcasts (a constant index, not a
    stride), which is why the windowed count is tracked separately — a pass that cannot fire must
    not be reported as one that merely missed."""
    from merlin.llvmlower.passes_quant_int import lower_conv_int8

    module = _prepared(_bundle_or_skip())
    report: dict = {}
    assert lower_conv_int8(module, report_out=report) == 0
    assert report["conv_prov_ops"] > 100, "the model must be full of convolution provenance"
    assert report["windowed_map_generics"] == 0
    assert report["compound_map_generics"] == 4       # broadcasts, not convs
    assert report["lowered"] == 0


# --- registration ----------------------------------------------------------------------------------

def test_feature_is_registered_eagerly_and_ranked():
    """``wholemodel_proposer._composes`` swallows the KeyError for an unregistered name and returns
    False, so an unregistered lever is not declined by the search — it is INVISIBLE to it. Both the
    eager registration and the RANKED_LEVERS entry are load-bearing."""
    from merlin.llvmlower.impr_features import QUANTIZE_BEFORE_GATHER_NAME, known, normalize
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS

    assert QUANTIZE_BEFORE_GATHER_NAME in known()
    assert normalize([QUANTIZE_BEFORE_GATHER_NAME]) == frozenset({QUANTIZE_BEFORE_GATHER_NAME})
    assert QUANTIZE_BEFORE_GATHER_NAME in [n for n, _ in RANKED_LEVERS]


def test_the_feature_name_reaches_the_quant_pass(monkeypatch):
    """The gate has to travel feature -> prepare_for_lowering -> apply_quant -> the pass. A feature
    that is registered and ranked but never threaded builds clean and changes nothing."""
    from merlin.llvmlower import passes_quant_int as Q
    from merlin.llvmlower import quant_passes as QP

    seen: list[bool] = []
    monkeypatch.setattr(Q, "lower_contraction_int8",
                        lambda _m, **kw: (seen.append(kw.get("prequant_gather", False)), 0)[1])
    for fn in ("lower_conv_int8", "lower_softmax_int", "lower_gelu_int", "lower_silu_int",
               "lower_rsqrt_int"):
        monkeypatch.setattr(Q, fn, lambda _m, **kw: 0)
    QP.apply_quant(object(), prequant_gather=True)
    assert seen == [True]
    seen.clear()
    QP.apply_quant(object())
    assert seen == [False], "the default must not pass the flag at all"


_DIAGONAL_GATHER = (
    "builtin.module { func.func @forward(%a: tensor<8x8xf32>) -> tensor<8xf32> { "
    "%e = tensor.empty() : tensor<8xf32> "
    "%g = linalg.generic {indexing_maps = ["
    "affine_map<(d0)->(d0,d0)>, affine_map<(d0)->(d0)>], "
    'iterator_types = ["parallel"]} '
    "ins(%a : tensor<8x8xf32>) outs(%e : tensor<8xf32>) { "
    "^bb(%x: f32, %o: f32): linalg.yield %x : f32 } -> tensor<8xf32> "
    "func.return %g : tensor<8xf32> } }")


def test_refuses_a_gather_whose_axes_are_coupled(tmp_path):
    """A dim driving two source axes reads a DIAGONAL, not a box.

    Every axis of `(d0, d0)` over an 8x8 source prices as fully covered when the axes are analysed
    independently, so a per-axis proof alone would substitute ``amax(A)`` for a strictly smaller
    ``amax(G(A))`` — the exact silent coarsening the coverage analysis exists to prevent, and the one
    failure mode that no shape or op count in the module would reveal."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_quant_int import _gather_coverage

    src = tmp_path / "diag.mlir"
    src.write_text(_DIAGONAL_GATHER, encoding="utf-8")
    gather = _pure_gathers(parse_mlir_file(src))[0]
    assert _gather_coverage(gather) == (None, [], "coupled_iteration_dim")
