"""A contraction spelled ``linalg.generic`` has an extent too, and it is read off its own line.

Two defects of one shape.

**1. The extent reader knew one spelling.** ``_matmul_extents`` walked the module's named ``linalg.matmul``
operations. A convolution lowered to im2col arrives as a batched-matmul ``linalg.generic`` -- maps
``(d0,d1,d3),(d0,d3,d2)->(d0,d1,d2)`` over ``[parallel, parallel, parallel, reduction]`` -- so it carried
no shape, ``compile.mesh.tile_builder_op`` correctly declined it (``has_shape`` False), and the site
counted ``n_unsynthesizable`` and failed the capsule at its cert tier. ``M2_microvit_gemmini`` had exactly
one such site left after the parallel-only gathers were excluded.

**2. The extents were joined positionally.** That reader produced a list over the named matmuls and
``model_op_demands`` indexed it with a counter advanced on ``prov.op == "matmul"`` tags -- which assumes
those tags ARE those operations. Measured over this corpus: 19 capsules disagree on the count
(``M1_lstmnetvit_gemmini``: 52 tags against 37 named matmuls), and from the first disagreement onward a
demand carried another layer's shape. It is the same positional-alignment defect ``prov.family`` and
``carrier_reduces`` were each fixed for, so every fact is now read from the tag's OWN line.

What the derivation refuses is the point: a real convolution, a broadcast, a pooling reduction and a
row-wise reduce all reach this reader, and none of them is a contraction.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_source import linalg_summary, model_op_demands

_GENERIC = (
    "  %0 = linalg.generic {{indexing_maps = [{maps}], iterator_types = [{iters}]}} "
    "ins({ins}) outs({outs}) "
    'attrs = {{prov.op = "{op}", prov.family = "{family}"}} {{ ^bb0: }}\n'
)


def _generic(maps: str, iters: str, ins: str, outs: str, op: str = "matmul", family: str = "contraction") -> str:
    return "module {\n" + _GENERIC.format(maps=maps, iters=iters, ins=ins, outs=outs, op=op, family=family) + "}\n"


_BATCHED_MAPS = (
    "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, "
    "affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, "
    "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"
)
_BATCHED_ITERS = '"parallel", "parallel", "parallel", "reduction"'
_PLAIN_MAPS = (
    "affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>"
)
_PLAIN_ITERS = '"parallel", "parallel", "reduction"'


def _only(text: str):
    (demand,) = model_op_demands(text, "int8")
    return demand


# --- what it reads ---------------------------------------------------------------------------------


def test_a_plain_generic_contraction_yields_its_mkn():
    d = _only(
        _generic(
            _PLAIN_MAPS,
            _PLAIN_ITERS,
            "%a, %b : tensor<1x2048xi8>, tensor<2048x1000xi8>",
            "%c : tensor<1x1000xi32>",
            op="int_matmul",
        )
    )
    assert (d.m, d.k, d.n, d.batch) == (1, 2048, 1000, 1)


def test_a_batched_generic_contraction_yields_the_repeating_unit_and_its_count():
    """The im2col'd convolution shape. The batch is RECORDED, not folded into M: a ``48x1x9 @ 48x9x16``
    contraction has 48 distinct right-hand operands, so a tile certified at ``48x9x16`` would be evidence
    about an operation the program does not contain."""
    d = _only(
        _generic(
            _BATCHED_MAPS,
            _BATCHED_ITERS,
            "%a, %b : tensor<48x1x9xf32>, tensor<48x9x16xf32>",
            "%c : tensor<48x1x16xf32>",
            op="convolution_im2col_matmul",
        )
    )
    assert (d.batch, d.m, d.k, d.n) == (48, 1, 9, 16)


def test_the_iterator_roles_come_from_the_maps_not_from_dim_order():
    """M is the parallel dim only the left operand reads, N the one only the right does -- whichever way
    round they are written."""
    swapped = (
        "affine_map<(d0, d1, d2) -> (d2, d0)>, "  # lhs reads d0 (=N side) ... deliberately reversed
        "affine_map<(d0, d1, d2) -> (d1, d2)>, "
        "affine_map<(d0, d1, d2) -> (d0, d1)>"
    )
    d = _only(
        _generic(
            swapped,
            '"parallel", "parallel", "reduction"',
            "%a, %b : tensor<7x4xf32>, tensor<5x7xf32>",
            "%c : tensor<4x5xf32>",
        )
    )
    assert (d.m, d.k, d.n) == (4, 7, 5)


# --- what it refuses -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "why,maps,iters,ins,outs",
    [
        (
            "a real convolution: the access is an affine expression, not a bare dimension",
            "affine_map<(d0, d1, d2, d3) -> (d0, ((d1 * 2) + d3))>, "
            "affine_map<(d0, d1, d2, d3) -> (d3, d2)>, "
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>",
            '"parallel", "parallel", "parallel", "reduction"',
            "%a, %b : tensor<4x9xf32>, tensor<3x8xf32>",
            "%c : tensor<4x4x8xf32>",
        ),
        (
            "a row-wise reduce: the reduction is read by one operand only",
            "affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>",
            '"parallel", "reduction"',
            "%a : tensor<1x2048xf32>",
            "%c : tensor<1xf32>",
        ),
        (
            "a broadcast: the same iterator twice in one access",
            "affine_map<(d0, d1, d2) -> (d0, d0)>, "
            "affine_map<(d0, d1, d2) -> (d2, d1)>, "
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
            '"parallel", "parallel", "reduction"',
            "%a, %b : tensor<4x4xf32>, tensor<6x5xf32>",
            "%c : tensor<4x5xf32>",
        ),
        (
            "the output does not index the whole parallel domain",
            "affine_map<(d0, d1, d2) -> (d0, d2)>, "
            "affine_map<(d0, d1, d2) -> (d2, d1)>, "
            "affine_map<(d0, d1, d2) -> (d0)>",
            '"parallel", "parallel", "reduction"',
            "%a, %b : tensor<4x6xf32>, tensor<6x5xf32>",
            "%c : tensor<4xf32>",
        ),
        (
            "two operands disagree about one iterator's trip count",
            _PLAIN_MAPS,
            _PLAIN_ITERS,
            "%a, %b : tensor<4x6xf32>, tensor<7x5xf32>",
            "%c : tensor<4x5xf32>",
        ),
        (
            "a window iterator: a kind that is neither parallel nor reduction",
            _PLAIN_MAPS,
            '"parallel", "parallel", "window"',
            "%a, %b : tensor<4x6xf32>, tensor<6x5xf32>",
            "%c : tensor<4x5xf32>",
        ),
    ],
)
def test_an_op_that_is_not_unambiguously_a_contraction_gets_no_extent(why, maps, iters, ins, outs):
    d = _only(_generic(maps, iters, ins, outs))
    assert (d.m, d.k, d.n, d.batch) == (None, None, None, None), why


def test_a_transposed_named_matmul_does_not_masquerade_as_a_plain_one():
    """``linalg.matmul_transpose_b`` describes a TRANSPOSED operand pair, so reading its ins types as
    ``MxK, KxN`` records the wrong shape. Only the exact op name is read."""
    text = (
        "module {\n"
        '  %0 = linalg.matmul_transpose_b {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<8x16xf32>, tensor<32x16xf32>) outs(%c : tensor<8x32xf32>)\n"
        '  %1 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%d, %e : tensor<8x16xf32>, tensor<16x32xf32>) outs(%f : tensor<8x32xf32>)\n"
        "}\n"
    )
    first, second = model_op_demands(text, "fp32")
    assert (first.m, first.k, first.n) == (None, None, None)
    assert (second.m, second.k, second.n) == (8, 16, 32)


def test_a_named_matmul_whose_reduction_extents_disagree_is_refused():
    text = (
        "module {\n"
        '  %0 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<8x16xf32>, tensor<17x32xf32>) outs(%c : tensor<8x32xf32>)\n"
        "}\n"
    )
    assert _only(text).has_shape is False


def test_a_named_matmul_over_a_shared_weight_reads_its_batch():
    """A capture spells a batched contraction over one shared weight as ``linalg.matmul`` with a rank-3
    left operand. The old reader took the first two dims of whatever it found, recording ``1x345x32 @
    32x32`` as ``M=1, K=345, N=32`` -- the right three numbers in the wrong roles, on 40 sites."""
    text = (
        "module {\n"
        '  %0 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%c : tensor<1x345x32xf32>)\n"
        "}\n"
    )
    d = _only(text)
    assert (d.batch, d.m, d.k, d.n) == (1, 345, 32, 32)


def test_a_named_matmul_whose_output_does_not_compose_is_refused():
    text = (
        "module {\n"
        '  %0 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<2x345x32xf32>, tensor<32x32xf32>) outs(%c : tensor<2x345x64xf32>)\n"
        "}\n"
    )
    assert _only(text).has_shape is False


# --- alignment: every fact comes off the tag's own line ----------------------------------------------


def test_the_extent_list_is_one_entry_per_tag():
    text = (
        "module {\n"
        '  %0 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<8x16xf32>, tensor<16x32xf32>) outs(%c : tensor<8x32xf32>)\n"
        '  %1 = tensor.expand_shape {prov.op = "matmul", prov.family = "contraction"} ...\n'
        + _GENERIC.format(
            maps=_PLAIN_MAPS,
            iters=_PLAIN_ITERS,
            ins="%d, %e : tensor<4x6xf32>, tensor<6x5xf32>",
            outs="%f : tensor<4x5xf32>",
            op="matmul",
            family="contraction",
        )
        + "}\n"
    )
    summary = linalg_summary(text)
    lengths = {len(summary[k]) for k in ("prov_ops", "prov_families", "carrying_ops", "carrier_reduces")}
    assert lengths == {len(summary["carrier_extents"])} == {len(summary["carrier_formats"])} == {3}
    assert summary["carrier_extents"] == [(1, 8, 16, 32), None, (1, 4, 6, 5)]


def test_a_generic_tagged_matmul_gets_its_own_shape_not_the_named_matmuls():
    """The positional defect itself. With a counter over the named matmuls, the generic below would be
    handed the ``linalg.matmul``'s ``(8, 16, 32)``."""
    text = (
        "module {\n"
        '  %0 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<8x16xf32>, tensor<16x32xf32>) outs(%c : tensor<8x32xf32>)\n"
        + _GENERIC.format(
            maps=_PLAIN_MAPS,
            iters=_PLAIN_ITERS,
            ins="%d, %e : tensor<345x32xf32>, tensor<32x2xf32>",
            outs="%f : tensor<345x2xf32>",
            op="matmul",
            family="contraction",
        )
        + "}\n"
    )
    named, generic = model_op_demands(text, "fp32")
    assert (named.m, named.k, named.n) == (8, 16, 32)
    assert (generic.m, generic.k, generic.n) == (345, 32, 2)


def test_a_parallel_only_generic_still_gets_no_extent():
    """An im2col GATHER carries its region's ``contraction`` family and is not one. It was excluded from
    the contraction set already; it must not acquire a shape by the back door either."""
    text = _generic(
        _BATCHED_MAPS,
        '"parallel", "parallel", "parallel", "parallel"',
        "%a, %b : tensor<48x1x9xf32>, tensor<48x9x16xf32>",
        "%c : tensor<48x1x16xf32>",
        op="convolution_im2col_matmul",
    )
    d = _only(text)
    assert d.family is None and d.has_shape is False


# --- the real capture ------------------------------------------------------------------------------


def test_microvits_last_unsynthesizable_site_is_closed():
    """``M2_microvit_gemmini`` ran bit-exact with 12 of 12 certifiable tiles passing and failed anyway,
    because its remaining conv-tagged reducing generic had no shape to synthesize a tile from."""
    from merlin.compile.mesh import tile_builder_op

    path = merlin_dir() / "contract/capsules/model/M2_microvit_gemmini/capsule.interface.mlir"
    if not path.is_file():
        pytest.skip("M2_microvit_gemmini is not in this checkout")
    demands = model_op_demands(path.read_text(encoding="utf-8"), "int8")
    contractions = [d for d in demands if d.family == "contraction"]
    assert len(contractions) == 13, "12 named matmuls plus the one reducing conv generic"
    assert all(d.has_shape for d in contractions), "every contraction now carries an extent"
    assert all(tile_builder_op(d)[0] is not None for d in contractions)

    (conv,) = [d for d in contractions if d.op == "convolution_im2col_matmul"]
    assert (conv.batch, conv.m, conv.k, conv.n) == (48, 1, 9, 16)
    # ...and the tile it now synthesizes is in the op's OWN format, not the model's declared int8.
    assert conv.in_fmt == "int8" and conv.elem_fmt == "fp32" and conv.tile_fmt == "fp32"
