"""A host window gather is read back through a pure element copy, and through nothing else.

The pieces of a captured patch matrix may carry a materialising clone between the NCHW->NHWC
transpose and the flattening view. Only a copy that cannot change a value is transparent; every
other op in that position keeps the concatenated-gather refusal.
"""

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.llvmlower.group_host_call import _host_window_gather

_PAR4 = ", ".join(["#linalg.iterator_type<parallel>"] * 4)
_ID4 = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>"
_SWAP4 = "affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>"


def _copy(n: int, src: str, *, variant: str) -> tuple[str, str, str]:
    """Text of the op between transpose and collapse, its result name and element type."""
    t8 = "tensor<1x2x2x2xi8>"
    init = f'%ce{n} = "tensor.empty"() : () -> {t8}'
    maps, kinds, body, out_t, operands, seg, args = (
        f"[{_ID4}, {_ID4}]",
        f"[{_PAR4}]",
        f'"linalg.yield"(%a{n}) : (i8) -> ()',
        t8,
        f"%{src}, %ce{n}",
        "1, 1",
        f"%a{n}: i8, %b{n}: i8",
    )
    if variant == "yield_init":
        body = f'"linalg.yield"(%b{n}) : (i8) -> ()'
    elif variant == "permuted_map":
        maps = f"[{_ID4}, {_SWAP4}]"
    elif variant == "arith_body":
        body = f'%x{n} = "arith.addi"(%a{n}, %a{n}) : (i8, i8) -> i8\n  "linalg.yield"(%x{n}) : (i8) -> ()'
    elif variant == "reduction":
        kinds = "[" + ", ".join(["#linalg.iterator_type<parallel>"] * 3 + ["#linalg.iterator_type<reduction>"]) + "]"
    elif variant == "widening":
        init = f'%ce{n} = "tensor.empty"() : () -> tensor<1x2x2x2xi16>'
        out_t = "tensor<1x2x2x2xi16>"
        args = f"%a{n}: i8, %b{n}: i16"
        body = f'%x{n} = "arith.extsi"(%a{n}) : (i8) -> i16\n  "linalg.yield"(%x{n}) : (i16) -> ()'
    elif variant == "two_inputs":
        maps, operands, seg, args = f"[{_ID4}, {_ID4}, {_ID4}]", f"%{src}, %{src}, %ce{n}", "2, 1", f"%a{n}: i8, %c{n}: i8, %b{n}: i8"
    in_types = ", ".join([t8] * (3 if variant == "two_inputs" else 2)) if variant != "widening" else f"{t8}, {out_t}"
    text = f"""{init}
    %cp{n} = "linalg.generic"({operands}) <{{indexing_maps = {maps}, iterator_types = {kinds},
        operandSegmentSizes = array<i32: {seg}>}}> ({{
    ^bb0({args}):
      {body}
    }}) : ({in_types}) -> {out_t}"""
    return text, f"cp{n}", out_t.split("x")[-1].rstrip(">")


def _gather(variant: str | None):
    """A 2x2-window, stride-2 patch matrix of one 1x2x4x4 image: 4 positions by 8 reduction rows."""
    lines, pieces = [], []
    dt = "i8"
    for n, (th, tw) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        lines.append(
            f'%s{n} = "tensor.extract_slice"(%img) <{{static_offsets = array<i64: 0, 0, {th}, {tw}>, '
            "static_sizes = array<i64: 1, 2, 2, 2>, static_strides = array<i64: 1, 1, 2, 2>, "
            "operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x2x4x4xi8>) -> tensor<1x2x2x2xi8>"
        )
        lines.append(f'%e{n} = "tensor.empty"() : () -> tensor<1x2x2x2xi8>')
        lines.append(
            f'%t{n} = "linalg.transpose"(%s{n}, %e{n}) <{{permutation = array<i64: 0, 2, 3, 1>}}> ({{\n'
            f'    ^bb0(%p{n}: i8, %q{n}: i8):\n      "linalg.yield"(%p{n}) : (i8) -> ()\n'
            "    }) : (tensor<1x2x2x2xi8>, tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8>"
        )
        last = f"t{n}"
        if variant is not None:
            text, last, dt = _copy(n, last, variant=variant)
            lines.append(text)
        lines.append(
            f'%f{n} = "tensor.collapse_shape"(%{last}) <{{reassociation = [[0, 1, 2, 3]]}}> '
            f": (tensor<1x2x2x2x{dt}>) -> tensor<8x{dt}>"
        )
        lines.append(
            f'%v{n} = "tensor.expand_shape"(%f{n}) <{{reassociation = [[0, 1]], static_output_shape = array<i64: 4, 2>}}> '
            f": (tensor<8x{dt}>) -> tensor<4x2x{dt}>"
        )
        pieces.append(f"%v{n}")
    lines.append(
        f'%cat = "tensor.concat"({", ".join(pieces)}) <{{dim = 1 : i64}}> : '
        f"({', '.join([f'tensor<4x2x{dt}>'] * 4)}) -> tensor<4x8x{dt}>"
    )
    body = "\n    ".join(lines)
    text = f"""builtin.module {{
  func.func @f(%img: tensor<1x2x4x4xi8>) -> tensor<4x8x{dt}> {{
    {body}
    func.return %cat : tensor<4x8x{dt}>
  }}
}}"""
    module = mq.parse(text)
    return next(op for op in module.walk() if op.name == "tensor.concat").results[0]


def test_the_plain_spelling_is_read_back_as_a_convolution():
    gather = _host_window_gather(_gather(None), 4, 8)
    assert gather["window"] == {"ci": 2, "Himg": 4, "Wimg": 4, "kh": 2, "kw": 2, "stride": [2, 2], "padding": [0, 0, 0, 0]}


def test_a_pure_copy_between_transpose_and_view_reads_back_the_same_convolution():
    plain, copied = _host_window_gather(_gather(None), 4, 8), _host_window_gather(_gather("identity"), 4, 8)
    assert copied["window"] == plain["window"]
    assert np.array_equal(copied["rows"], plain["rows"])
    # The device is handed the very image the gather read, not anything the copy produced.
    assert type(copied["image"]).__name__ == type(plain["image"]).__name__ == "BlockArgument"


@pytest.mark.parametrize("variant", ["yield_init", "permuted_map", "arith_body", "reduction", "widening", "two_inputs"])
def test_anything_that_can_change_a_value_keeps_the_refusal(variant):
    with pytest.raises(ValueError, match="concatenated gather whose image cannot be read back: not an NCHW->NHWC transpose"):
        _host_window_gather(_gather(variant), 4, 8)
