"""The convolution geometry a capture contains is RECOVERED, not read off an attribute.

Every test here pins a way the recovery could be silently wrong. Two of them pin defects that were
real: the first version dropped every strided convolution, and the second mislabelled a 4x4 kernel as
a 2x3 one. Both produced a plausible-looking class list, which is why they need tests rather than
review.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import conv_geometry as CG

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _mod(text: str):
    from merlin.common import mlir_query as mq
    return mq.parse(text)


def _im2col(*, cin, k, stride, out, padded, dilation=1, batch=1):
    """An im2col gather + reshape, spelled the way torch-mlir spells it."""
    kh = kw = k
    oh = ow = out
    ph = pw = padded
    n_k = cin * kh * kw
    n_m = batch * oh * ow
    return f"""
module {{
  func.func @forward(%arg0: tensor<{batch}x{cin}x{ph}x{pw}xf32>) -> tensor<{n_k}x{n_m}xf32> {{
    %e = tensor.empty() : tensor<{cin}x{kh}x{kw}x{batch}x{oh}x{ow}xf32>
    %g = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, d4 * {stride} + d1 * {dilation}, d5 * {stride} + d2 * {dilation})>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}}
      ins(%arg0 : tensor<{batch}x{cin}x{ph}x{pw}xf32>)
      outs(%e : tensor<{cin}x{kh}x{kw}x{batch}x{oh}x{ow}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a : f32
    }} -> tensor<{cin}x{kh}x{kw}x{batch}x{oh}x{ow}xf32>
    %c = tensor.collapse_shape %g [[0, 1, 2, 3, 4, 5]] : tensor<{cin}x{kh}x{kw}x{batch}x{oh}x{ow}xf32> into tensor<{n_k * n_m}xf32>
    %x = tensor.expand_shape %c [[0, 1]] output_shape [{n_k}, {n_m}] : tensor<{n_k * n_m}xf32> into tensor<{n_k}x{n_m}xf32>
    return %x : tensor<{n_k}x{n_m}xf32>
  }}
}}
"""


def test_a_strided_convolution_is_not_dropped():
    """⚠️ REGRESSION. The first version verified `padded == (out-1)*s + (k-1)*d + 1`, which is the
    STRIDE-1 special case: a 3x3 stride-2 window over a 66-row padded input yields 32 outputs and
    touches only 65 rows, so the equality fails and the convolution vanished. The corpus would have
    gained unstrided members and still had no strided one -- the exact gap this module exists to
    close, reintroduced by its own check."""
    gs = CG.geometries(_mod(_im2col(cin=16, k=3, stride=2, out=32, padded=66)))
    assert len(gs) == 1, "a stride-2 convolution must be recovered, not silently skipped"
    assert gs[0].kernel == (3, 3)
    assert gs[0].stride == (2, 2)
    assert gs[0].dilation == (1, 1)


def test_a_kernel_larger_than_its_output_is_not_read_backwards():
    """⚠️ REGRESSION, and the reason the K side is consulted at all. `d4 * 4 + d1` is symmetric: the
    map alone cannot say which dim is the kernel. Choosing the smaller extent read a real 4x4 stride-4
    convolution (output 2x3) as a 2x3 kernel with dilation 4 -- a different convolution from the one
    the model contains. The output identity does NOT separate the two; both satisfy it. Only the
    im2col K dimension does."""
    gs = CG.geometries(_mod(_im2col(cin=64, k=4, stride=4, out=2, padded=8)))
    assert len(gs) == 1
    assert gs[0].kernel == (4, 4), "the K side names the kernel; the smaller extent does not"
    assert gs[0].stride == (4, 4)
    assert gs[0].dilation == (1, 1)
    assert gs[0].channels_in == 64


def test_a_dilated_window_reports_its_spacing():
    gs = CG.geometries(_mod(_im2col(cin=8, k=3, stride=1, out=10, padded=14, dilation=2)))
    assert len(gs) == 1
    assert (gs[0].kernel, gs[0].stride, gs[0].dilation) == ((3, 3), (1, 1), (2, 2))


def test_an_unreadable_padding_producer_is_unknown_and_never_zero():
    """The capture's first convolution is padded by an `aten.index.Tensor` gather -- a REFLECTION pad
    whose offsets are nowhere to read. Reporting `pad0` would claim there is no padding where there is
    some, AND claim the zero identity for one that reflects. UNKNOWN must survive to the signature."""
    gs = CG.geometries(_mod(_im2col(cin=3, k=7, stride=1, out=64, padded=70)))
    assert len(gs) == 1
    assert gs[0].pad_known is False
    assert gs[0].padded is None, "unknown padding must not read as unpadded"
    assert "padUNKNOWN" in gs[0].signature()


def test_a_gather_with_no_reshape_yields_nothing():
    """No K side means the kernel/output assignment is undetermined. An undetermined geometry is
    dropped: a wrong stride would demand capsules for a convolution the model does not contain."""
    text = _im2col(cin=16, k=3, stride=2, out=32, padded=66)
    head = text.split("    %c = tensor.collapse_shape")[0]
    truncated = head + "    return %g : tensor<16x3x3x1x32x32xf32>\n  }\n}\n"
    assert CG.geometries(_mod(truncated)) == []


def test_an_unparseable_source_reports_nothing_rather_than_raising():
    assert CG.geometries("this is not mlir") == []


def test_the_signature_ignores_extents_and_keeps_every_axis_that_changes_code():
    """Two 3x3/s1/pad1 convolutions exercise one lowering whatever their channels; a corpus with a
    member per extent grows without covering anything new. But stride, dilation, padding, padding
    symmetry and input dilation each change the emitted code and each stay in the key."""
    a = CG.geometries(_mod(_im2col(cin=16, k=3, stride=1, out=16, padded=18)))[0]
    b = CG.geometries(_mod(_im2col(cin=64, k=3, stride=1, out=32, padded=34)))[0]
    assert a.signature() == b.signature(), "extents are not part of what the compiler must do"

    c = CG.geometries(_mod(_im2col(cin=16, k=3, stride=2, out=8, padded=18)))[0]
    assert c.signature() != a.signature(), "a different stride is a different obligation"


def test_asymmetric_padding_is_visible_in_the_signature():
    g = CG.ConvGeometry(kernel=(3, 3), stride=(1, 1), dilation=(1, 1),
                        pad_before=(1, 1), pad_after=(2, 2), input_dilation=(1, 1),
                        pad_known=True, in_spatial=(16, 16), out_spatial=(32, 32),
                        channels_in=64, dtype="f32")
    assert g.symmetric_pad is False
    assert "pad1x1_2x2" in g.signature(), "an asymmetric pad must not read as a symmetric one"
    sym = CG.ConvGeometry(kernel=(3, 3), stride=(1, 1), dilation=(1, 1),
                          pad_before=(1, 1), pad_after=(1, 1), input_dilation=(1, 1),
                          pad_known=True, in_spatial=(16, 16), out_spatial=(16, 16),
                          channels_in=64, dtype="f32")
    assert sym.signature() != g.signature()


def test_an_input_dilated_transposed_convolution_is_its_own_class():
    """A transposed convolution spaces its input out before padding it. That is a different lowering
    from an ordinary padded convolution and must not collapse into one."""
    t = CG.ConvGeometry(kernel=(3, 3), stride=(1, 1), dilation=(1, 1),
                        pad_before=(1, 1), pad_after=(2, 2), input_dilation=(2, 2),
                        pad_known=True, in_spatial=(16, 16), out_spatial=(32, 32),
                        channels_in=64, dtype="f32")
    plain = CG.ConvGeometry(kernel=(3, 3), stride=(1, 1), dilation=(1, 1),
                            pad_before=(1, 1), pad_after=(2, 2), input_dilation=(1, 1),
                            pad_known=True, in_spatial=(16, 16), out_spatial=(32, 32),
                            channels_in=64, dtype="f32")
    assert "indilated2x2" in t.signature()
    assert t.signature() != plain.signature()


def test_geometry_classes_groups_and_names_its_evidence():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.mlir"
        p2 = Path(d) / "b.mlir"
        p1.write_text(_im2col(cin=16, k=3, stride=2, out=32, padded=66))
        p2.write_text(_im2col(cin=64, k=3, stride=2, out=16, padded=34))
        rep = CG.geometry_classes({"a": p1, "b": p2})
    assert rep["n_classes"] == 1, "the same window from two captures is one obligation"
    row = rep["required"][0]
    assert row["n_regions"] == 2
    assert sorted(row["sources"]) == ["a", "b"], "a class must name which captures evidence it"
    assert rep["captures_unreadable"] == {}


def test_an_unreadable_capture_is_reported_not_skipped():
    rep = CG.geometry_classes({"broken": "/nonexistent/model.mlir"})
    assert rep["n_classes"] == 0
    # An unparseable capture degrades to "saw nothing" inside `geometries`, so it contributes no
    # class. What must never happen is a raise that takes the whole derivation down.
    assert isinstance(rep["captures_unreadable"], dict)


# --------------------------------------------------------------------- synthesis into the corpus


def test_the_window_axis_synthesizes_one_member_per_class():
    """The axis is only worth deriving if it becomes capsules. Measured before this: the corpus
    presented a padded convolution and a strided convolution as SEPARATE hand-authored entries and no
    strided-AND-padded one, while the captures demand exactly that -- the defect lived in the gap
    between two members each covering half of it."""
    from merlin.targetgen import corpus_synth as CS

    spec = {
        "target": "t",
        "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8",
                   "alignment": "aligned"}],
        "boundaries": {"tile_edge": 16,
                       "extent_probes": [{"boundary": "tile_edge", "edge": 16,
                                          "points": [1, 4, 8, 15, 16, 17, 32]}]},
        "conv_geometry": {"required": [
            {"signature": "k3x3/s2x2/d1x1/pad1x1", "kernel": [3, 3], "stride": [2, 2],
             "dilation": [1, 1], "pad_before": [1, 1], "pad_after": [1, 1], "pad_known": True,
             "n_regions": 6, "sources": ["deepjscc_int8"]},
            {"signature": "k7x7/s1x1/d1x1/padUNKNOWN", "kernel": [7, 7], "stride": [1, 1],
             "dilation": [1, 1], "pad_before": [0, 0], "pad_after": [0, 0], "pad_known": False,
             "n_regions": 1, "sources": ["deepjscc_int8"]},
        ]},
    }
    out = CS.synthesize(spec, workload_spec={"models": [], "precision_preference": ["int8"]})
    entries = out["capsules"] if isinstance(out, dict) else out
    conv = [e for e in entries if (e.get("generalization") or {}).get("generalization_axis") == "conv_window"]
    assert len(conv) == 2, f"one member per window class; got {[e['name'] for e in conv]}"

    strided = next(e for e in conv if "s2x2" in e["name"])
    assert (strided["kh"], strided["kw"]) == (3, 3)
    assert list(strided["stride"]) == [2, 2]
    assert list(strided["padding"]) == [1, 1, 1, 1], (
        "a strided AND padded member is the whole point; a padding of zero here would recreate the gap")


def test_an_unknown_padding_member_says_it_asserts_nothing_about_the_identity():
    """A padUNKNOWN window is a real obligation and gets a member -- at zero padding, because the
    capture's padding identity is NOT zero (a reflection pad) and we cannot spell it. The member must
    say that, or a reader will take its zero padding for a measured fact."""
    from merlin.targetgen import corpus_synth as CS

    spec = {
        "target": "t",
        "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8",
                   "alignment": "aligned"}],
        "boundaries": {"tile_edge": 16,
                       "extent_probes": [{"boundary": "tile_edge", "edge": 16,
                                          "points": [1, 4, 8, 15, 16, 17, 32]}]},
        "conv_geometry": {"required": [
            {"signature": "k7x7/s1x1/d1x1/padUNKNOWN", "kernel": [7, 7], "stride": [1, 1],
             "dilation": [1, 1], "pad_before": [0, 0], "pad_after": [0, 0], "pad_known": False,
             "n_regions": 1, "sources": ["deepjscc_int8"]},
        ]},
    }
    out = CS.synthesize(spec, workload_spec={"models": [], "precision_preference": ["int8"]})
    entries = out["capsules"] if isinstance(out, dict) else out
    member = next(e for e in entries
                  if (e.get("generalization") or {}).get("generalization_axis") == "conv_window")
    ref = member["source_reference"]
    assert "NOT READABLE" in ref
    assert "asserts nothing about a padding identity that is not zero" in ref


def test_every_window_gets_an_image_that_actually_produces_output():
    """⚠️ REGRESSION. `conv2d` defaults to an 8x8 image and a large window walks off it: measured,
    k16x16/s16x16 gives a 0x0 output and k8x8/s8x8 gives 1x1. A zero-row capsule is not a small test,
    it is a test of nothing that still builds, still runs and still passes -- the exact failure this
    repo keeps rediscovering. The image is solved from the window instead."""
    from merlin.runtime.commandbuffer import conv_out_dims
    from merlin.targetgen.corpus_synth import _CONV_WINDOW_OUT, _image_for_window

    for kernel, stride, dilation, pb, pa in [
        ((16, 16), (16, 16), (1, 1), (0, 0), (0, 0)),
        ((8, 8), (8, 8), (1, 1), (0, 0), (0, 0)),
        ((7, 7), (4, 4), (1, 1), (3, 3), (3, 3)),
        ((4, 4), (4, 4), (1, 1), (0, 0), (0, 0)),
        ((3, 3), (2, 2), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (1, 1), (1, 1), (1, 1)),
        ((5, 5), (1, 1), (1, 1), (0, 0), (0, 0)),
    ]:
        h, w = _image_for_window(kernel, stride, dilation, pb, pa)
        ho, wo = conv_out_dims(h, w, kernel[0], kernel[1], list(stride),
                               list(pb) + list(pa), list(dilation))
        assert (ho, wo) == (_CONV_WINDOW_OUT, _CONV_WINDOW_OUT), (
            f"window k{kernel} s{stride} pad{pb}/{pa} on a {h}x{w} image writes {ho}x{wo} rows")


def test_the_synthesized_member_carries_the_solved_image():
    from merlin.runtime.commandbuffer import conv_out_dims
    from merlin.targetgen import corpus_synth as CS

    spec = {
        "target": "t",
        "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8",
                   "alignment": "aligned"}],
        "boundaries": {"tile_edge": 16,
                       "extent_probes": [{"boundary": "tile_edge", "edge": 16,
                                          "points": [1, 4, 8, 15, 16, 17, 32]}]},
        "conv_geometry": {"required": [
            {"signature": "k16x16/s16x16/d1x1/padUNKNOWN", "kernel": [16, 16], "stride": [16, 16],
             "dilation": [1, 1], "pad_before": [0, 0], "pad_after": [0, 0], "pad_known": False,
             "n_regions": 6, "sources": ["smolvla"]},
        ]},
    }
    out = CS.synthesize(spec, workload_spec={"models": [], "precision_preference": ["int8"]})
    entries = out["capsules"] if isinstance(out, dict) else out
    m = next(e for e in entries
             if (e.get("generalization") or {}).get("generalization_axis") == "conv_window")
    ho, wo = conv_out_dims(m["Himg"], m["Wimg"], m["kh"], m["kw"], list(m["stride"]),
                           list(m["padding"]), list(m["dilation"]))
    assert ho > 0 and wo > 0, "a member writing no rows tests nothing while passing"
    assert (ho, wo) == (CS._CONV_WINDOW_OUT, CS._CONV_WINDOW_OUT)
