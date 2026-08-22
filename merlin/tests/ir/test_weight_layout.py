"""Run-time weight re-layouts are found and priced, and the unsafe ones are not offered.

`test_an_argument_read_elsewhere_is_not_hoistable` is the one that matters: pre-transposing a weight
is only sound when the transpose is that argument's sole consumer, because anything else reading the
argument would silently begin seeing transposed data.
"""
from __future__ import annotations

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.xdsl_dialects.lowering.weight_layout import weight_layout_report

SOLE_USE = """
module {
  func.func @forward(%w: tensor<256000x2304xi8>) -> tensor<2304x256000xi8> {
    %e = tensor.empty() : tensor<2304x256000xi8>
    %t = linalg.transpose ins(%w:tensor<256000x2304xi8>) outs(%e:tensor<2304x256000xi8>)
         permutation = [1, 0]
    return %t : tensor<2304x256000xi8>
  }
}
"""

# the same weight is ALSO read untransposed -- pre-transposing would corrupt the second reader
TWO_READERS = """
module {
  func.func @forward(%w: tensor<4x8xi8>) -> (tensor<8x4xi8>, tensor<4x8xi8>) {
    %e = tensor.empty() : tensor<8x4xi8>
    %t = linalg.transpose ins(%w:tensor<4x8xi8>) outs(%e:tensor<8x4xi8>) permutation = [1, 0]
    %e2 = tensor.empty() : tensor<4x8xi8>
    %c = linalg.copy ins(%w : tensor<4x8xi8>) outs(%e2 : tensor<4x8xi8>) -> tensor<4x8xi8>
    return %t, %c : tensor<8x4xi8>, tensor<4x8xi8>
  }
}
"""

# a transpose of a COMPUTED tensor is real work, not a layout the packer could have chosen
COMPUTED = """
module {
  func.func @forward(%a: tensor<4x8xi8>) -> tensor<8x4xi8> {
    %e0 = tensor.empty() : tensor<4x8xi8>
    %c = linalg.copy ins(%a : tensor<4x8xi8>) outs(%e0 : tensor<4x8xi8>) -> tensor<4x8xi8>
    %e = tensor.empty() : tensor<8x4xi8>
    %t = linalg.transpose ins(%c:tensor<4x8xi8>) outs(%e:tensor<8x4xi8>) permutation = [1, 0]
    return %t : tensor<8x4xi8>
  }
}
"""


def _report(src: str):
    with IR_LOCK:
        return weight_layout_report(mq.parse(src))


def test_a_sole_use_weight_transpose_is_hoistable_and_priced():
    rep = _report(SOLE_USE)
    assert len(rep.relayouts) == 1
    r = rep.relayouts[0]
    assert r.hoistable and r.reason == ""
    assert r.shape == [256000, 2304] and r.result_shape == [2304, 256000]
    assert r.dtype == "i8"
    assert r.bytes_moved == 256000 * 2304        # the 562.5 MiB head
    assert rep.hoistable_bytes == 256000 * 2304
    assert rep.unpriceable == []


def test_an_argument_read_elsewhere_is_not_hoistable():
    """The soundness condition: another reader would silently see transposed data."""
    rep = _report(TWO_READERS)
    assert len(rep.relayouts) == 1
    assert rep.blocked and not rep.hoistable
    assert "consumers" in rep.blocked[0].reason
    assert rep.hoistable_bytes == 0, "an unsafe re-layout must not be counted as a saving"
    assert rep.total_bytes == 4 * 8, "but it is still real traffic and is still reported"


def test_transposing_a_computed_tensor_is_not_a_weight_relayout():
    rep = _report(COMPUTED)
    assert rep.relayouts == []
    assert rep.hoistable_bytes == 0


def test_bytes_are_priced_by_element_width():
    src = SOLE_USE.replace("xi8", "xf32")
    rep = _report(src)
    assert rep.relayouts[0].dtype == "f32"
    assert rep.relayouts[0].bytes_moved == 256000 * 2304 * 4


def test_a_model_with_no_transposes_reports_nothing():
    src = """
    module {
      func.func @forward(%a: tensor<4x8xi8>) -> tensor<4x8xi8> { return %a : tensor<4x8xi8> }
    }
    """
    rep = _report(src)
    assert rep.relayouts == [] and rep.hoistable_bytes == 0 and rep.total_bytes == 0
