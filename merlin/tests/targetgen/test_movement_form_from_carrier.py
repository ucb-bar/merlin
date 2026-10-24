"""A movement region's form is read from its IR carrier and indexing maps, not its aten name.

WHY THIS EXISTS. ``operation_form`` named a form only for ``linalg.copy``/``memref.copy`` (copy) and
the transpose carriers (permutation), and returned None for every other carrier. A movement capability
that declares ``forms`` refuses a None form, so on realistic captures -- a fill is ``linalg.fill`` or
``tensor.splat``, cat is ``tensor.concat``, a scatter is ``tensor.insert_slice``, a pad is
``tensor.pad``, clone is an identity yield-only ``linalg.generic`` -- every movement op but permute was
refused although the hardware's DMA forms cover all of them.
"""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.targetgen import eligibility as el
from merlin.targetgen import semantic_families as sf

_ID = "affine_map<(d0, d1) -> (d0, d1)>"


@pytest.mark.parametrize(
    "carrier, extra, form",
    [
        ("linalg.copy", {}, "copy"),
        ("tensor.concat", {}, "copy"),
        ("tensor.insert_slice", {}, "copy"),
        ("tensor.extract_slice", {}, "copy"),
        ("linalg.transpose", {}, "permutation"),
        ("linalg.broadcast", {}, "broadcast"),
        ("linalg.fill", {"constant_zero": True}, "zero_fill"),
        ("linalg.fill", {"constant_zero": False}, "broadcast"),
        ("linalg.fill", {}, "broadcast"),
        ("tensor.splat", {"constant_zero": True}, "zero_fill"),
        ("tensor.pad", {"constant_zero": True}, "copy+zero_fill"),
        ("tensor.pad", {}, "broadcast+copy"),
        ("tensor.empty", {}, None),
    ],
)
def test_named_carriers(carrier, extra, form):
    assert sf.operation_form(None, carrier_op=carrier, **extra) == form


@pytest.mark.parametrize(
    "source_map, body, form",
    [
        (_ID, ["linalg.yield"], "copy"),
        ("affine_map<(d0, d1) -> (d1, d0)>", ["linalg.yield"], "permutation"),
        ("affine_map<(d0, d1) -> (d1)>", ["linalg.yield"], "broadcast"),
        ("affine_map<(d0, d1) -> (0, d1)>", ["linalg.yield"], "broadcast"),
        ("affine_map<(d0, d1) -> (d0)>", ["linalg.yield"], "broadcast"),
        # An offset or a reflected index is a gather, not a form this classifier states.
        ("affine_map<(d0, d1) -> (d0 + 1, d1)>", ["linalg.yield"], None),
        (_ID, ["tensor.extract", "linalg.yield"], None),
        (_ID, ["arith.addi", "linalg.yield"], None),
    ],
)
def test_yield_only_generic_form_comes_from_its_maps(source_map, body, form):
    observed = sf.operation_form(
        None, carrier_op="linalg.generic", indexing_maps=[source_map, _ID], body_operations=body, yields_input=True
    )
    assert observed == form


def test_a_generic_without_its_maps_stays_unknown():
    assert sf.operation_form("clone", carrier_op="linalg.generic") is None


_IR = """
module {
  func.func @f(%x: tensor<4x8xi8>, %r: tensor<8xi8>) -> tensor<4x8xi8> {
    %z = arith.constant 0 : i8
    %c = arith.constant 3 : i8
    %e = tensor.empty() : tensor<4x8xi8>
    %f0 = linalg.fill ins(%z : i8) outs(%e : tensor<4x8xi8>) -> tensor<4x8xi8>
    %f1 = linalg.fill ins(%c : i8) outs(%e : tensor<4x8xi8>) -> tensor<4x8xi8>
    %id = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel"]}
          ins(%x : tensor<4x8xi8>) outs(%e : tensor<4x8xi8>) {
    ^bb0(%a: i8, %o: i8):
      linalg.yield %a : i8
    } -> tensor<4x8xi8>
    %bc = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel"]}
          ins(%r : tensor<8xi8>) outs(%e : tensor<4x8xi8>) {
    ^bb0(%a: i8, %o: i8):
      linalg.yield %a : i8
    } -> tensor<4x8xi8>
    %fl = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          outs(%e : tensor<4x8xi8>) {
    ^bb0(%o: i8):
      linalg.yield %z : i8
    } -> tensor<4x8xi8>
    return %id : tensor<4x8xi8>
  }
}
"""


def test_carrier_form_reads_the_parsed_operation():
    with IR_LOCK:
        module = mq.parse(_IR)
        forms = [sf.carrier_form(op) for op in mq.walk(module) if op.name in {"linalg.fill", "linalg.generic"}]
    assert forms == ["zero_fill", "broadcast", "copy", "broadcast", "zero_fill"]


def test_a_joined_form_needs_every_part():
    assert sf.form_admitted("copy+zero_fill", ("copy", "zero_fill"))
    assert not sf.form_admitted("copy+zero_fill", ("copy",))
    assert not sf.form_admitted(None, ("copy",))
    assert sf.from_op("broadcast") == "movement"


@pytest.mark.target("gemmini")
@pytest.mark.parametrize("form", ["copy", "permutation", "broadcast", "zero_fill", "copy+zero_fill"])
def test_gemmini_working_tree_movement_forms_cover_the_carriers(form):
    from merlin.common.paths import repo_root
    from merlin.common.yaml import load_yaml

    contract = load_yaml(repo_root() / "examples/gemmini/target/contracts/target_contract.yaml")
    cmap = el.capability_map_from_contract(contract)
    if set(cmap["movement"].forms) < {"copy", "permutation", "broadcast", "zero_fill"}:
        pytest.skip("the selected contract does not declare the four DMA movement forms")
    region = el.RegionDescriptor(op="movement", family="movement", in_dtype="int8", rank=4, form=form)
    assert el.is_eligible(region, cmap).eligible
