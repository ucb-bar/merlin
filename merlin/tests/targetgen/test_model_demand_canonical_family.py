"""Model demand uses the verified operation's family, not its routing spelling."""

# ruff: noqa: E501 -- the typed MLIR fixture preserves its exact operation serialization.

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from merlin.targetgen import eligibility
from merlin.targetgen.capsule_source import model_accelerator_demand
from merlin.targetgen.compute_units import SemanticCapability

_TYPED_CONTRACTION = """builtin.module {
  func.func @forward(%a: tensor<4x19xi8>, %b: tensor<19x8xi8>) -> tensor<4x8xi32> {
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %init = "tensor.splat"(%zero) : (i32) -> tensor<4x8xi32>
    %result = "linalg.generic"(%a, %b, %init) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs32 = "arith.extsi"(%lhs) : (i8) -> i32
      %rhs32 = "arith.extsi"(%rhs) : (i8) -> i32
      %product = "arith.muli"(%lhs32, %rhs32) : (i32, i32) -> i32
      %sum = "arith.addi"(%acc, %product) : (i32, i32) -> i32
      "linalg.yield"(%sum) : (i32) -> ()
    }) {prov.op = "int_matmul", prov.family = "contraction"} : (tensor<4x19xi8>, tensor<19x8xi8>, tensor<4x8xi32>) -> tensor<4x8xi32>
    func.return %result : tensor<4x8xi32>
  }
}"""


def _binding() -> SimpleNamespace:
    return SimpleNamespace(
        target="neutral_test",
        operand_dtype="int8",
        cap_dtype=lambda dtype: dtype,
        classes_for=lambda *, op, output_dtype: ["LOAD", "COMPUTE", "STORE"] if op == "int_matmul" else [],
    )


def test_typed_integer_contraction_derives_demand_from_canonical_family(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    assert model_accelerator_demand(_TYPED_CONTRACTION, _binding()) == (
        "contraction",
        ["LOAD", "COMPUTE", "STORE"],
    )


def test_unknown_source_family_cannot_create_accelerator_demand(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    unknown = _TYPED_CONTRACTION.replace('prov.family = "contraction"', 'prov.family = "unregistered"')
    assert model_accelerator_demand(unknown, _binding()) == (None, [])


def test_a_contraction_label_on_noncontracting_ir_cannot_create_demand(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    noncontracting = _TYPED_CONTRACTION.replace(
        "#linalg.iterator_type<reduction>", "#linalg.iterator_type<parallel>"
    )
    assert model_accelerator_demand(noncontracting, _binding()) == (None, [])


def test_an_undeclared_dtype_cannot_create_demand(monkeypatch) -> None:
    monkeypatch.setattr(
        eligibility,
        "capability_map_for_target",
        lambda target: {"contraction": SemanticCapability(family="contraction", dtypes=("fp32",))},
    )
    assert model_accelerator_demand(_TYPED_CONTRACTION, _binding()) == (None, [])


def _role_binding():
    from merlin.targetgen.corpus_instruction_roles import selected_classes

    roster = [{"class": name} for name in ("FETCH", "RETAIN", "PRODUCT", "PUBLISH")]
    contract = {
        "rocc_operand_roles": {"instructions": roster},
        "corpus_instruction_roles": {
            "version": 1,
            "instruction_roster_sha256": hashlib.sha256(
                json.dumps(roster, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "by_role": {"memory": ["FETCH"], "weight_load": ["RETAIN"],
                        "matmul": ["PRODUCT"], "acc_readout": ["PUBLISH"]},
        },
    }
    binding = _binding()
    binding.classes_for = selected_classes(contract)
    return binding


@pytest.mark.parametrize("tag", ["int_matmul", "matmul"])
def test_checked_integer_source_uses_original_generic_role_deriver(monkeypatch, tag):
    monkeypatch.setattr(
        eligibility, "capability_map_for_target",
        lambda _: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    source = _TYPED_CONTRACTION.replace('prov.op = "int_matmul"', f'prov.op = "{tag}"')
    assert model_accelerator_demand(source, _role_binding()) == (
        "contraction", ["FETCH", "RETAIN", "PRODUCT", "PUBLISH"],
    )


@pytest.mark.parametrize("fault", ["unknown_tag", "unknown_family", "noncontracting", "missing_class"])
def test_integer_role_bridge_keeps_original_source_and_role_refusals(monkeypatch, fault):
    monkeypatch.setattr(
        eligibility, "capability_map_for_target",
        lambda _: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    binding, source = _role_binding(), _TYPED_CONTRACTION
    if fault == "unknown_tag":
        source = source.replace('prov.op = "int_matmul"', 'prov.op = "unregistered_dot"')
    elif fault == "unknown_family":
        source = source.replace('prov.family = "contraction"', 'prov.family = "unregistered"')
    elif fault == "noncontracting":
        source = source.replace("#linalg.iterator_type<reduction>", "#linalg.iterator_type<parallel>")
    else:
        binding.classes_for = lambda **_: []
    assert not model_accelerator_demand(source, binding)[1]


def test_existing_integer_class_provider_retains_its_original_choice(monkeypatch):
    monkeypatch.setattr(
        eligibility, "capability_map_for_target",
        lambda _: {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))},
    )
    binding = _binding()
    binding.classes_for = lambda *, op, output_dtype: ["OWN_INTEGER"] if op == "int_matmul" else ["OTHER"]
    assert model_accelerator_demand(_TYPED_CONTRACTION, binding) == ("contraction", ["OWN_INTEGER"])
