"""Actual artifact/model binding and live Lean controls for authored checks."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.planning import blocks_in
from merlin.verify import lean_checked as CHECK
from merlin.verify.lean_contract import LeanToolchain, LeanTransformReceipt, coverage_policy_record
from merlin.verify.lean_state_effects import EffectScheduleContract

BEFORE = """builtin.module { func.func @f(%a:i8,%b:i8)->i8 {
  %r = arith.addi %a,%b : i8
  func.return %r : i8
}}"""
AFTER = BEFORE.replace("%a,%b", "%b,%a")


def _contract():
    return EffectScheduleContract(standard_memory_sequential=True, standard_allocations_successful=True, index_bits=64)


@pytest.fixture
def selected_toolchain(tmp_path):
    selection = os.environ.get("MERLIN_LEAN_TOOLCHAIN")
    if not selection:
        pytest.skip("explicit independently selected Lean toolchain unavailable")
    return LeanToolchain.from_manifest(Path(selection), work_root=tmp_path)


def test_v3_adds_domains_without_reinterpreting_historical_coverage():
    assert coverage_policy_record(version="1")["domains"] == [
        "scalar_integer_v1",
        "ordered_loop_segmentation_v1",
        "buffer_lifetime_v1",
        "dependency_order_v1",
    ]
    assert coverage_policy_record(version="2")["domains"] == coverage_policy_record(version="3")["domains"][:-2]
    assert coverage_policy_record(version="3")["domains"][-2:] == ["resource_geometry_v1", "state_effect_schedule_v1"]


def test_assertions_and_saved_summaries_cannot_become_live_evidence():
    forged = CHECK.verify_authored_transform(
        BEFORE, AFTER, entry="f", registry={}, contract=_contract(), toolchain=None
    )
    assert forged.status == "unavailable"
    forged = replace(forged, status="verified", toolchain_sha256="a" * 64)
    assert not CHECK.accepted_authored_receipt(
        forged, BEFORE, AFTER, entry="f", registry={}, contract=_contract(), toolchain=None
    )
    saved = LeanTransformReceipt(**json.loads(json.dumps(forged.to_dict())))
    assert not CHECK.accepted_authored_receipt(
        saved, BEFORE, AFTER, entry="f", registry={}, contract=_contract(), toolchain=None
    )


def test_changed_arithmetic_is_never_accepted_as_schedule_correspondence():
    result = CHECK.verify_effect_schedule(
        BEFORE, AFTER.replace("arith.addi", "arith.subi"), entry="f", contract=_contract(), toolchain=None
    )
    assert result.status == "unsupported" and not result.verified


def test_actual_resource_fault_refuses_before_any_formal_tool_invocation():
    from test_lean_resource_geometry import _case, _change_address, _emit

    module, model = _case()
    emitted = _emit(module, model)
    _change_address(emitted, 2, 10)
    result = CHECK.verify_resource_assignment(ir_text(module), ir_text(emitted), toolchain=None, **model)
    assert result.status == "unsupported" and not result.verified
    assert not CHECK.accepted_resource_receipt(result, ir_text(module), ir_text(emitted), toolchain=None, **model)


def test_real_authored_scalar_rewrite_is_bound_to_actual_ir_and_semantics(selected_toolchain):
    parameters = dict(entry="f", registry={}, contract=_contract(), toolchain=selected_toolchain)
    receipt = CHECK.verify_authored_transform(BEFORE, AFTER, **parameters)
    assert receipt.verified, receipt.reason
    assert CHECK.accepted_authored_receipt(receipt, BEFORE, AFTER, **parameters)
    assert not CHECK.accepted_authored_receipt(replace(receipt), BEFORE, AFTER, **parameters)
    assert not CHECK.accepted_authored_receipt(receipt, BEFORE, AFTER + "\n", **parameters)
    assert not CHECK.accepted_authored_receipt(receipt, BEFORE, AFTER, **{**parameters, "entry": "other"})
    assert not CHECK.accepted_authored_receipt(receipt, BEFORE, AFTER, **{**parameters, "registry": {"other": {}}})
    wrong = CHECK.verify_authored_transform(BEFORE, AFTER.replace("arith.addi", "arith.subi"), **parameters)
    assert not wrong.verified and wrong.status != "unsupported"


def test_real_host_tactics_check_strength_reduction_under_one_budget(selected_toolchain):
    source = """builtin.module { func.func @f(%a:i8)->i8 {
      %two = arith.constant 2 : i8
      %r = arith.muli %a,%two : i8
      func.return %r : i8
    }}"""
    emitted = """builtin.module { func.func @f(%a:i8)->i8 {
      %r = arith.addi %a,%a : i8
      func.return %r : i8
    }}"""
    receipt = CHECK.verify_authored_transform(
        source, emitted, entry="f", registry={}, contract=_contract(), toolchain=selected_toolchain
    )
    assert receipt.verified, receipt.reason


def test_real_state_schedule_handles_static_fresh_memory_and_rejects_aliases(selected_toolchain):
    source = """builtin.module { func.func @f(%a:i8,%b:i8) {
      %index = arith.constant 0 : index
      %left = memref.alloc() : memref<1xi8>
      %right = memref.alloc() : memref<1xi8>
      memref.store %a,%left[%index] : memref<1xi8>
      memref.store %b,%right[%index] : memref<1xi8>
      memref.dealloc %left : memref<1xi8>
      memref.dealloc %right : memref<1xi8>
      func.return
    }}"""
    module = parse_ir(source)
    block = blocks_in(module)[1]
    ops = tuple(block.ops)
    for op in ops:
        block.detach_op(op)
    block.add_ops(ops[index] for index in (0, 1, 2, 4, 3, 5, 6, 7))
    emitted = ir_text(module)
    parameters = dict(entry="f", contract=_contract(), toolchain=selected_toolchain)
    receipt = CHECK.verify_effect_schedule(source, emitted, **parameters)
    assert receipt.verified, receipt.reason
    assert CHECK.accepted_effect_receipt(receipt, source, emitted, **parameters)
    assert not CHECK.accepted_effect_receipt(
        receipt,
        source,
        emitted,
        **{**parameters, "contract": replace(_contract(), standard_allocations_successful=False)},
    )
    source_alias = source.replace("%b,%right[%index]", "%b,%left[%index]")
    emitted_alias = emitted.replace("%b, %right", "%b, %left")
    # Reconstruct a real aliasing permutation rather than relying on printed names.
    alias_module = parse_ir(source_alias)
    alias_block = blocks_in(alias_module)[1]
    alias_ops = tuple(alias_block.ops)
    for op in alias_ops:
        alias_block.detach_op(op)
    alias_block.add_ops(alias_ops[index] for index in (0, 1, 2, 4, 3, 5, 6, 7))
    emitted_alias = ir_text(alias_module)
    wrong = CHECK.verify_effect_schedule(source_alias, emitted_alias, **parameters)
    assert wrong.status == "unsupported" and not wrong.verified


def test_real_resource_receipt_rechecks_current_original_constraints(selected_toolchain):
    from test_lean_resource_geometry import _case, _emit

    module, model = _case()
    emitted = _emit(module, model)
    before, after = ir_text(module), ir_text(emitted)
    receipt = CHECK.verify_resource_assignment(before, after, toolchain=selected_toolchain, **model)
    assert receipt.verified, receipt.reason
    assert CHECK.accepted_resource_receipt(receipt, before, after, toolchain=selected_toolchain, **model)
    assert not CHECK.accepted_resource_receipt(replace(receipt), before, after, toolchain=selected_toolchain, **model)
    changed = {**model, "banks": tuple(replace(bank, capacity=14) for bank in model["banks"])}
    assert not CHECK.accepted_resource_receipt(receipt, before, after, toolchain=selected_toolchain, **changed)
