"""Actual V2 pipeline transformations require independent live Lean completion."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.transforms import PipelineOptions, apply_common_pipeline
from merlin.transforms.certificates import TransformationChange, parse_ir
from merlin.transforms.validation import validate_change
from merlin.verify.lean_checker import accepted_pipeline_receipt, verify_pipeline_change
from merlin.verify.lean_contract import LeanToolchain

SCALAR = """builtin.module {
  func.func @kernel(%x: i8, %y: i8) -> (i8, i8) {
    %a = arith.muli %x, %y : i8
    %b = arith.muli %x, %y : i8
    %dead = arith.addi %x, %y : i8
    func.return %a, %b : i8, i8
  }
}"""
LOOP = """builtin.module {
  func.func @kernel(%x: i8) -> i8 {
    %l = arith.constant 3 : index
    %u = arith.constant 31 : index
    %s = arith.constant 4 : index
    %result = scf.for %i = %l to %u step %s iter_args(%acc = %x) -> (i8) {
      %three = arith.constant 3 : i8
      %invariant = arith.muli %x, %three : i8
      %r = arith.addi %acc, %invariant : i8
      scf.yield %r : i8
    }
    func.return %result : i8
  }
}"""


@pytest.mark.parametrize(
    "source,options,kind",
    [
        (SCALAR, {"eliminate_redundancy": True}, "scalar_redundancy_elimination"),
        (LOOP, {"hoist_invariants": True}, "scalar_loop_invariant_motion"),
        (LOOP, {"unroll_factors": (3,)}, "ordered_loop_unrolling"),
    ],
)
def test_actual_pipeline_changes_have_closed_transport_and_reject_wrong_ir(source, options, kind):
    parameters = PipelineOptions(64, normalize_shapes=False, simplify_integers=False, reuse_buffers=False, **options)
    result = apply_common_pipeline(parse_ir(source), parameters, plan_only=True)
    assert len(result.changes) == 1 and result.changes[0].kind == kind
    change = result.changes[0]
    decoded = TransformationChange.from_dict(json.loads(json.dumps(change.to_dict())))
    validate_change(decoded)
    assert decoded == change and decoded.certificate.schema == "merlin.shared_transform.v2"
    wrong = change.after_ir.replace('"func.return"', '"fresh.invalid_return"')
    assert wrong != change.after_ir
    with pytest.raises(ValueError):
        validate_change(replace(change, after_ir=wrong))
    with pytest.raises(ValueError, match="schema"):
        validate_change(replace(change, certificate=replace(change.certificate, schema="merlin.shared_transform.v1")))


@pytest.mark.parametrize(
    "source,options,domain",
    [
        (SCALAR, {"eliminate_redundancy": True}, "total_scalar_rewrite_v1"),
        (LOOP, {"hoist_invariants": True}, "total_scalar_rewrite_v1"),
        (LOOP.replace("31 : index", "1000000000003 : index"), {"unroll_factors": (7,)}, "ordered_loop_unrolling_v1"),
    ],
)
def test_real_v2_proofs_bind_actual_pipeline_and_large_shapes(source, options, domain, tmp_path):
    manifest = os.environ.get("MERLIN_LEAN_TOOLCHAIN_MANIFEST")
    if not manifest:
        pytest.skip("explicit independently selected Lean toolchain not installed")
    toolchain = LeanToolchain.from_manifest(Path(manifest), work_root=tmp_path)
    receipts = []

    def verify(change):
        receipt = verify_pipeline_change(change, toolchain=toolchain)
        assert accepted_pipeline_receipt(receipt, change, toolchain=toolchain), receipt.reason
        assert receipt.domain == domain
        receipts.append(receipt)
        return receipt

    parameters = PipelineOptions(64, normalize_shapes=False, simplify_integers=False, reuse_buffers=False, **options)
    result = apply_common_pipeline(parse_ir(source), parameters, verifier=verify)
    assert receipts and not result.plan_only
    assert len(result.changes[0].after_ir) < 15000
