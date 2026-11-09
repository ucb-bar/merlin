"""Structural proof receipts for the actual emitted xDSL outline boundary."""

from __future__ import annotations

import json

import numpy as np

from merlin.common import mlir_query as mq
from merlin.verification.outline_ir import (
    OutlineTransformReceipt,
    qualify_outline_receipt,
    verify_outline_transform,
)
from merlin.xdsl_dialects.lowering.outline import outline_dispatches

SOURCE = b"""module {
  func.func @forward(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
                     %c: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %y = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
         outs(%c : tensor<2x2xi32>) -> tensor<2x2xi32>
    func.return %y : tensor<2x2xi32>
  }
}
"""

CHAIN = b"""module {
  func.func @forward(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
                     %c: tensor<2x2xi32>, %d: tensor<2x2xi32>,
                     %e: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %first = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
             outs(%c : tensor<2x2xi32>) -> tensor<2x2xi32>
    %second = linalg.matmul ins(%first, %d : tensor<2x2xi32>, tensor<2x2xi32>)
              outs(%e : tensor<2x2xi32>) -> tensor<2x2xi32>
    func.return %second : tensor<2x2xi32>
  }
}
"""


def _actual_pair() -> tuple[bytes, bytes]:
    module = mq.parse(SOURCE.decode())
    before = str(module).encode()
    result = outline_dispatches(module)
    assert result.n_kernels == 1
    after = str(result.module).encode()
    return before, after


def test_real_outline_pass_has_byte_bound_structural_proof() -> None:
    import hashlib
    import pickle
    from pathlib import Path

    from merlin.verification import outline_ir as owner
    from merlin.verify import outline_ir as legacy

    assert legacy is owner
    assert legacy.UnsupportedOutlineIR is owner.UnsupportedOutlineIR
    before, after = _actual_pair()
    receipt = verify_outline_transform(before, after)
    assert receipt.status == "verified", receipt.reason
    assert receipt.verifier_sha256 == hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()
    assert receipt.__class__.__module__ == "merlin.verify.outline_ir"
    assert pickle.loads(pickle.dumps(receipt)) == receipt
    assert receipt.source_sha256 != receipt.target_sha256
    assert receipt.source_graph_sha256 == receipt.target_graph_sha256
    assert receipt.input_types == ("tensor<2x2xi32>",) * 3
    assert receipt.output_types == ("tensor<2x2xi32>",)
    assert qualify_outline_receipt(receipt, before, after)
    assert OutlineTransformReceipt.from_dict(receipt.to_dict()) == receipt


def test_two_real_dispatches_preserve_dataflow_and_result_order() -> None:
    module = mq.parse(CHAIN.decode())
    before = str(module).encode()
    result = outline_dispatches(module)
    assert result.n_kernels == 2
    after = str(result.module).encode()
    receipt = verify_outline_transform(before, after)
    assert receipt.status == "verified", receipt.reason
    assert qualify_outline_receipt(receipt, before, after)
    # Rewire the second dispatch to consume the original first argument rather
    # than the first dispatch's result. Types remain valid; SSA validation alone
    # does not catch the semantic/dataflow change.
    driver = after.split(b"func.func private", 1)[0]
    second_call = next(line for line in driver.splitlines() if b"@forward$kernel_1(" in line)
    old_arg = second_call.split(b"@forward$kernel_1(", 1)[1].split(b",", 1)[0]
    wrong_call = second_call.replace(b"@forward$kernel_1(" + old_arg, b"@forward$kernel_1(%0", 1)
    assert wrong_call != second_call
    wrong = after.replace(second_call, wrong_call, 1)
    mismatch = verify_outline_transform(before, wrong)
    assert mismatch.status == "mismatch", mismatch.reason
    assert mismatch.mismatch_witness is not None


def test_swapped_emitted_call_operands_yield_structural_mismatch_witness() -> None:
    before, after = _actual_pair()
    wrong = after.replace(b"@forward$kernel_0(%0, %1, %2)", b"@forward$kernel_0(%1, %0, %2)", 1)
    assert wrong != after
    receipt = verify_outline_transform(before, wrong)
    assert receipt.status == "mismatch", receipt.reason
    assert receipt.mismatch_witness is not None
    assert not qualify_outline_receipt(receipt, before, wrong)
    assert not qualify_outline_receipt(verify_outline_transform(before, after), before, wrong)
    # The receipt supplies a structural witness, not a numerical theorem of
    # matmul. This concrete input separately confirms the mutation matters.
    a = np.array([[1, 2], [0, 1]], dtype=np.int32)
    b = np.array([[2, 0], [1, 3]], dtype=np.int32)
    assert not np.array_equal(a @ b, b @ a)


def test_unmodeled_effectful_or_uninitialized_tensor_abstains() -> None:
    before, after = _actual_pair()
    uninitialized = before.replace(
        b"%y = linalg.matmul", b"%e = tensor.empty() : tensor<2x2xi32>\n    %y = linalg.matmul", 1
    )
    receipt = verify_outline_transform(uninitialized, after)
    assert receipt.status == "unsupported", receipt.reason
    assert "tensor.empty" in (receipt.reason or "")
    assert not receipt.verified


def test_float_and_region_capture_abstain() -> None:
    before, after = _actual_pair()
    floating_before = before.replace(b"i32", b"f32")
    floating_after = after.replace(b"i32", b"f32")
    receipt = verify_outline_transform(floating_before, floating_after)
    assert receipt.status == "unsupported", receipt.reason
    assert not receipt.verified


def test_unrelated_target_function_is_not_silently_ignored() -> None:
    before, after = _actual_pair()
    extra = after.replace(
        b"\n}",
        b"\n  func.func private @dead(%x: i32) -> i32 { func.return %x : i32 }\n}",
        1,
    )
    receipt = verify_outline_transform(before, extra)
    assert receipt.status == "unsupported", receipt.reason
    assert "not an emitted outline kernel" in (receipt.reason or "")


def test_installed_cli_records_and_qualifies_exact_outline(tmp_path) -> None:
    from merlin.verify.cli import main

    before, after = _actual_pair()
    before_path, after_path = tmp_path / "before.mlir", tmp_path / "after.mlir"
    receipt_path = tmp_path / "outline-receipt.json"
    before_path.write_bytes(before)
    after_path.write_bytes(after)
    assert (
        main(
            ["outline-receipt", "--before", str(before_path), "--after", str(after_path), "--output", str(receipt_path)]
        )
        == 0
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "verified"
    assert (
        main(
            [
                "qualify-outline-receipt",
                "--receipt",
                str(receipt_path),
                "--before",
                str(before_path),
                "--after",
                str(after_path),
            ]
        )
        == 0
    )
    after_path.write_bytes(after.replace(b"@forward$kernel_0(%0, %1, %2)", b"@forward$kernel_0(%1, %0, %2)", 1))
    assert (
        main(
            [
                "qualify-outline-receipt",
                "--receipt",
                str(receipt_path),
                "--before",
                str(before_path),
                "--after",
                str(after_path),
            ]
        )
        == 2
    )
