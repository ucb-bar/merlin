"""Translation validation of exact emitted IR from the real chunk-forward pass."""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.perf.whole_model_chunks import chunk_forward
from merlin.verify.scalar_ir import ScalarTransformReceipt, qualify_scalar_receipt, verify_scalar_transform

pytest.importorskip("z3")


SOURCE = """module {
  func.func @forward(%x: i32) -> i32 {
    %a = arith.addi %x, %x : i32
    %b = arith.muli %a, %x : i32
    %c = arith.subi %b, %a : i32
    %d = arith.addi %c, %x : i32
    func.return %d : i32
  }
}
"""


@pytest.mark.parametrize("chunk_ops", [1, 2, 3])
def test_actual_chunk_forward_ir_has_byte_bound_proof(chunk_ops: int) -> None:
    module = mq.parse(SOURCE)
    before = str(module).encode("utf-8")
    made = chunk_forward(module, chunk_ops=chunk_ops)
    assert made >= 2
    after = str(module).encode("utf-8")
    assert b"merlin_forward_chunk_" in after

    receipt = verify_scalar_transform(before, after)
    assert receipt.status == "verified", receipt.reason
    assert receipt.source_sha256 != receipt.target_sha256
    assert receipt.input_types == ("i32",)
    assert receipt.output_types == ("i32",)
    assert receipt.query_sha256
    assert qualify_scalar_receipt(receipt, before, after)
    assert not qualify_scalar_receipt(receipt, before, after.replace(b"arith.muli", b"arith.addi"))


def test_wrong_helper_arithmetic_is_refuted_with_concrete_input() -> None:
    module = mq.parse(SOURCE)
    before = str(module).encode("utf-8")
    chunk_forward(module, chunk_ops=1)
    after = str(module).encode("utf-8")
    assert b"arith.muli" in after
    wrong = after.replace(b"arith.muli", b"arith.addi", 1)
    receipt = verify_scalar_transform(before, wrong)
    assert receipt.status == "refuted", receipt.reason
    assert receipt.counterexample is not None
    assert not qualify_scalar_receipt(receipt, before, wrong)


def test_unmodeled_operation_abstains_instead_of_becoming_a_proof() -> None:
    module = mq.parse(SOURCE)
    before = str(module).encode("utf-8")
    chunk_forward(module, chunk_ops=1)
    after = str(module).encode("utf-8")
    unknown = after.replace(b"arith.muli", b"arith.divsi", 1)
    receipt = verify_scalar_transform(before, unknown)
    assert receipt.status == "unsupported", receipt.reason
    assert "arith.divsi" in (receipt.reason or "")
    assert not qualify_scalar_receipt(receipt, before, unknown)


def test_unmodeled_float_source_abstains() -> None:
    raw = b"module { func.func @forward(%x: f32) -> f32 { func.return %x : f32 } }"
    receipt = verify_scalar_transform(raw, raw)
    assert receipt.status == "unsupported", receipt.reason
    assert not receipt.verified


def test_external_call_with_unknown_effects_abstains() -> None:
    raw = b"""module {
      func.func private @external(%x: i32) -> i32
      func.func @forward(%x: i32) -> i32 {
        %y = func.call @external(%x) : (i32) -> i32
        func.return %y : i32
      }
    }"""
    receipt = verify_scalar_transform(raw, raw)
    assert receipt.status == "unsupported", receipt.reason
    assert "no single-block definition" in (receipt.reason or "")
    assert not receipt.verified


SATURATE = b"""module {func.func @forward(%x:i32)->i8 {
 %lo=arith.constant -128:i32
 %hi=arith.constant 127:i32
 %l=arith.maxsi %x,%lo:i32
 %h=arith.minsi %l,%hi:i32
 %r=arith.trunci %h:i32 to i8
 return %r:i8 }}"""


def test_clamp_minmax_and_signed_compare_select_have_identical_bitvector_semantics():
    selected = SATURATE.replace(b"%l=arith.maxsi %x,%lo:i32", b"""
    %p=arith.cmpi slt,%x,%lo:i32
    %l=arith.select %p,%lo,%x:i32""").replace(b"%h=arith.minsi %l,%hi:i32", b"""
    %q=arith.cmpi sgt,%l,%hi:i32
    %h=arith.select %q,%hi,%l:i32""")
    assert verify_scalar_transform(SATURATE, selected).verified


@pytest.mark.parametrize("changed", [
    SATURATE.replace(b"constant -128", b"constant -127"),
    SATURATE.replace(b"arith.maxsi", b"arith.maxui"),
    SATURATE.replace(b"%h:i32 to i8", b"%x:i32 to i8"),
])
def test_saturation_changes_are_refuted_over_all_accumulator_bits(changed):
    receipt = verify_scalar_transform(SATURATE, changed)
    assert receipt.status == "refuted" and receipt.counterexample is not None


@pytest.mark.parametrize("extension", ["extsi", "extui"])
def test_extension_then_truncate_preserves_original_bits(extension):
    original = b"module {func.func @forward(%x:i8)->i8 {return %x:i8}}"
    extended = f"""module {{func.func @forward(%x:i8)->i8 {{
    %w=arith.{extension} %x:i8 to i32
    %r=arith.trunci %w:i32 to i8
    return %r:i8 }}}}""".encode()
    assert verify_scalar_transform(original, extended).verified


def test_cli_records_and_replays_exact_chunk_forward_artifacts(tmp_path) -> None:
    import json

    from merlin.verify.cli import main

    module = mq.parse(SOURCE)
    before = str(module).encode("utf-8")
    chunk_forward(module, chunk_ops=2)
    after = str(module).encode("utf-8")
    before_path, after_path = tmp_path / "before.mlir", tmp_path / "after.mlir"
    receipt_path = tmp_path / "receipt.json"
    before_path.write_bytes(before)
    after_path.write_bytes(after)

    assert (
        main(
            ["scalar-receipt", "--before", str(before_path), "--after", str(after_path), "--output", str(receipt_path)]
        )
        == 0
    )
    receipt = ScalarTransformReceipt.from_dict(json.loads(receipt_path.read_text(encoding="utf-8")))
    assert qualify_scalar_receipt(receipt, before_path.read_bytes(), after_path.read_bytes())
    assert not qualify_scalar_receipt(receipt, before, after.replace(b"arith.muli", b"arith.addi"))
    assert (
        main(
            [
                "qualify-scalar-receipt",
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

    after_path.write_bytes(after.replace(b"arith.muli", b"arith.addi"))
    assert (
        main(
            [
                "qualify-scalar-receipt",
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
