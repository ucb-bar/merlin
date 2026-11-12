from __future__ import annotations

from verification_witnesses.scalar_ir import (
    HAS_Z3,
    SOURCE,
    ScalarTransformReceipt,
    chunk_forward,
    mq,
    pytest,
    qualify_scalar_receipt,
    verify_scalar_transform,
)


@pytest.mark.parametrize("chunk_ops", [2])
@pytest.mark.skipif(not HAS_Z3, reason="requires the explicitly installed z3 verification prerequisite")
def test_actual_chunk_forward_ir_has_byte_bound_proof(chunk_ops: int) -> None:
    import hashlib
    import pickle
    from pathlib import Path

    from merlin.verification import scalar_ir as legacy
    from merlin.verification import scalar_ir as owner

    assert legacy is owner
    assert legacy.UnsupportedScalarIR is owner.UnsupportedScalarIR
    module = mq.parse(SOURCE)
    before = str(module).encode("utf-8")
    made = chunk_forward(module, chunk_ops=chunk_ops)
    assert made >= 2
    after = str(module).encode("utf-8")
    assert b"merlin_forward_chunk_" in after
    receipt = verify_scalar_transform(before, after)
    assert receipt.status == "verified", receipt.reason
    assert receipt.verifier_sha256 == hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()
    assert receipt.__class__.__module__ == "merlin.verification.scalar_ir"
    assert pickle.loads(pickle.dumps(receipt)) == receipt
    assert receipt.source_sha256 != receipt.target_sha256
    assert receipt.input_types == ("i32",)
    assert receipt.output_types == ("i32",)
    assert receipt.query_sha256
    assert qualify_scalar_receipt(receipt, before, after)
    assert not qualify_scalar_receipt(receipt, before, after.replace(b"arith.muli", b"arith.addi"))


@pytest.mark.skipif(not HAS_Z3, reason="requires the explicitly installed z3 verification prerequisite")
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


@pytest.mark.skipif(not HAS_Z3, reason="requires the explicitly installed z3 verification prerequisite")
def test_external_call_with_unknown_effects_abstains() -> None:
    raw = b"module {\n      func.func private @external(%x: i32) -> i32\n      func.func @forward(%x: i32) -> i32 {\n        %y = func.call @external(%x) : (i32) -> i32\n        func.return %y : i32\n      }\n    }"
    receipt = verify_scalar_transform(raw, raw)
    assert receipt.status == "unsupported", receipt.reason
    assert "no single-block definition" in (receipt.reason or "")
    assert not receipt.verified
