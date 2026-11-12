from __future__ import annotations

from verification_witnesses.transform_receipts import (
    _TRANSLATOR,
    HAS_XDSL,
    HAS_Z3,
    Path,
    _backend_pair,
    _source_pair,
    _tool,
    copy,
    explicit_test_target,
    hashlib,
    json,
    os,
    pytest,
    pytestmark,
    replace,
    yaml,
)


@pytest.mark.parametrize(
    ("boundary", "pair"), [("linalg_to_interface", _source_pair), ("interface_to_command_buffer", _backend_pair)]
)
@pytest.mark.usefixtures("explicit_test_target")
@pytestmark
def test_real_transform_receipt_proves_and_replays_exact_artifacts(boundary, pair):
    from merlin.verification.receipts import TransformReceipt, qualify_receipt, verify_transformation

    (source, target) = pair()
    (translator, digest) = _tool()
    receipt = verify_transformation(boundary, source, target, translator=translator, expected_translator_sha256=digest)
    assert receipt.verified, (receipt.status, receipt.reason)
    assert receipt.smt2_sha256
    assert receipt.source["sha256"] != receipt.target["sha256"]
    assert receipt.toolchain["mlir_translate_sha256"] == digest
    assert receipt.toolchain["z3_version"]
    assert receipt.toolchain["verifier_sha256"]
    assert receipt.typed_signatures["source"]["function_count"] == 1
    assert receipt.typed_signatures["source"]["inputs"]
    assert qualify_receipt(receipt, source, target, translator=translator)
    loaded = TransformReceipt.from_dict(json.loads(json.dumps(receipt.to_dict())))
    assert qualify_receipt(loaded, source, target, translator=translator)
    assert not qualify_receipt(replace(receipt, method="syntax_verification"), source, target, translator=translator)


@pytest.mark.usefixtures("explicit_test_target")
@pytestmark
def test_source_mutation_invalidates_an_old_receipt_and_yields_counterexample():
    from xdsl.dialects.builtin import IntegerAttr, i32

    from merlin.verification.receipts import qualify_receipt, verify_transformation

    (source, target) = _source_pair()
    (translator, digest) = _tool()
    old = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert old.verified
    zero_point = next((op for op in source.walk() if op.name == "arith.constant"))
    zero_point.properties["value"] = IntegerAttr(7, i32)
    assert not qualify_receipt(old, source, target, translator=translator)
    current = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert current.status == "refuted", (current.status, current.reason)
    assert current.counterexample and current.counterexample["inputs"]
    assert current.source["sha256"] != old.source["sha256"]


@pytest.mark.usefixtures("explicit_test_target")
@pytestmark
def test_bare_tensor_empty_never_produces_a_qualified_receipt():
    from merlin.verification.receipts import qualify_receipt, verify_transformation
    from merlin.xdsl_dialects.lowering.pipeline import lower_repeated_rhs_matmul

    legacy = lower_repeated_rhs_matmul(reuse=2, m=2, k=2, n=2, target="toy_npu")
    (translator, digest) = _tool()
    receipt = verify_transformation(
        "linalg_to_interface",
        legacy.input_module,
        legacy.interface_module,
        translator=translator,
        expected_translator_sha256=digest,
    )
    assert receipt.status == "unsupported"
    assert "tensor.empty with unspecified contents" in (receipt.reason or "")
    assert not receipt.verified
    assert not qualify_receipt(receipt, legacy.input_module, legacy.interface_module, translator=translator)


@pytest.mark.usefixtures("explicit_test_target")
@pytestmark
def test_backend_mutation_refutes_and_changed_digest_cannot_qualify():
    from merlin.verification.receipts import qualify_receipt, verify_transformation

    (source, target) = _backend_pair()
    (translator, digest) = _tool()
    old = verify_transformation(
        "interface_to_command_buffer", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert old.verified
    changed = copy.deepcopy(target)
    from merlin.verification.faults import CB_CORPUS

    next((f for f in CB_CORPUS if f.name == "cb_swapped_matmul_operands")).mutate(changed)
    current = verify_transformation(
        "interface_to_command_buffer", source, changed, translator=translator, expected_translator_sha256=digest
    )
    assert current.status == "refuted", (current.status, current.reason)
    assert current.counterexample and current.counterexample["inputs"]
    assert not qualify_receipt(old, source, changed, translator=translator)


@pytest.mark.usefixtures("explicit_test_target")
@pytestmark
def test_wrong_tool_pin_and_solver_unknown_never_qualify(monkeypatch):
    from merlin.verification import refine
    from merlin.verification.receipts import qualify_receipt, verify_transformation
    from merlin.verification.smt_export import Verdict

    (source, target) = _source_pair()
    (translator, _) = _tool()
    mismatch = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256="0" * 64
    )
    assert mismatch.status == "unavailable"
    assert not mismatch.verified
    monkeypatch.setattr(refine, "validate_pass", lambda *args, **kwargs: Verdict("unknown", reason="timeout"))
    unknown = verify_transformation("linalg_to_interface", source, target, translator=translator)
    assert unknown.status == "unknown" and unknown.reason == "timeout"
    assert not qualify_receipt(unknown, source, target, translator=translator)


@pytest.mark.usefixtures("explicit_test_target")
@pytestmark
def test_float_source_is_explicitly_unsupported():
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f32
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from merlin.verification.receipts import verify_transformation

    (_, target) = _source_pair()
    tensor_type = TensorType(f32, [2, 2])
    block = Block(arg_types=[tensor_type])
    block.add_op(ReturnOp(block.args[0]))
    source = ModuleOp([FuncOp("float_source", FunctionType.from_lists([tensor_type], [tensor_type]), Region([block]))])
    (translator, digest) = _tool()
    receipt = verify_transformation(
        "linalg_to_interface", source, target, translator=translator, expected_translator_sha256=digest
    )
    assert receipt.status == "unsupported", (receipt.status, receipt.reason)
    assert "f32" in (receipt.reason or "")
