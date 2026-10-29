"""Compiler capture and explicit draft selection; no simulated proof acceptance."""

from types import SimpleNamespace

import pytest
from lean_bf16_f32acc_cases import matmul

from merlin.llvmlower.passes_xdsl import lower_bf16_matmul_f32acc
from merlin.transforms.certificates import parse_ir
from merlin.verify import compiler_transforms as transforms
from merlin.verify import pass_validation


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
def test_actual_pass_and_entry_capture_exactly_one_draft_job(tmp_path, fmt, monkeypatch):
    # Capture-only control; source recognition and kernel acceptance have their
    # own real-tool tests. This stub grants neither kind of proof authority.
    monkeypatch.setattr(pass_validation, "source_admission", lambda *args, **kwargs: "capture-only-test")
    jobs = []
    validation = SimpleNamespace(
        passes=frozenset({transforms.MACC}),
        index_bits=64,
        tools=SimpleNamespace(sdk_bin="/test-unused-sdk", compiler_python="/test-unused-python"),
        gate=SimpleNamespace(_worker_path=[], _aliases={}, submit=lambda name, **job: jobs.append((name, job))),
    )
    token = pass_validation._ACTIVE.set(validation)
    try:
        final, summary = transforms.apply(
            matmul(fmt), tmp_path / "captures", options=transforms.Options(macc_f32acc=True)
        )
    finally:
        pass_validation._ACTIVE.reset(token)
    assert len(jobs) == 1 and jobs[0][0] == "compiler_macc_contract"
    request = jobs[0][1]
    assert request["before"] == (tmp_path / "captures/macc-before.mlir").read_text()
    assert request["after"] == final == (tmp_path / "captures/macc-after.mlir").read_text()
    assert summary["changed"] == summary["selected_checks"] == [transforms.MACC]


def test_actual_pass_identity_submits_no_draft_job(tmp_path, monkeypatch):
    monkeypatch.setattr(pass_validation, "source_admission", lambda *args, **kwargs: "capture-only-test")
    jobs = []
    validation = SimpleNamespace(
        passes=frozenset({transforms.MACC}),
        gate=SimpleNamespace(submit=lambda *args, **kwargs: jobs.append((args, kwargs))),
    )
    token = pass_validation._ACTIVE.set(validation)
    try:
        assert lower_bf16_matmul_f32acc(parse_ir("builtin.module {}")) == 0
    finally:
        pass_validation._ACTIVE.reset(token)
    assert jobs == []


def corrupt_producer(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    count = lower_bf16_matmul_f32acc(module)
    operation = next(op for op in module.walk() if isinstance(op, arith.AddfOp))
    Rewriter.replace_op(operation, arith.SubfOp(operation.lhs, operation.rhs))
    return count


def test_capture_includes_mutation_after_decorated_producer(tmp_path, monkeypatch):
    from merlin.llvmlower import passes_xdsl

    jobs = []
    monkeypatch.setattr(pass_validation, "source_admission", lambda *args, **kwargs: "capture-only-test")
    validation = SimpleNamespace(
        passes=frozenset({transforms.MACC}),
        index_bits=64,
        tools=SimpleNamespace(sdk_bin="/test-unused-sdk", compiler_python="/test-unused-python"),
        gate=SimpleNamespace(_worker_path=[], _aliases={}, submit=lambda name, **job: jobs.append((name, job))),
    )
    monkeypatch.setattr(passes_xdsl, "lower_bf16_matmul_f32acc", corrupt_producer)
    token = pass_validation._ACTIVE.set(validation)
    try:
        final, _ = transforms.apply(matmul("f16"), tmp_path / "captures", options=transforms.Options(macc_f32acc=True))
    finally:
        pass_validation._ACTIVE.reset(token)
    assert len(jobs) == 1
    assert jobs[0][1]["after"] == final
    assert '"arith.subf"' in final


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
def test_macc_draft_compiler_gate_matches_actual_ungated_llvm(tmp_path, pass_gate_tools, fmt):
    from merlin.llvmlower import lower
    from merlin.verify.speculative_gate import SpeculativeGate

    options = transforms.Options(macc_f32acc=True)
    baseline = lower.lower_model(matmul(fmt), tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({transforms.MACC}))
    checked = lower.lower_model(
        matmul(fmt), tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["claim_type"] == "draft_policy_conformance"
    assert detail["policy_reviewed"] is False and detail["numerical_policy_verdict"] == "abstracted"
    assert detail["original_source_equivalence"] is False


def test_corrupt_actual_macc_producer_stops_compiler(tmp_path, pass_gate_tools, monkeypatch):
    from merlin.llvmlower import lower, passes_xdsl
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    monkeypatch.setattr(passes_xdsl, "lower_bf16_matmul_f32acc", corrupt_producer)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({transforms.MACC}))
    with pytest.raises(GateFailure):
        lower.lower_model(
            matmul("f16"),
            tmp_path / "checked",
            targets=(),
            formal_transforms=transforms.Options(macc_f32acc=True),
            pass_validation=validation,
        )
    assert any(row["verdict"] == "not_established" for row in worker.verdicts)
    assert not tuple((tmp_path / "checked").glob("*.ll"))


def test_all_excludes_draft_and_explicit_selection_preserves_it(tmp_path, pass_gate_tools, monkeypatch):
    monkeypatch.setenv(pass_validation.ENV_SDK_BIN, str(pass_gate_tools.sdk_bin))
    monkeypatch.setenv(pass_validation.ENV_COMPILER_PYTHON, str(pass_gate_tools.compiler_python))
    monkeypatch.setenv(pass_validation.ENV_PASSES, "all")
    default = pass_validation.from_environment(tmp_path / "default")
    assert transforms.MACC not in default.passes
    assert {"cse", "ordered_loop_unrolling", "ordered_loop_segmentation", "buffer_reuse"} <= default.passes
    monkeypatch.setenv(pass_validation.ENV_PASSES, transforms.MACC)
    explicit = pass_validation.from_environment(tmp_path / "explicit")
    assert explicit.passes == frozenset({transforms.MACC})


def test_cli_selects_macc_transform_separately_from_draft_validation(monkeypatch):
    from argparse import Namespace

    from merlin.compile.command import apply_pass_validation

    monkeypatch.delenv(transforms.ENV_MACC, raising=False)
    monkeypatch.delenv(pass_validation.ENV_PASSES, raising=False)
    apply_pass_validation(Namespace(validate_passes=transforms.MACC, lean_sdk_bin=None))
    assert transforms.from_environment().macc_f32acc is False
    monkeypatch.setenv(transforms.ENV_MACC, "0")
    apply_pass_validation(Namespace(validate_passes=None, lean_sdk_bin=None, macc_f32acc=True))
    assert transforms.from_environment().macc_f32acc is True


@pytest.mark.parametrize("name", list(pass_validation.CHECKS))
@pytest.mark.parametrize("override", ["alias", "worker_path"])
def test_formal_selection_cannot_be_rerouted(tmp_path, pass_gate_tools, name, override):
    from merlin.verify.speculative_gate import SpeculativeGate

    options = (
        {"aliases": {pass_validation.CHECKS[name]: "untrusted:accept"}}
        if override == "alias"
        else {"worker_path": [tmp_path]}
    )
    worker = SpeculativeGate(tmp_path / "gate", **options)
    with pytest.raises(ValueError, match="canonical handler"):
        pass_validation.PassValidation(worker, pass_gate_tools, frozenset({name}))
    assert worker.verdicts == [] and worker._threads == []
