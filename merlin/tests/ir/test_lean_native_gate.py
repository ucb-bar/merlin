"""Gate composition refuses stale evidence and independently checks every roster.

These unit tests mock proof/tool execution. Actual native qualification is separate.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from merlin.llvmlower import pipeline
from merlin.llvmlower import proof_stage_transport as transport
from merlin.verify.lean import native_gate, plans, stages
from merlin.verify.lean.handoff import Options, VerificationError, VerifiedPreprocessing


@pytest.fixture
def session(tmp_path, monkeypatch):
    for key in ["MERLIN_GENERALIZE_BEFORE_FUSE", "MERLIN_FUSE_POST", "MERLIN_NO_DEALLOC", "MERLIN_PASSES"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pipeline, "_sink_deallocs", lambda: False)
    compiler = sys.executable
    measured = {"compiler_python": transport.fingerprint(compiler)}
    monkeypatch.setattr(native_gate, "m2m_python", lambda: compiler)
    monkeypatch.setattr(transport, "identity", lambda _, **kwargs: measured)
    source = "module {}"
    root = tmp_path / "prefix"
    plan = plans.preprocessing()
    root.mkdir()
    (root / "expected-plan.json").write_bytes(plan.canonical_bytes())
    output = root / "preprocessed.mlir"
    output.write_text(source)
    with stages.capture(root / "stages"):
        for b in plan.boundaries:
            stages.record_boundary(b.name, b.pass_entry, source, source)
        stages.record_handoff(output)
    verified = VerifiedPreprocessing(source, {}, root / "verification.json")
    obj = native_gate.Session(Options("forward", 64, compiler), source, verified, plan)
    return obj


def _prepare(session, tmp_path):
    session.prepare(
        tmp_path / "work", session.verified.text, features=frozenset(), index_bits=64, data_layout="e-p:64:64"
    )
    runner = tmp_path / "runner.py"
    runner.write_text("# frozen ordinary runner\n")
    source_path = tmp_path / "source.mlir"
    source_path.write_text(session.verified.text)
    command = [
        sys.executable,
        str(runner),
        str(source_path),
        "out.ll",
        session.plan.pipeline(),
        *(["0"] * 13),
        "e-p:64:64",
        "0",
        "0",
        "0",
    ]
    session.freeze_command(command)
    return command


def _native_records(session):
    source = session.verified.text
    with stages.capture(session.owner / "stages") as capture:
        capture.run = session.run
        for b in session.plan.boundaries.boundaries:
            stages.record_boundary(b.name, b.pass_entry, source, source)
        held = session.owner / "current.mlir"
        held.write_text(source)
        stages.record_handoff(held)
    request = {"held_sha256": hashlib.sha256(source.encode()).hexdigest()}
    (session.owner / "request.json").write_bytes(transport.canonical(request))
    return request


@pytest.fixture
def proof_checks(monkeypatch):
    called = []

    def stage(path, **kwargs):
        called.append(("stage", path))
        return {"verdict": "verified"}

    def source(path, function, out, **kwargs):
        called.append(("source", path, function))
        out.mkdir()
        (out / "receipt.json").write_text('{"verdict":"verified"}')
        return {"verdict": "verified"}

    def compose(records, support, function, out, **kwargs):
        called.append(("chain", records, support, function, kwargs))
        result = {"verdict": "verified", "domain": "mock typed domain", "assumptions": ["mock"]}
        out.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(native_gate.check, "check_typed_stage", stage)
    monkeypatch.setattr(native_gate.check, "check_support", source)
    monkeypatch.setattr(native_gate.check, "check_definedness", source)
    monkeypatch.setattr(native_gate.check, "compose_chain", compose)
    return called


def test_combines_actual_original_source_and_every_native_boundary(tmp_path, session, proof_checks):
    _prepare(session, tmp_path)
    request = _native_records(session)
    proof = session._verify(request, session.owner)
    assert proof.request_sha256 == hashlib.sha256((session.owner / "request.json").read_bytes()).hexdigest()
    assert len([x for x in proof_checks if x[0] == "stage"]) == 5
    source_calls = [x for x in proof_checks if x[0] == "source"]
    assert len(source_calls) == 2
    assert all(x[1] == session.prefix_records[0] / "before.mlir" for x in source_calls)
    chain = [x for x in proof_checks if x[0] == "chain"][0]
    assert chain[1] == session.prefix_records + session.local_records
    assert chain[-1]["definedness"] == session.root / "definedness/receipt.json"
    report = json.loads(proof.report.read_text())
    assert report["boundary_count"] == len(session.preprocessing_plan.boundaries) + 5
    assert "native execution proof" in report["trusted_suffix"]


@pytest.mark.parametrize("fault", ["source", "output", "retained-plan", "expected-plan"])
def test_prefix_status_never_overrides_actual_bytes_or_authored_roster(tmp_path, session, fault):
    # A fabricated success dictionary/report does not substitute for replay.
    session.verified.report.write_text('{"verdict":"verified"}')
    records = stages.completed(session.verified.report.parent / "stages")
    if fault == "source":
        session.source = "module { /* different */ }"
    elif fault == "output":
        session.verified = VerifiedPreprocessing("module { /* different */ }", {}, session.verified.report)
    elif fault == "retained-plan":
        (session.verified.report.parent / "expected-plan.json").write_text("{}")
    else:
        session.preprocessing_plan = plans.Plan("forged", session.preprocessing_plan.boundaries[:-1])
    with pytest.raises((VerificationError, plans.PlanMismatch)):
        _prepare(session, tmp_path)
    assert records


@pytest.mark.parametrize("fault", ["interpreter", "width", "native-input"])
def test_consumer_mismatch_refuses_before_entry(tmp_path, session, monkeypatch, fault):
    actual, bits = session.verified.text, 64
    if fault == "interpreter":
        monkeypatch.setattr(native_gate, "m2m_python", lambda: "/different/python")
    elif fault == "width":
        bits = 32
    else:
        actual = "module { /* changed */ }"
    with pytest.raises(VerificationError):
        session.prepare(tmp_path / "work", actual, features=frozenset(), index_bits=bits, data_layout="e-p:64:64")


@pytest.mark.parametrize("fault", ["pipeline", "layout", "callback", "extra-argv"])
def test_actual_command_configuration_is_bound(tmp_path, session, fault):
    command = _prepare(session, tmp_path)
    if fault == "pipeline":
        command[4] = "canonicalize"
    elif fault == "layout":
        command[18] = "e-p:32:32"
    elif fault == "callback":
        command[6] = "1"
    else:
        command.append("1")
    with pytest.raises(VerificationError, match="actual native command"):
        session.freeze_command(command)


@pytest.mark.parametrize(
    "fault", ["held", "native-plan", "prefix", "binding", "consumer", "features", "options-function", "options-width"]
)
def test_stale_pending_evidence_refuses_before_ack(tmp_path, session, proof_checks, monkeypatch, fault):
    _prepare(session, tmp_path)
    request = _native_records(session)
    if fault == "held":
        (session.owner / "current.mlir").write_text("changed")
    elif fault == "native-plan":
        (session.root / "expected-native-plan.json").write_text("{}")
    elif fault == "prefix":
        (session.prefix_records[0] / "before.mlir").write_text("changed")
    elif fault == "binding":
        (session.root / "binding.json").write_text("{}")
    elif fault == "options-function":
        session.options = Options("different", 64, session.options.compiler_python)
    elif fault == "options-width":
        session.options = Options("forward", 32, session.options.compiler_python)
    elif fault == "consumer":
        monkeypatch.setattr(transport, "identity", lambda _, **kwargs: {"changed": True})
    else:
        from merlin.llvmlower import optional_passes

        monkeypatch.setattr(optional_passes, "selected_features", lambda _: {"changed-feature"})
        from merlin.llvmlower import impr_features

        monkeypatch.setattr(impr_features, "normalize", frozenset)
    with pytest.raises(VerificationError):
        session._verify(request, session.owner)
    assert not (session.root / "verification.json").exists()


@pytest.mark.parametrize("fault", ["boundary", "support", "definedness", "chain", "late-mutation"])
def test_incomplete_proof_or_late_mutation_never_grants_ack(tmp_path, session, proof_checks, monkeypatch, fault):
    _prepare(session, tmp_path)
    request = _native_records(session)
    if fault in ("boundary", "support", "definedness"):
        name = {"boundary": "check_typed_stage", "support": "check_support", "definedness": "check_definedness"}[fault]
        monkeypatch.setattr(native_gate.check, name, lambda *a, **k: {"verdict": "abstracted"})
    else:
        original = native_gate.check.compose_chain

        def compose(*a, **k):
            result = original(*a, **k)
            if fault == "chain":
                result["verdict"] = "abstracted"
            else:
                (session.owner / "current.mlir").write_text("changed")
            return result

        monkeypatch.setattr(native_gate.check, "compose_chain", compose)
    with pytest.raises(VerificationError):
        session._verify(request, session.owner)
    assert not (session.root / "verification.json").exists()


def test_pipeline_opt_in_passes_exact_actual_command_and_retains_proof_paths(tmp_path, session, monkeypatch):
    import subprocess

    monkeypatch.setattr(pipeline, "m2m_python", lambda: sys.executable)
    seen = []

    def consume(command, *, timeout):
        # Only IPC/proof execution is mocked; production pipeline/runner/config
        # construction and Session admission/command freezing run unchanged.
        seen.append(command)
        assert command[4] == session.plan.pipeline()
        assert "module = _pg_parse(src_path, ctx)" in open(command[1]).read()
        for name in ("verification.json", "chain.json", "status.json"):
            (session.root / name).write_text('{"verdict":"verified"}')
        open(command[3], "w").write("define void @forward() { ret void }")
        return subprocess.CompletedProcess(command, 0, "OK fusion_guard 0 pinned\n", "")

    monkeypatch.setattr(session, "run_command", consume)
    selected = {}
    result = pipeline.lower_to_llvm_ir(
        session.verified.text,
        workdir=tmp_path / "native",
        features=frozenset(),
        index_bits=64,
        data_layout="e-p:64:64",
        shared_pass_native=session,
        lowering_selection=selected,
    )
    assert len(seen) == 1
    assert result == "define void @forward() { ret void }"
    assert selected["shared_pass_native_verification"] == str(session.root / "verification.json")


@pytest.mark.parametrize("output", ["absent", "empty"])
def test_gated_pipeline_rejects_stale_or_empty_success_output(tmp_path, session, monkeypatch, output):
    import subprocess

    work = tmp_path / "native"
    work.mkdir()
    (work / "model.ll").write_text("stale successful output")
    monkeypatch.setattr(pipeline, "m2m_python", lambda: sys.executable)

    def consume(command, **kwargs):
        assert not (work / "model.ll").exists()
        if output == "empty":
            (work / "model.ll").touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(session, "run_command", consume)
    with pytest.raises(pipeline.PipelineError, match="upstream lowering failed"):
        pipeline.lower_to_llvm_ir(
            session.verified.text,
            workdir=work,
            features=frozenset(),
            index_bits=64,
            data_layout="e-p:64:64",
            shared_pass_native=session,
        )


def test_native_verification_requires_existing_shared_source_proof_before_preprocessing(tmp_path, monkeypatch):
    from merlin.llvmlower import lower

    monkeypatch.setattr(lower, "preprocess_text", lambda *a, **k: pytest.fail("preprocessing ran"))
    with pytest.raises(ValueError, match="requires shared-pass verification"):
        lower.lower_model("module {}", tmp_path, targets=(), verify_native_prefix=True)


def test_file_entry_forwards_explicit_native_flag(tmp_path, monkeypatch):
    from merlin.llvmlower import lower

    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    seen = []
    monkeypatch.setattr(lower, "lower_model", lambda *a, **k: seen.append((a, k)))
    lower.lower_model_file(source, tmp_path / "work", targets=(), verify_native_prefix=True)
    assert seen[0][0][0] == "module {}"
    assert seen[0][1]["verify_native_prefix"] is True


@pytest.mark.parametrize("fault", ["none", "function", "width", "inventory", "request", "proof-path"])
def test_bounded_worker_result_and_current_options_are_revalidated(tmp_path, session, proof_checks, monkeypatch, fault):
    import time

    _prepare(session, tmp_path)
    request = _native_records(session)
    session.proof_deadline = time.monotonic() + 30

    def worker(callback, *args, **kwargs):
        # Exercise actual worker proof assembly without spawning mocked checker
        # closures. Real spawn/deadline/group behavior has separate controls.
        result = callback(*args)
        if fault == "function":
            session.options = Options("changed-function", 64, session.options.compiler_python)
        elif fault == "width":
            session.options = Options("forward", 32, session.options.compiler_python)
        elif fault == "inventory":
            result["local_frozen"] = {}
        elif fault == "request":
            result["request_sha256"] = "0" * 64
        elif fault == "proof-path":
            result["proof_frozen"] = {"unexpected.json": "0" * 64}
        return result

    monkeypatch.setattr(native_gate.proof_worker, "run", worker)
    if fault == "none":
        proof = session._verify_bounded(request, session.owner)
        assert proof.report == session.root / "verification.json"
        assert len(session.local_records) == 5
    else:
        with pytest.raises(VerificationError):
            session._verify_bounded(request, session.owner)


def test_identity_rechecks_share_the_original_invocation_budget(tmp_path, session, monkeypatch):
    import time

    observed = []
    measured = {"compiler_python": transport.fingerprint(session.options.compiler_python)}

    def identify(compiler, *, timeout):
        observed.append(timeout)
        return measured

    monkeypatch.setattr(transport, "identity", identify)
    _prepare(session, tmp_path)
    original_deadline = session.proof_deadline
    session.proof_deadline = time.monotonic() + 0.25
    session._recheck_selection()
    assert 0 < observed[-1] <= 0.25
    assert original_deadline > session.proof_deadline
    session.proof_deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="explicit deadline"):
        session._recheck_selection()


def test_proof_child_keeps_identity_probes_in_parent(tmp_path, session, proof_checks, monkeypatch):
    _prepare(session, tmp_path)
    request = _native_records(session)
    identities = []
    measured = session.consumer

    def identify(*args, **kwargs):
        identities.append("parent")
        return measured

    def worker(callback, *args, **kwargs):
        before = len(identities)
        result = callback(*args)
        assert len(identities) == before
        return result

    monkeypatch.setattr(transport, "identity", identify)
    monkeypatch.setattr(native_gate.proof_worker, "run", worker)
    session._verify_bounded(request, session.owner)
    assert identities == ["parent", "parent"]
    from merlin.common import owned_process

    assert str(Path(owned_process.__file__)) in session.sources


def test_worker_rechecks_actual_consumer_files_without_spawning_probe(tmp_path, session, monkeypatch):
    binding = tmp_path / "mock-loaded-binding"
    binding.write_bytes(b"original-binding")
    measured = {
        "compiler_python": transport.fingerprint(session.options.compiler_python),
        "mlir_binding": transport.fingerprint(binding),
    }
    monkeypatch.setattr(transport, "identity", lambda *args, **kwargs: measured)
    _prepare(session, tmp_path)
    binding.write_bytes(b"changed-binding")

    def forbidden(*args, **kwargs):
        pytest.fail("proof worker must not spawn an identity probe")

    monkeypatch.setattr(transport, "identity", forbidden)
    with pytest.raises(VerificationError, match="consumer file changed"):
        session._recheck_selection(identity_in_parent=True)
