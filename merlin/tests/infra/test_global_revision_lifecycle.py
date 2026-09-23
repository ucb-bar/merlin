"""Exercise native submission decisions through the installed revision journal."""

import copy
import socket
import subprocess

import pytest
import test_global_perf_experiment as fixtures
from merlin_experiments.phase2 import global_inputs as GI
from merlin_experiments.phase2 import static_cache


@pytest.fixture(autouse=True)
def refuse_processes(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("revision lifecycle tests must not launch processes")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.mark.parametrize("valid_source", [False, True])
def test_work_order_keeps_source_admission_before_candidate_and_decode(tmp_path, monkeypatch, valid_source):
    experiment, candidate, calls = fixtures.setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def optimize():\n return 1\ndef schedule():\n return 1\n")
    fixtures._freeze_test_mechanism_catalog(
        experiment,
        candidate,
        tmp_path,
        [{"id": "epilogue", "selectors": [{"kind": "function", "path": "compiler.py", "symbol": "optimize"}]}],
    )
    path = tmp_path / "work-order.json"
    path.write_bytes(fixtures.P2_CONTRACTS.canonical_json(fixtures._test_mechanism_work_order(experiment, candidate)))
    digest = fixtures.P2_CONTRACTS.sha256_file(path)
    if valid_source:
        path.chmod(0o444)
    events = []

    def watch(owner, name, label):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            events.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, observed)

    watch(experiment.mechanism_program, "prepare_work_order", "source")
    watch(experiment.revision_session, "validate_candidate_scope", "candidate")
    watch(experiment.revision_session, "check_inputs", "inputs")
    watch(experiment.mechanism_program, "freeze_work_order", "decode-and-publish")
    if valid_source:
        experiment.freeze_mechanism_work_order(path, digest, candidate=candidate)
        assert events == ["source", "candidate", "inputs", "decode-and-publish", "inputs"]
    else:
        with pytest.raises(ValueError, match="immutable absolute"):
            experiment.freeze_mechanism_work_order(path, digest, candidate=candidate)
        assert events == ["source"]
    assert calls == []


def test_work_order_analysis_keeps_native_admission_and_continuation_lookup(tmp_path, monkeypatch):
    experiment, candidate, _ = fixtures.setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def optimize():\n return 1\ndef schedule():\n return 1\n")
    fixtures._freeze_test_mechanism_catalog(
        experiment,
        candidate,
        tmp_path,
        [{"id": "epilogue", "selectors": [{"kind": "function", "path": "compiler.py", "symbol": "optimize"}]}],
    )
    fixtures._freeze_test_mechanism_work_order(experiment, candidate, tmp_path)
    record = experiment.analysis.analyze(candidate, hypothesis="bind initial work-order evidence")
    events = []

    def watch(owner, name, label):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            events.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, observed)

    watch(experiment.revision_session, "check_inputs", "inputs")
    watch(experiment.analysis, "immutable_reusable_iteration", "immutable-iteration")
    watch(experiment.mechanism_program, "bind_analysis", "bind")
    binding = experiment.analysis.bind_mechanism_work_order_analysis(record)
    assert events == ["inputs", "bind", "inputs"]
    events.clear()
    assert experiment.analysis.bind_mechanism_work_order_analysis(record) == binding
    assert events == ["inputs", "immutable-iteration", "bind"]


def test_absent_work_order_analysis_binding_remains_a_noop(tmp_path, monkeypatch):
    experiment, _, calls = fixtures.setup_experiment(tmp_path)
    monkeypatch.setattr(
        experiment.revision_session, "check_inputs", lambda: pytest.fail("absent work order invoked admission")
    )
    assert experiment.analysis.bind_mechanism_work_order_analysis({}) is None
    assert calls == []


@pytest.mark.parametrize("reject_snapshot", [False, True])
def test_mechanism_round_preserves_both_scope_admission_checkpoints(tmp_path, monkeypatch, reject_snapshot):
    experiment, candidate, calls = fixtures.setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def optimize():\n return 1\ndef schedule():\n return 1\n")
    fixtures._freeze_test_mechanism_catalog(
        experiment,
        candidate,
        tmp_path,
        [{"id": "epilogue", "selectors": [{"kind": "function", "path": "compiler.py", "symbol": "optimize"}]}],
    )
    events = []
    original_scope = experiment.revision_session.validate_candidate_scope

    def scope(path):
        events.append("candidate" if path == candidate else "snapshot")
        if path != candidate and reject_snapshot:
            raise ValueError("snapshot admission refused")
        return original_scope(path)

    def watch(owner, name, label):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            events.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, observed)

    monkeypatch.setattr(experiment.revision_session, "validate_candidate_scope", scope)
    watch(experiment.revision_session, "check_inputs", "inputs")
    watch(experiment.mechanism_rounds, "capture", "capture")
    watch(experiment.mechanism_rounds, "publish_start", "publish")
    expected = ["inputs", "candidate", "inputs", "capture", "snapshot"]
    if reject_snapshot:
        with pytest.raises(ValueError, match="snapshot admission refused"):
            experiment.revision_session.begin_mechanism_round(candidate, round_index=0)
        assert events == expected
        assert not (experiment.output / "mechanism_round_start_0000.json").exists()
        assert experiment.mechanism_rounds.active is None
    else:
        experiment.revision_session.begin_mechanism_round(candidate, round_index=0)
        assert events == expected + ["inputs", "publish"]
        events.clear()
        compiler = candidate / "compiler.py"
        compiler.write_text(compiler.read_text().replace("return 1", "return 2", 1))
        watch(experiment.mechanism_rounds, "finalize", "finalize")
        assert experiment.revision_session.finalize_mechanism_round(candidate, round_index=0)["status"] == "allowed"
        assert events == ["inputs", "finalize"]
    assert calls == []


@pytest.mark.parametrize("configured", [False, True])
def test_edit_authority_keeps_native_input_admission_order(tmp_path, monkeypatch, configured):
    experiment, candidate, _ = fixtures.setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def schedule():\n return 1\n")
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    if configured:
        contract = {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
            "helper_extensions": [],
        }
        contract["sha256"] = fixtures.P2_CONTRACTS.document_sha256(contract)
        experiment.freeze_edit_scope(candidate, contract)
    events = []
    original_inputs = experiment.revision_session.check_inputs
    original_validate = experiment.edit_authority.validate_candidate
    original_inspect = experiment.edit_authority.inspect_optimization_surfaces

    def inputs():
        events.append("inputs")
        return original_inputs()

    def validate(path):
        events.append("candidate")
        return original_validate(path)

    def inspect(path):
        events.append("surfaces")
        return original_inspect(path)

    monkeypatch.setattr(experiment.revision_session, "check_inputs", inputs)
    monkeypatch.setattr(experiment.edit_authority, "validate_candidate", validate)
    monkeypatch.setattr(experiment.edit_authority, "inspect_optimization_surfaces", inspect)
    experiment.revision_session.inspect_optimization_surfaces(candidate)
    assert events == ["inputs"] * (2 if configured else 1) + ["candidate", "surfaces"]
    events.clear()
    experiment.revision_session.validate_candidate_scope(candidate)
    assert events == (["inputs"] if configured else []) + ["candidate"]


@pytest.mark.parametrize(
    "entrypoint",
    ["validate_candidate_scope", "inspect_optimization_surfaces"],
)
def test_cleared_frozen_contract_cannot_disable_native_admission(tmp_path, monkeypatch, entrypoint):
    from merlin.perf import compiler_edit_scope

    experiment, candidate, calls = fixtures.setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def schedule():\n return 1\n")
    contract = {
        "schema": "compiler_edit_contract_v1",
        "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
        "helper_extensions": [],
    }
    contract["sha256"] = fixtures.P2_CONTRACTS.document_sha256(contract)
    experiment.freeze_edit_scope(candidate, contract)
    experiment.edit_authority.contract = None
    admissions = []
    original = experiment.revision_session.check_inputs

    def inputs():
        admissions.append("inputs")
        return original()

    monkeypatch.setattr(experiment.revision_session, "check_inputs", inputs)
    monkeypatch.setattr(
        compiler_edit_scope, "inspect_compiler_edits", lambda *_: pytest.fail("corrupted authority reached AST checker")
    )
    with pytest.raises(ValueError, match="authority changed"):
        getattr(experiment.revision_session, entrypoint)(candidate)
    assert admissions == ["inputs"]
    assert calls == []


@pytest.mark.parametrize("reject_second", [False, True])
def test_static_import_keeps_scope_and_readiness_before_each_artifact_decode(tmp_path, monkeypatch, reject_second):
    seed, seed_candidate, _ = fixtures._portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"}, name="order_seed"
    )
    seed.analysis.analyze(seed_candidate, hypothesis="produce ordered portfolio seed")
    checkpoint = seed.revision_session.seal(seed_candidate)
    current, candidate, calls = fixtures._portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"}, name="order_current"
    )
    events = []
    active = [None]
    original_iter = static_cache.LoadedStaticBundle.__iter__
    original_decode = static_cache.StaticCacheMember.decode_artifacts
    original_scope = current.revision_session.validate_candidate_scope
    original_readiness = fixtures.EA.global_iteration_readiness

    def members(bundle):
        for index, member in enumerate(original_iter(bundle)):
            active[0] = index
            events.append((index, "identity"))
            yield member
        active[0] = None

    def scope(path):
        if active[0] is not None:
            events.append((active[0], "scope"))
            if reject_second and active[0] == 1:
                raise ValueError("synthetic second-member scope refusal")
        return original_scope(path)

    def readiness(analysis):
        if active[0] is not None:
            events.append((active[0], "readiness"))
        return original_readiness(analysis)

    def decode(member, *, analysis):
        events.append((active[0], "decode"))
        return original_decode(member, analysis=analysis)

    monkeypatch.setattr(static_cache.LoadedStaticBundle, "__iter__", members)
    monkeypatch.setattr(static_cache.StaticCacheMember, "decode_artifacts", decode)
    monkeypatch.setattr(current.revision_session, "validate_candidate_scope", scope)
    monkeypatch.setattr(fixtures.EA, "global_iteration_readiness", readiness)
    if reject_second:
        with pytest.raises(ValueError, match="second-member scope refusal"):
            current.static_analysis_import.import_checkpoint(
                candidate, checkpoint=checkpoint, checkpoint_sha256=fixtures.P2_CONTRACTS.sha256_file(checkpoint)
            )
        assert current.revisions.iterations == []
    else:
        receipt = current.static_analysis_import.import_checkpoint(
            candidate, checkpoint=checkpoint, checkpoint_sha256=fixtures.P2_CONTRACTS.sha256_file(checkpoint)
        )
        assert receipt["status"] == "hit"
    assert events == [(0, "identity"), (0, "scope"), (0, "readiness"), (0, "decode")] + (
        [(1, "identity"), (1, "scope")]
        if reject_second
        else [(1, "identity"), (1, "scope"), (1, "readiness"), (1, "decode")]
    )
    assert calls == []


@pytest.mark.parametrize(
    ("accessor", "expected"),
    [("optimization_baseline_artifacts", 2), ("optimization_baseline_artifact_binding", 3)],
)
def test_baseline_access_preserves_native_admission_checkpoints(tmp_path, monkeypatch, accessor, expected):
    experiment, candidate, _, _ = fixtures._comparison_arm_fixture(tmp_path)
    original = experiment.revision_session.current
    admissions = []

    def admitted(path):
        admissions.append(path)
        return original(path)

    monkeypatch.setattr(experiment.revision_session, "current", admitted)
    observed = getattr(experiment.revision_session, accessor)(candidate)
    assert admissions == [candidate] * expected
    assert observed["structural_plan_status"] == "UNVERIFIED"
    assert observed["numerical_qualification"] == "UNPROVEN"
    admissions.clear()
    experiment.revisions.baseline_artifacts["lowered_text"] += "changed"
    with pytest.raises(ValueError, match="artifact bytes changed"):
        getattr(experiment.revision_session, accessor)(candidate)
    # Invalid baseline evidence refuses before the nested current-artifact admission.
    assert admissions == [candidate]


def test_fresh_duplicate_and_reverted_submissions_keep_chronological_artifacts(tmp_path, monkeypatch):
    snapshot, snapshot_sha = fixtures._make_test_source_snapshot(tmp_path, "policy", "POLICY = 1\n")
    policy = fixtures._test_host_policy(snapshot)
    monkeypatch.setattr(GI.HP, "build_record", lambda **kwargs: copy.deepcopy(policy))
    analyzer = fixtures._static_cache_analyzer(calls := [])
    monkeypatch.setattr(fixtures.EA, "analyze_whole_model_emission", analyzer)
    experiment, candidate = fixtures._make_static_cache_experiment(
        tmp_path, "current", snapshot, snapshot_sha, analyzer
    )
    original = (candidate / "source.txt").read_text()
    first = experiment.analysis.analyze(candidate, hypothesis="initial synthetic analysis")
    journal = experiment.revisions
    artifacts = journal.artifacts
    portfolio = journal.portfolio_artifacts
    path = experiment.output / "iteration_0000.json"
    raw = path.read_bytes()
    digest = fixtures.P2_CONTRACTS.sha256_file(path)
    experiment.revisions.iterations[0]["probe_receipts"].append({"path": "synthetic.json", "sha256": "a" * 64})
    duplicate = experiment.analysis.analyze(candidate, hypothesis="same submitted bytes")
    assert duplicate["iteration"] == 0
    assert len(calls) == len(experiment.revisions.iterations) == 1
    assert journal.artifacts is artifacts
    assert journal.portfolio_artifacts is portfolio
    (candidate / "source.txt").write_text("changed synthetic compiler")
    experiment.analysis.analyze(candidate, hypothesis="second submitted revision")
    second_artifacts = journal.artifacts
    second_portfolio = journal.portfolio_artifacts
    (candidate / "source.txt").write_text(original)
    revisited = experiment.analysis.analyze(candidate, hypothesis="return to first compiler")
    assert len(calls) == 2
    assert [row["iteration"] for row in experiment.revisions.iterations] == [0, 1, 2]
    assert revisited["candidate_sha256"] == first["candidate_sha256"]
    assert revisited["static_comparison"]["previous_iteration"] == 1
    assert revisited["probe_receipts"] == []
    assert journal.artifacts == artifacts
    assert journal.previous_artifacts is second_artifacts
    assert journal.previous_portfolio_artifacts is second_portfolio
    assert journal.artifacts is journal.iteration_artifacts[2]
    assert journal.portfolio_artifacts is journal.iteration_portfolio_artifacts[2]
    assert path.read_bytes() == raw
    assert journal.record_sha256[0] == fixtures.P2_CONTRACTS.sha256_file(path) == digest


def test_cross_run_import_then_analysis_preserves_imported_previous_identity(tmp_path, monkeypatch):
    snapshot, snapshot_sha = fixtures._make_test_source_snapshot(tmp_path, "policy", "POLICY = 1\n")
    policy = fixtures._test_host_policy(snapshot)
    monkeypatch.setattr(GI.HP, "build_record", lambda **kwargs: copy.deepcopy(policy))
    analyzer = fixtures._static_cache_analyzer(calls := [])
    monkeypatch.setattr(fixtures.EA, "analyze_whole_model_emission", analyzer)
    seed, seed_candidate = fixtures._make_static_cache_experiment(tmp_path, "seed", snapshot, snapshot_sha, analyzer)
    seed.analysis.analyze(seed_candidate, hypothesis="synthetic cache seed")
    checkpoint = seed.revision_session.seal(seed_candidate)
    current, candidate = fixtures._make_static_cache_experiment(tmp_path, "current", snapshot, snapshot_sha, analyzer)
    receipt = current.static_analysis_import.import_checkpoint(
        candidate, checkpoint=checkpoint, checkpoint_sha256=fixtures.P2_CONTRACTS.sha256_file(checkpoint)
    )
    assert receipt["status"] == "hit"
    assert len(calls) == 1
    journal = current.revisions
    imported_artifacts, imported_portfolio = journal.artifacts, journal.portfolio_artifacts
    path = current.output / "iteration_0000.json"
    raw = path.read_bytes()
    (candidate / "source.txt").write_text("new compiler after imported seed")
    row = current.analysis.analyze(candidate, hypothesis="continue imported chronology")
    assert len(calls) == 2
    assert [item["iteration"] for item in current.revisions.iterations] == [0, 1]
    assert row["static_comparison"]["previous_iteration"] == 0
    assert journal.previous_artifacts is imported_artifacts
    assert journal.previous_portfolio_artifacts is imported_portfolio
    assert path.read_bytes() == raw
    assert journal.record_sha256[0] == fixtures.P2_CONTRACTS.sha256_file(path)
