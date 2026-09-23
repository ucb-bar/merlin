"""Recorded-source static import with synthetic emissions, not archived execution proof."""

import copy
import importlib.util
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments import source_snapshot
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import static_analysis_import as IMPORT

from merlin.common.digest import sha256_bytes


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("static import must not execute compilers, archived controllers, or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def helper_module():
    spec = importlib.util.spec_from_file_location(
        "portfolio_analysis_fixtures", Path(__file__).with_name("portfolio_analysis_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def captured_policy(tmp_path):
    controller = tmp_path / "capture-controller.py"
    controller.write_text("SYNTHETIC_CONTROLLER = True\n")
    resources = tmp_path / "capture-contract"
    for relative in HP.RESOURCE_FILES:
        path = resources / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_resource": relative}))
    return HP.build_record(controller_source=controller, contract_root=resources)


def snapshot_policy(captured, root):
    """Copy real captured installed bytes and explicitly preserve recorded closure ownership."""
    staging = root / "staging"
    staging.mkdir(parents=True)
    record = copy.deepcopy(captured)
    destinations = {}
    for namespace, closure in record["closures"].items():
        directory = staging / "policy/closures" / namespace
        closure["roots"] = [str(directory)]
        for relative, identity in closure["members"].items():
            destinations[identity] = directory / relative
    for identity in record["identities"]:
        if identity not in destinations:
            destinations[identity] = staging / "policy/individual" / identity
    for identity, original in captured["identities"].items():
        destination = destinations[identity]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(original).read_bytes())
    compiler = staging / "compiler"
    compiler.mkdir()
    (compiler / "__init__.py").write_text("")
    (compiler / "helper.py").write_text("VALUE = 1\n")
    snapshot = root / "sealed"
    source_snapshot.create(
        staging,
        snapshot,
        output_root=root / "unused-output",
        source_roots=("policy", "compiler"),
        python_roots=(),
        legacy_roots=(),
    )
    receipt = source_snapshot.verify(snapshot)
    record["sources"] = {}
    for identity, destination in destinations.items():
        selected = snapshot / destination.relative_to(staging)
        record["identities"][identity] = str(selected)
        record["sources"][str(selected)] = captured["sources"][captured["identities"][identity]]
    for namespace, closure in record["closures"].items():
        closure["roots"] = [str(snapshot / "policy/closures" / namespace)]
    record["location_sha256"] = C.document_sha256({key: record[key] for key in ("sources", "identities", "closures")})
    assert HP.content_sha256(record, source_root=snapshot) == captured["sha256"]
    return snapshot, C.document_sha256(receipt["files"]), record


@pytest.fixture
def cases(tmp_path, monkeypatch):
    original_build_record = HP.build_record
    captured = captured_policy(tmp_path)
    helper = helper_module()
    seed_snapshot, seed_snapshot_sha, seed_policy = snapshot_policy(captured, tmp_path / "seed-source")
    current_snapshot, current_snapshot_sha, current_policy = snapshot_policy(captured, tmp_path / "current-source")
    # Only the live observation is fixture-selected. Real snapshot sealing, recorded V3
    # source admission, artifact hashes and all concrete owners remain unpatched.
    monkeypatch.setattr(HP, "build_record", lambda **kwargs: copy.deepcopy(seed_policy))
    seed = helper.build_case(
        tmp_path / "seed",
        monkeypatch,
        source_snapshot_root=seed_snapshot,
        source_snapshot_files_sha256=seed_snapshot_sha,
    )
    seed.owner.analyze(seed.candidate, hypothesis="synthetic seed static analysis")
    seed.journal.iterations[0]["probe_receipts"] = [{"synthetic_dynamic_marker": "never imported"}]
    seed.journal.iterations[0]["semantic_receipts"] = [{"synthetic_dynamic_marker": "never imported"}]
    seed.journal.iterations[0]["decision_feedback"] = {"synthetic_dynamic_marker": "never imported"}
    checkpoint = seed.session.seal(seed.candidate)
    monkeypatch.setattr(HP, "build_record", lambda **kwargs: copy.deepcopy(current_policy))
    current = helper.build_case(
        tmp_path / "current",
        monkeypatch,
        source_snapshot_root=current_snapshot,
        source_snapshot_files_sha256=current_snapshot_sha,
    )
    importer = IMPORT.StaticAnalysisImport(
        current.owner,
        prior_shared_source_relative=Path("compiler"),
        prior_shared_source_fallback=current.inputs.compiler_shared_source_root,
    )
    return SimpleNamespace(
        seed=seed,
        current=current,
        importer=importer,
        checkpoint=checkpoint,
        checkpoint_sha256=C.sha256_file(checkpoint),
        seed_policy=seed_policy,
        current_policy=current_policy,
        captured=captured,
        original_build_record=original_build_record,
    )


def import_seed(cases):
    return cases.importer.import_checkpoint(
        cases.current.candidate, checkpoint=cases.checkpoint, checkpoint_sha256=cases.checkpoint_sha256
    )


def test_real_sealed_snapshot_static_hit_rebinds_current_artifacts_without_dynamic_evidence(cases):
    receipt = import_seed(cases)
    assert receipt["status"] == "hit"
    assert len(cases.seed.analyzer.calls) == 2
    assert cases.current.analyzer.calls == []
    assert receipt["full_graph_compiler_invoked"] is False
    assert receipt["full_model_simulation_executed"] is False
    assert receipt["probe_or_timing_receipts_reused"] is False
    assert receipt["semantic_or_decision_feedback_reused"] is False
    row = cases.current.journal.iterations[0]
    assert row["readiness"]["status"] == "ready_for_probe_admission"
    assert row["probe_receipts"] == [] and not row.get("semantic_receipts") and not row.get("decision_feedback")
    assert row["portfolio"]["analysis_concurrency"]["admitted_workers"] == 0
    for index, sentinel in enumerate(cases.current.inputs.portfolio_sentinels):
        context = cases.current.session.current_portfolio_member_context(cases.current.candidate, index=index)
        assert context["interface"].is_relative_to(Path(sentinel.frozen_source_path))
        assert context["member_binding"]["lowered_sha256"] == sha256_bytes(
            context["artifacts"]["lowered_text"].encode()
        )
        assert (
            context["artifacts"]["lowered_text"]
            == cases.seed.journal.portfolio_artifacts[sentinel.capsule_sha256]["lowered_text"]
        )
    assert not Path(row["submitted_snapshot"]).stat().st_mode & 0o222
    assert receipt["compiler_sandbox_reconstruction"]["previous_probe_compilation_available"] is False
    with pytest.raises(ValueError, match="exactly once"):
        import_seed(cases)


def test_exact_content_change_is_recorded_miss_then_normal_analysis_remains_available(cases):
    (cases.current.candidate / "version.txt").write_text("1")
    result = import_seed(cases)
    assert result["status"] == "miss" and result["reason"].startswith("exact_content_identity_changed:")
    assert cases.current.journal.iterations == [] and cases.current.analyzer.calls == []
    with pytest.raises(ValueError, match="exactly once"):
        import_seed(cases)
    row = cases.current.owner.analyze(cases.current.candidate, hypothesis="cold analysis after a safe miss")
    assert row["iteration"] == 0 and len(cases.current.analyzer.calls) == 2


def test_different_verified_controller_source_is_policy_miss(cases, monkeypatch):
    controller = cases.current.root / "different-controller.py"
    controller.write_text("SYNTHETIC_CONTROLLER = 'different implementation'\n")
    resource = Path(cases.captured["identities"]["resource/contract/schemas/manifest.schema.json"])
    captured = cases.original_build_record(controller_source=controller, contract_root=resource.parent.parent)
    snapshot, snapshot_sha, policy = snapshot_policy(captured, cases.current.root / "different-source")
    cases.current.inputs.host_policy = policy
    cases.current.inputs.source_snapshot_root = snapshot
    cases.current.inputs.source_snapshot_files_sha256 = snapshot_sha
    monkeypatch.setattr(HP, "build_record", lambda **kwargs: copy.deepcopy(policy))
    result = import_seed(cases)
    assert result["status"] == "miss"
    assert result["reason"] == "host_verification_policy_content_changed"
    assert not cases.current.analyzer.calls and not cases.current.journal.iterations


def test_prior_shared_source_fallback_is_explicit_not_checkout_discovery(cases):
    cases.importer = IMPORT.StaticAnalysisImport(
        cases.current.owner,
        prior_shared_source_relative=Path("intentionally-absent-layout"),
        prior_shared_source_fallback=cases.current.inputs.compiler_shared_source_root,
    )
    assert import_seed(cases)["status"] == "hit"
    assert not cases.current.analyzer.calls


def test_v3_does_not_infer_source_snapshot_from_policy_locations(cases):
    document = C.mapping_file(cases.checkpoint)
    document.pop("source_snapshot")
    cases.checkpoint.chmod(0o644)
    cases.checkpoint.write_bytes(C.canonical_json(document))
    cases.checkpoint.chmod(0o444)
    cases.checkpoint_sha256 = C.sha256_file(cases.checkpoint)
    with pytest.raises(ValueError, match="explicit source_snapshot"):
        import_seed(cases)


@pytest.mark.parametrize(
    "mutation",
    [
        "checkpoint_bytes",
        "checkpoint_link",
        "checkpoint_relative",
        "snapshot_bytes",
        "snapshot_link",
        "iteration_bytes",
        "candidate_bytes",
    ],
)
def test_tampered_or_redirected_evidence_refuses_and_consumes_attempt(cases, mutation):
    document = C.mapping_file(cases.checkpoint)
    if mutation == "checkpoint_bytes":
        cases.checkpoint.chmod(0o644)
        cases.checkpoint.write_text("{}")
    elif mutation == "checkpoint_link":
        alias = cases.current.root / "checkpoint-link.json"
        alias.symlink_to(cases.checkpoint)
        cases.checkpoint = alias
    elif mutation == "checkpoint_relative":
        cases.checkpoint = Path("relative.json")
    elif mutation == "snapshot_bytes":
        path = Path(cases.seed_policy["identities"]["controller/global"])
        path.chmod(0o644)
        path.write_text("CHANGED = True\n")
    elif mutation == "snapshot_link":
        alias = cases.current.root / "source-link"
        alias.symlink_to(cases.current.inputs.source_snapshot_root, target_is_directory=True)
        cases.current.inputs.source_snapshot_root = alias
    elif mutation == "iteration_bytes":
        path = Path(document["iteration_record"])
        path.chmod(0o644)
        path.write_text("{}")
    else:
        path = Path(document["candidate_path"]) / "version.txt"
        path.chmod(0o644)
        path.write_text("changed")
    with pytest.raises((ValueError, source_snapshot.SnapshotError)):
        import_seed(cases)
    assert cases.current.journal.iterations == [] and cases.current.analyzer.calls == []
    with pytest.raises(ValueError, match="exactly once"):
        import_seed(cases)


def test_attempt_after_existing_iteration_refused(cases):
    cases.current.owner.analyze(cases.current.candidate, hypothesis="already started")
    before = len(cases.current.analyzer.calls)
    with pytest.raises(ValueError, match="exactly once"):
        import_seed(cases)
    assert len(cases.current.analyzer.calls) == before


def test_initial_input_refusal_also_consumes_import_attempt(cases):
    source = cases.current.inputs.baseline / "version.txt"
    original = source.read_text()
    source.write_text("changed")
    with pytest.raises(ValueError, match="frozen compiler changed"):
        import_seed(cases)
    source.write_text(original)
    with pytest.raises(ValueError, match="exactly once"):
        import_seed(cases)


@pytest.mark.parametrize("invalid", [None, "dependencies", "overlay", "scratch"])
def test_current_sandbox_factory_reconstruction_without_execution(cases, invalid):
    from merlin.perf.analysis_worker import IsolatedAnalysisWorker

    calls = []

    def factory(baseline, candidate, scratch):
        calls.append((baseline, candidate, scratch))
        result = {}
        for arm, package in (("baseline", baseline), ("candidate", candidate)):
            result[arm] = {
                "package_path": str(package),
                "scratch_path": str(scratch),
                "compiler_dependencies": cases.current.inputs.compiler_dependencies(package),
                "command_prefix": ["never-executed-bwrap", "never-executed-python"],
                "bwrap_argv_length": 1,
                "answer_surfaces": [],
                "overlay_trees": {},
            }
        if invalid == "dependencies":
            result["candidate"]["compiler_dependencies"] = {}
        elif invalid == "overlay":
            overlay = cases.current.root / "synthetic-overlay"
            overlay.mkdir()
            source = overlay / "source.py"
            source.write_text("VALUE = 1\n")
            result["candidate"]["overlay_trees"] = {str(overlay): C.exact_tree_record(overlay)["sha256"]}
            source.write_text("VALUE = 2\n")
        elif invalid == "scratch":
            (scratch / "unexpected-output").write_text("factory must not execute or populate scratch")
        return result

    cases.current.owner.analyzer = IsolatedAnalysisWorker(
        analysis_source=Path(EA.__file__),
        contract_root=cases.current.inputs.contract_root,
        sandbox_factory=factory,
        output=cases.current.output / "unused-workers",
    )
    if invalid:
        reason = {
            "dependencies": "stale package, dependency, or policy identity",
            "overlay": "dependency overlay changed",
            "scratch": "unexpectedly populated compiler scratch",
        }
        with pytest.raises(ValueError, match=reason[invalid]):
            import_seed(cases)
        assert not cases.current.journal.iterations
    else:
        result = import_seed(cases)
        reconstruction = result["compiler_sandbox_reconstruction"]
        assert reconstruction["status"] == "prepared_from_current_trusted_factory"
        assert reconstruction["compiler_invoked"] is False
        assert reconstruction["previous_probe_compilation_available"] is True
        assert not list(Path(reconstruction["scratch"]).iterdir())
        assert cases.current.journal.compiler_sandbox_sha256[0] == reconstruction["policy_set_sha256"]
    assert len(calls) == 1
    assert cases.current.analyzer.calls == []
