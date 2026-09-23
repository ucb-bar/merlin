"""Installed revision access and publication over synthetic, content-bound evidence."""

import copy
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import portfolio_checkpoint as PC
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.global_inputs import GlobalExperimentInputs
from merlin_experiments.phase2.mechanism_program import MechanismProgram
from merlin_experiments.phase2.mechanism_rounds import MechanismRounds
from merlin_experiments.phase2.portfolio_evaluation import FastPortfolioEvaluation
from merlin_experiments.phase2.revision_journal import RevisionJournal
from merlin_experiments.phase2.revision_session import RevisionSession
from merlin_experiments.phase2.stage_inputs import StageE2ESentinel

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.perf import phase2_edit_contract


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("revision sessions must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def digest(text):
    return sha256_bytes(text.encode())


def publish(case, *, blocked=False):
    """Publish synthetic static evidence; no claim of actual compiler qualification."""
    index = len(case.journal.iterations)
    candidate = case.root / f"candidate-{index}"
    candidate.mkdir()
    (candidate / "compiler.py").write_text(
        f"from merlin.helper import VALUE\n\ndef schedule():\n    return VALUE + {index}\n"
    )
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    candidate_digest = hash_tree(candidate)["sha256"]
    snapshot = case.output / f"submitted-{index}"
    shutil.copytree(candidate, snapshot)
    for path in [*snapshot.rglob("*"), snapshot]:
        path.chmod(path.stat().st_mode & ~0o222)
    analyses, artifacts = [], {}
    baseline_lowered, baseline_commands = "baseline synthetic IR", '{"baseline":true}'
    for member_index, sentinel in enumerate(case.inputs.portfolio_sentinels):
        interface = Path(sentinel.frozen_source_path) / "capsule.interface.mlir"
        source_digest = C.sha256_file(interface)
        lowered = f"synthetic IR revision={index} member={member_index}"
        commands = json.dumps({"revision": index, "member": member_index})
        emission = {
            "candidate_lowered_sha256": digest(lowered),
            "candidate_command_buffer_sha256": digest(commands),
            "baseline_lowered_sha256": digest(baseline_lowered),
            "baseline_command_buffer_sha256": digest(baseline_commands),
        }
        graph_digest = digest(f"graph-{member_index}")
        plan = {
            "status": "verified",
            "plan_digest": digest(f"plan-{index}-{member_index}"),
            "candidate_sha256": candidate_digest,
            "logical_dispatch_digest": graph_digest,
            "source_sha256": source_digest,
            **emission,
        }
        if blocked and member_index == 1:
            plan = {}
        analyses.append(
            {
                "candidate_sha256": candidate_digest,
                "workload": {"capsule_sha256": sentinel.capsule_sha256},
                "emission": emission,
                "diagnostics": {
                    "captured_logical_graph": {
                        "status": "verified",
                        "logical_dispatch_digest": graph_digest,
                        "source_sha256": source_digest,
                    },
                    "arms": {"candidate": {"status": "emitted"}},
                    "verified_global_plan_emission": plan,
                },
            }
        )
        artifacts[sentinel.capsule_sha256] = {
            "candidate_sha256": candidate_digest,
            "candidate_lowered_sha256": digest(lowered),
            "candidate_command_buffer_sha256": digest(commands),
            "lowered_text": lowered,
            "command_buffer_text": commands,
            "command_buffer": json.loads(commands),
            "interface": str(interface),
        }
    members = [
        {"identity": identity, "readiness": EA.global_iteration_readiness(analysis)}
        for identity, analysis in zip(case.inputs.portfolio_identity["members"], analyses, strict=True)
    ]
    members[0].update(analysis_ref="/analysis", static_comparison_ref="/static_comparison")
    members[1]["analysis"] = analyses[1]
    row = {
        "schema": "global_perf_iteration_v1",
        "iteration": index,
        "candidate_sha256": candidate_digest,
        "compiler_dependencies": case.inputs.compiler_dependencies(candidate),
        "submitted_snapshot": str(snapshot),
        "analysis": analyses[0],
        "readiness": EA.global_iteration_readiness(analyses[1]),
        "portfolio": {
            "portfolio_sha256": case.inputs.portfolio_identity_sha256,
            "candidate_sha256": candidate_digest,
            "members": members,
            "members_total": 2,
            "members_ready": 1 if blocked else 2,
            "full_model_simulation_allowed": False,
        },
        "probe_receipts": [],
        "baseline_sha256": case.inputs.baseline_sha256,
        "optimization_baseline_sha256": case.inputs.optimization_baseline_sha256,
        "optimization_baseline": case.inputs.optimization_baseline_binding,
    }
    for analysis in analyses:
        analysis["optimization_baseline"] = case.inputs.optimization_baseline_binding
    primary = artifacts[case.inputs.sentinel.capsule_sha256]
    primary["baseline_artifacts"] = {
        "identity": {
            "baseline_sha256": case.inputs.baseline_sha256,
            "capsule_sha256": case.inputs.sentinel.capsule_sha256,
            "target": case.inputs.target,
        },
        "lowered_text": baseline_lowered,
        "lowered_sha256": digest(baseline_lowered),
        "command_buffer_text": baseline_commands,
        "command_buffer_sha256": digest(baseline_commands),
    }
    case.journal.publish_analysis(row, primary, artifacts)
    return candidate, row


@pytest.fixture
def case(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("from merlin.helper import VALUE\n")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    (shared / "helper.py").write_text("VALUE = 1\n")
    sentinels = []
    for index in range(2):
        source = tmp_path / f"capsule-{index}"
        source.mkdir()
        (source / "capsule.interface.mlir").write_text(f"module {{ // member {index}\n}}\n")
        sentinels.append(
            StageE2ESentinel(
                source.name, str(source), str(source), C.exact_tree_record(source)["sha256"], ("lane",), ("L2",)
            )
        )
    controller = tmp_path / "controller.py"
    controller.write_text("SYNTHETIC = True\n")
    resources = tmp_path / "contract"
    for relative in HP.RESOURCE_FILES:
        path = resources / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_resource": relative}))
    output = tmp_path / "run"
    prepared = GlobalExperimentInputs.prepare(
        baseline=baseline,
        baseline_sha256=hash_tree(baseline)["sha256"],
        sentinel=sentinels[0],
        portfolio_sentinels=sentinels[1:],
        target="synthetic-unregistered-target",
        target_sha256=digest("target"),
        output=output,
        compiler_shared_source_root=shared,
        contract_root=resources,
        controller_source=controller,
    )
    output.mkdir()
    inputs = prepared.materialize(output)
    edit = FrozenEditAuthority(output)
    program = MechanismProgram(
        output,
        edit,
        portfolio_identity=inputs.portfolio_identity,
        portfolio_identity_sha256=inputs.portfolio_identity_sha256,
    )
    rounds = MechanismRounds(program)
    fast = FastPortfolioEvaluation(inputs.portfolio_sentinels)
    journal = RevisionJournal(output)
    session = RevisionSession(
        inputs=inputs,
        edit_authority=edit,
        mechanism_program=program,
        mechanism_rounds=rounds,
        fast_evaluation=fast,
        journal=journal,
    )
    experiment = {
        "baseline_sha256": inputs.baseline_sha256,
        "target_sha256": inputs.target_sha256,
        "optimization_baseline_sha256": inputs.optimization_baseline_sha256,
        "optimization_baseline": inputs.optimization_baseline_binding,
        "phase1_qualification": inputs.phase1_binding,
        "portfolio": inputs.portfolio_identity,
        "portfolio_sha256": inputs.portfolio_identity_sha256,
    }
    (output / "experiment.json").write_bytes(C.canonical_json(experiment))
    return SimpleNamespace(
        root=tmp_path,
        output=output,
        inputs=inputs,
        edit=edit,
        program=program,
        rounds=rounds,
        fast=fast,
        journal=journal,
        session=session,
    )


def test_current_requires_analyzed_candidate_and_ready_revision(case):
    candidate = case.root / "unsubmitted"
    shutil.copytree(case.inputs.baseline, candidate)
    with pytest.raises(ValueError, match="compile the complete-model graph"):
        case.session.current(candidate)
    candidate, row = publish(case, blocked=True)
    assert case.session.current(candidate, require_ready=False) is row
    with pytest.raises(ValueError, match="not ready"):
        case.session.current(candidate)
    with pytest.raises(ValueError, match="not ready"):
        with case.session.action(candidate):
            pytest.fail("blocked revision admitted to ready action")


def test_changed_candidate_and_shared_dependency_refused(case):
    candidate, row = publish(case)
    assert case.session.current(candidate) is row
    original = (candidate / "compiler.py").read_text()
    (candidate / "compiler.py").write_text("CHANGED = True\n")
    with pytest.raises(ValueError, match="candidate changed"):
        case.session.current(candidate)
    (candidate / "compiler.py").write_text(original)
    (case.inputs.compiler_shared_source_root / "helper.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="shared compiler dependencies changed"):
        case.session.current(candidate)


def test_candidate_admission_enforces_real_frozen_edit_scope(case):
    candidate = case.root / "editable"
    candidate.mkdir()
    source = candidate / "compiler.py"
    source.write_text("def schedule():\n    return 1\n\ndef protected():\n    return 2\n")
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    contract = phase2_edit_contract.seal(
        {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
            "helper_extensions": [],
        }
    )
    case.edit.freeze(candidate, contract, has_iterations=False)
    assert case.session.validate_candidate_scope(candidate)["status"] == "allowed"
    source.write_text("def schedule():\n    return 3\n\ndef protected():\n    return 2\n")
    assert case.session.validate_candidate_scope(candidate)["status"] == "allowed"
    source.write_text("def schedule():\n    return 3\n\ndef protected():\n    return 4\n")
    with pytest.raises(ValueError):
        case.session.validate_candidate_scope(candidate)


def test_action_caches_only_within_action_and_cleans_up_exceptions(case, monkeypatch):
    candidate, row = publish(case)
    original = case.inputs.verify
    calls = []

    def counted(**kwargs):
        calls.append(threading.get_ident())
        return original(**kwargs)

    monkeypatch.setattr(case.inputs, "verify", counted)
    with pytest.raises(RuntimeError, match="synthetic action failure"):
        with case.session.action(candidate) as (outer, owns):
            assert outer is row and owns is True
            with case.session.action(candidate) as (inner, nested_owns):
                assert inner is row and nested_owns is False
                assert case.session.current(candidate) is row
                case.session.current_probe_binding(candidate)
                case.session.current_artifacts(candidate)
            assert len(calls) == 1
            raise RuntimeError("synthetic action failure")
    assert case.session.current(candidate) is row
    assert len(calls) == 2


def test_action_cannot_substitute_candidate_or_new_revision(case):
    candidate, _ = publish(case)
    another = case.root / "same-bytes-different-path"
    shutil.copytree(candidate, another)
    with case.session.action(candidate):
        with pytest.raises(ValueError, match="substitute another candidate"):
            case.session.current(another)
        with pytest.raises(ValueError, match="substitute another candidate"):
            with case.session.action(another):
                pytest.fail("substituted candidate entered nested action")
        replacement = copy.deepcopy(case.journal.iterations[-1])
        replacement["iteration"] += 1
        case.journal.iterations.append(replacement)
        with pytest.raises(ValueError, match="revision changed"):
            case.session.revalidate(candidate)


def test_revalidate_candidate_then_refresh_all_inputs(case, monkeypatch):
    candidate, row = publish(case)
    original = case.inputs.verify
    calls = []

    def counted(**kwargs):
        calls.append(True)
        return original(**kwargs)

    monkeypatch.setattr(case.inputs, "verify", counted)
    with case.session.action(candidate):
        assert case.session.revalidate(candidate) is row
        assert len(calls) == 1
        assert case.session.refresh(candidate) is row
        assert len(calls) == 2
        case.inputs.controller_source.write_text("CHANGED = True\n")
        assert case.session.current(candidate) is row
        assert case.session.revalidate(candidate) is row
        with pytest.raises(ValueError, match="host verification policy"):
            case.session.refresh(candidate)


def test_revalidate_refuses_candidate_mutation_within_action(case):
    candidate, row = publish(case)
    with case.session.action(candidate):
        (candidate / "compiler.py").write_text("CHANGED_DURING_ACTION = True\n")
        assert case.session.current(candidate) is row
        with pytest.raises(ValueError, match="candidate changed"):
            case.session.revalidate(candidate)


def test_action_cache_is_thread_local(case, monkeypatch):
    candidate, row = publish(case)
    original = case.inputs.verify
    observations, errors = [], []

    def counted(**kwargs):
        observations.append(threading.get_ident())
        return original(**kwargs)

    monkeypatch.setattr(case.inputs, "verify", counted)

    def other_thread():
        try:
            with case.session.action(candidate) as (current, owns):
                assert current is row and owns is True
                assert case.session.current(candidate) is row
        except BaseException as exc:
            errors.append(exc)

    with case.session.action(candidate):
        thread = threading.Thread(target=other_thread, daemon=True)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        assert case.session.current(candidate) is row
    assert not errors
    assert len(observations) == 2 and len(set(observations)) == 2


def test_previous_current_and_baseline_contexts_keep_distinct_authority(case):
    previous_candidate, previous = publish(case)
    with pytest.raises(ValueError, match="two submitted"):
        case.session.previous_artifacts(previous_candidate)
    candidate, current = publish(case)
    for index in range(2):
        before = case.session.previous_portfolio_member_context(candidate, index=index)
        after = case.session.current_portfolio_member_context(candidate, index=index)
        assert before["member_binding"]["candidate_sha256"] == previous["candidate_sha256"]
        assert after["member_binding"]["candidate_sha256"] == current["candidate_sha256"]
        assert before["member_binding"]["arm"] == "previous"
        assert after["member_binding"]["arm"] == "current"
    assert case.session.previous_probe_binding(candidate) != case.session.current_probe_binding(candidate)
    baseline = case.session.optimization_baseline_artifacts(candidate)
    binding = case.session.optimization_baseline_artifact_binding(candidate)
    assert baseline["structural_plan_status"] == "UNVERIFIED"
    assert baseline["numerical_qualification"] == "UNPROVEN"
    assert binding["schema"] == "optimization_baseline_artifact_binding_v1"
    assert binding["compiler_sha256"] == case.inputs.baseline_sha256
    selection = case.session.select_changed_portfolio_member(candidate)
    assert selection["portfolio_index"] == 0
    assert case.session.selected_changed_portfolio_context(candidate, selection)["selection"] == selection
    with pytest.raises(ValueError, match="stale or caller-substituted"):
        case.session.selected_changed_portfolio_context(candidate, {**selection, "portfolio_index": 1})


@pytest.mark.parametrize("artifact", ["current", "previous", "baseline"])
def test_changed_artifact_bytes_refused(case, artifact):
    publish(case)
    candidate, _ = publish(case)
    if artifact == "baseline":
        case.journal.baseline_artifacts["lowered_text"] += "changed"

        def accessor():
            return case.session.optimization_baseline_artifacts(candidate)
    else:
        by_capsule = (
            case.journal.portfolio_artifacts if artifact == "current" else case.journal.previous_portfolio_artifacts
        )
        by_capsule[case.inputs.sentinel.capsule_sha256]["lowered_text"] += "changed"
        method = (
            case.session.current_portfolio_member_context
            if artifact == "current"
            else case.session.previous_portfolio_member_context
        )

        def accessor():
            return method(candidate, index=0)

    with pytest.raises(ValueError, match="artifact.*changed|binding changed"):
        accessor()


@pytest.mark.parametrize("blocked", [False, True])
def test_checkpoint_publication_readonly_exact_bytes_and_installed_consumption(case, blocked):
    candidate, row = publish(case, blocked=blocked)
    path = (
        case.session.checkpoint_authoring(candidate, name="next_authoring") if blocked else case.session.seal(candidate)
    )
    document = C.mapping_file(path)
    assert not path.stat().st_mode & 0o222
    snapshot = Path(document["candidate_path"])
    assert hash_tree(snapshot)["sha256"] == row["candidate_sha256"]
    assert all(not item.stat().st_mode & 0o222 for item in [snapshot, *snapshot.rglob("*")])
    assert document["iteration_record_sha256"] == case.journal.record_sha256[row["iteration"]]
    assert document["analysis_sha256"] == C.document_sha256(row["analysis"])
    assert document["full_model_timing_status"] == "UNMEASURED"
    assert document["global_speedup_proven"] is False
    context = PC.CheckpointVerificationContext(case.inputs.host_policy, case.inputs.compiler_shared_source_root)
    assert PC.consume_round_checkpoint(path, context=context) == document
    if blocked:
        assert document["consumer"] == "next_bounded_global_authoring_round_only"
        with pytest.raises(ValueError, match="not ready"):
            case.session.seal(candidate, name="invalid-ready")
    else:
        assert document["promotion_status"] == "unqualified_candidate_for_review"
        with pytest.raises(ValueError, match="only for blocked"):
            case.session.checkpoint_authoring(candidate, name="invalid-blocked")


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("mutation", ["static_bytes", "static_link", "static_missing", "submitted_bytes"])
def test_publication_cannot_repin_changed_or_redirected_static_evidence(case, blocked, mutation):
    candidate, row = publish(case, blocked=blocked)
    iteration_path = case.output / f"iteration_{row['iteration']:04d}.json"
    if mutation == "static_bytes":
        iteration_path.chmod(0o644)
        iteration_path.write_text("{}")
    elif mutation == "static_link":
        copied = case.root / "same-byte-iteration.json"
        copied.write_bytes(iteration_path.read_bytes())
        iteration_path.unlink()
        iteration_path.symlink_to(copied)
    elif mutation == "static_missing":
        iteration_path.unlink()
    else:
        submitted = Path(row["submitted_snapshot"]) / "compiler.py"
        submitted.chmod(0o644)
        submitted.write_text("changed submitted bytes")
    with pytest.raises(ValueError):
        if blocked:
            case.session.checkpoint_authoring(candidate, name="refused")
        else:
            case.session.seal(candidate, name="refused")
    assert not (case.output / "refused.json").exists()


def test_later_probe_receipt_does_not_rewrite_static_iteration_identity(case):
    candidate, row = publish(case)
    iteration_path = case.output / "iteration_0000.json"
    original = iteration_path.read_bytes()
    probe = {
        "binding": case.session.current_probe_binding(candidate).to_dict(),
        "scope": "mechanism_probe_only",
        "full_model_cycles": None,
        "warmup_runs": 1,
        "measured_runs": 1,
    }
    probe_path = case.output / "probe.json"
    probe_path.write_bytes(C.canonical_json(probe))
    row["probe_receipts"].append({"path": str(probe_path), "sha256": C.sha256_file(probe_path)})
    sealed = case.session.seal(candidate)
    document = C.mapping_file(sealed)
    assert iteration_path.read_bytes() == original
    assert document["iteration_record_sha256"] == sha256_bytes(original)
    assert document["probe_receipts"] == row["probe_receipts"]
    assert C.mapping_file(iteration_path)["probe_receipts"] == []
    context = PC.CheckpointVerificationContext(case.inputs.host_policy, case.inputs.compiler_shared_source_root)
    assert PC.consume_global_candidate(sealed, context=context) == document


@pytest.mark.parametrize("name", ["", "..", "../escape", "/absolute", "nested/name"])
@pytest.mark.parametrize("blocked", [False, True])
def test_publication_name_is_one_component(case, name, blocked):
    candidate, _ = publish(case, blocked=blocked)
    with pytest.raises(ValueError, match="safe path component"):
        if blocked:
            case.session.checkpoint_authoring(candidate, name=name)
        else:
            case.session.seal(candidate, name=name)


@pytest.mark.parametrize("mutation", [None, "baseline", "controller"])
def test_whole_model_broker_inventory_uses_real_session_and_terminal_refusal(case, mutation):
    candidate = case.root / "inventory-candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("def schedule():\n    return 1\n")
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    contract = phase2_edit_contract.seal(
        {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
            "helper_extensions": [],
        }
    )
    declaration = {
        "id": "schedule",
        "scope": "codegen",
        "path": "compiler.py",
        "symbol": "schedule",
        "effects": ["latency_hiding", "movement"],
        "cca_axes": ["dispatch.dma_overlap"],
        "mechanism": "Schedule independent transfers",
        "emitted_delta": "Changed issue ordering",
        "validation": "Source dependency proof",
        "abandonment": "Invalid dependency",
    }
    case.edit.freeze(candidate, contract, has_iterations=False, host_surface_declarations=[declaration])
    receipt_path = case.output / "broker-control" / "receipts.jsonl"
    action = B.BrokerAction(BP.INVENTORY_ACTION, ("must-never-execute",), (), "inspect scoped surfaces", False)
    experiment = SimpleNamespace(revision_session=case.session)
    workflow = BP.select_workflow(
        BP.WHOLE_MODEL_V1,
        candidate=candidate,
        target_experiment=SimpleNamespace(),
        receipt_path=receipt_path,
        global_experiment=experiment,
    )
    broker = B.Broker(
        SimpleNamespace(argv=(), network="available_not_an_isolation_claim", clear_environment=True),
        SimpleNamespace(),
        candidate,
        (action,),
        receipt_path,
        deadline=time.monotonic() + 30,
        max_calls=1,
        max_tool_seconds=10,
        workflow=workflow,
    )
    if mutation == "baseline":
        (case.inputs.baseline / "compiler.py").write_text("CHANGED_BASELINE = True\n")
    elif mutation == "controller":
        case.inputs.controller_source.write_text("CHANGED_CONTROLLER = True\n")
    if mutation is None:
        result = broker.execute({"action": BP.INVENTORY_ACTION})
        assert result["returncode"] == 0
        inventory = json.loads(result["stdout"])
        assert [surface["id"] for surface in inventory["surfaces"]] == ["schedule"]
        assert inventory["surfaces"][0]["mechanism"] == declaration["mechanism"]
    else:
        with pytest.raises(B.StageGateError, match="input integrity refused"):
            broker.execute({"action": BP.INVENTORY_ACTION})
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert len(receipts) == 1
    assert receipts[0]["index"] == 0
    assert receipts[0]["action"] == BP.INVENTORY_ACTION
    assert receipts[0]["state"] == ("complete" if mutation is None else "rejected")
    assert (receipts[0]["returncode"] == 0) is (mutation is None)
