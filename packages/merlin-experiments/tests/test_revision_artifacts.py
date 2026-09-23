"""Synthetic retained evidence exercises the installed journal without native engines."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2.revision_journal import RevisionJournal

from merlin.perf.mechanism_probe import ProbeBinding


@pytest.fixture(autouse=True)
def refuse_processes(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("artifact validation must not launch a process")

    monkeypatch.setattr(subprocess, "Popen", refused)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    sentinels = []
    for index in range(2):
        source = tmp_path / f"source-{index}"
        source.mkdir()
        (source / "interface.mlir").write_text(f"module {{ /* synthetic {index} */ }}")
        sentinels.append(
            SimpleNamespace(
                capsule=f"model-{index}",
                capsule_sha256=digest(f"capsule-{index}"),
                frozen_source_path=str(source),
                required_lanes=("synthetic",),
                required_tiers=("L2",),
            )
        )
    output = tmp_path / "journal"
    output.mkdir()
    return RevisionJournal(output), sentinels


def revision(sentinels, iteration):
    candidate = digest(f"candidate-{iteration}")
    members, portfolio = [], {}
    for index, sentinel in enumerate(sentinels):
        interface = Path(sentinel.frozen_source_path) / "interface.mlir"
        lowered = f"synthetic lowered {iteration}/{index}"
        command = json.dumps({"commands": [iteration, index]})
        emission = {
            "candidate_lowered_sha256": digest(lowered),
            "candidate_command_buffer_sha256": digest(command),
        }
        graph_digest = digest(f"graph-{index}")
        source_digest = contracts.sha256_file(interface)
        analysis = {
            "candidate_sha256": candidate,
            "workload": {"capsule_sha256": sentinel.capsule_sha256},
            "emission": emission,
            "diagnostics": {
                "arms": {"candidate": {"status": "emitted"}},
                "captured_logical_graph": {
                    "status": "verified",
                    "logical_dispatch_digest": graph_digest,
                    "source_sha256": source_digest,
                },
                "verified_global_plan_emission": {
                    "status": "verified",
                    "plan_digest": digest(f"plan-{iteration}-{index}"),
                    "candidate_sha256": candidate,
                    "logical_dispatch_digest": graph_digest,
                    "source_sha256": source_digest,
                    **emission,
                },
            },
        }
        identity = {
            "capsule": sentinel.capsule,
            "capsule_sha256": sentinel.capsule_sha256,
            "required_lanes": list(sentinel.required_lanes),
            "required_tiers": list(sentinel.required_tiers),
            "role": "primary" if index == 0 else "training",
            "analysis": "full_graph_compile_and_static_only",
            "full_model_simulation_allowed": False,
        }
        members.append({"identity": identity, "analysis": analysis})
        portfolio[sentinel.capsule_sha256] = {
            "candidate_sha256": candidate,
            **emission,
            "interface": interface,
            "lowered_text": lowered,
            "command_buffer_text": command,
            "command_buffer": json.loads(command),
        }
    row = {
        "iteration": iteration,
        "candidate_sha256": candidate,
        "analysis": members[0]["analysis"],
        "portfolio": {"members": members},
        "readiness": {"status": "ready_for_probe_admission"},
        "compiler_dependencies": {"compiler_implementation_sha256": digest(f"compiler-{iteration}")},
    }
    return row, portfolio[sentinels[0].capsule_sha256], portfolio


def context(journal, row, sentinels, *, index=1, arm="current"):
    arguments = {"index": index, "sentinels": sentinels, "target_sha256": digest("target")}
    if arm == "previous":
        return journal.previous_portfolio_member_context(**arguments)
    return journal.current_portfolio_member_context(row, **arguments)


@pytest.mark.parametrize("field", ["candidate_sha256", "candidate_lowered_sha256"])
@pytest.mark.parametrize("arm", ["current", "previous"])
def test_primary_identity_mismatch_is_refused(evidence, field, arm):
    journal, sentinels = evidence
    first, primary, portfolio = revision(sentinels, 0)
    journal.publish_analysis(first, primary, portfolio)
    row, current, portfolio = revision(sentinels, 1)
    journal.publish_analysis(row, current, portfolio)
    selected = current if arm == "current" else primary
    selected[field] = digest("changed")
    with pytest.raises(ValueError):
        if arm == "current":
            journal.current_artifacts(row)
        else:
            journal.retained_previous_artifacts()


@pytest.mark.parametrize("mutation", ["missing", "order", "candidate", "lowered"])
def test_current_portfolio_requires_ordered_complete_identity(evidence, mutation):
    journal, sentinels = evidence
    row, primary, portfolio = revision(sentinels, 0)
    journal.publish_analysis(row, primary, portfolio)
    assert journal.current_portfolio_artifacts(row, sentinels=sentinels) is portfolio
    secondary = sentinels[1].capsule_sha256
    if mutation == "missing":
        portfolio.pop(secondary)
    elif mutation == "order":
        entries = list(portfolio.items())
        portfolio.clear()
        portfolio.update(reversed(entries))
    else:
        field = "candidate_sha256" if mutation == "candidate" else "candidate_lowered_sha256"
        portfolio[secondary][field] = digest("changed")
    with pytest.raises(ValueError):
        journal.current_portfolio_artifacts(row, sentinels=sentinels)


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "identity",
        "source",
        "plan",
        "graph",
        "emission",
        "lowered",
        "command_hash",
        "command_json",
        "malformed_json",
        "compiler",
        "readiness",
    ],
)
@pytest.mark.parametrize("arm", ["current", "previous"])
def test_ordered_member_context_refuses_changed_bindings(evidence, mutation, arm):
    journal, sentinels = evidence
    row, primary, portfolio = revision(sentinels, 0)
    journal.publish_analysis(row, primary, portfolio)
    if arm == "previous":
        journal.publish_analysis(*revision(sentinels, 1))
    assert isinstance(context(journal, row, sentinels, arm=arm)["probe_binding"], ProbeBinding)
    artifact = portfolio[sentinels[1].capsule_sha256]
    analysis = row["portfolio"]["members"][1]["analysis"]
    plan = analysis["diagnostics"]["verified_global_plan_emission"]
    if mutation == "order":
        items = list(portfolio.items())
        portfolio.clear()
        portfolio.update(reversed(items))
    elif mutation == "identity":
        row["portfolio"]["members"][1]["identity"]["capsule_sha256"] = digest("changed")
    elif mutation == "source":
        artifact["interface"].write_text("changed source")
    elif mutation == "plan":
        plan["candidate_sha256"] = digest("changed")
    elif mutation == "graph":
        plan["logical_dispatch_digest"] = digest("changed")
    elif mutation == "emission":
        analysis["emission"]["candidate_lowered_sha256"] = digest("changed")
    elif mutation == "lowered":
        artifact["lowered_text"] += " changed"
    elif mutation == "command_hash":
        artifact["candidate_command_buffer_sha256"] = digest("changed")
    elif mutation == "command_json":
        artifact["command_buffer"] = {"unbound": True}
    elif mutation == "malformed_json":
        artifact["command_buffer_text"] = "{"
    elif mutation == "compiler":
        row["compiler_dependencies"]["compiler_implementation_sha256"] = "unknown"
    elif mutation == "readiness":
        analysis["diagnostics"]["arms"]["candidate"]["status"] = "declined"
    with pytest.raises(ValueError):
        context(journal, row, sentinels, arm=arm)


@pytest.mark.parametrize("linked", [False, True])
def test_member_context_refuses_escaped_or_linked_interface(evidence, tmp_path, linked):
    journal, sentinels = evidence
    row, primary, portfolio = revision(sentinels, 0)
    artifact = portfolio[sentinels[1].capsule_sha256]
    original = artifact["interface"]
    outside = tmp_path / "outside.mlir"
    outside.write_bytes(original.read_bytes())
    if linked:
        link = original.with_name("linked.mlir")
        link.symlink_to(original)
        artifact["interface"] = link
    else:
        artifact["interface"] = outside
    journal.publish_analysis(row, primary, portfolio)
    with pytest.raises(ValueError, match="source"):
        context(journal, row, sentinels)


def test_reuse_and_import_keep_originating_member_bindings(evidence):
    journal, sentinels = evidence
    first, primary, portfolio = revision(sentinels, 0)
    journal.publish_imported_seed(first, primary, portfolio, reconstructed_sandboxes=None, sandbox_digest=None)
    origin = context(journal, first, sentinels)
    second, new_primary, new_portfolio = revision(sentinels, 1)
    journal.publish_analysis(second, new_primary, new_portfolio)
    previous = context(journal, second, sentinels, arm="previous")
    assert previous["probe_binding"] == origin["probe_binding"]
    assert previous["member_binding"]["candidate_sha256"] == first["candidate_sha256"]
    reused = copy.deepcopy(first)
    reused["iteration"] = 2
    journal.publish_reuse(reused, copy.deepcopy(primary), source_iteration=0)
    assert context(journal, reused, sentinels)["probe_binding"] == origin["probe_binding"]
    assert (
        context(journal, reused, sentinels, arm="previous")["member_binding"]["candidate_sha256"]
        == second["candidate_sha256"]
    )
    assert journal.portfolio_artifacts == portfolio
    assert journal.portfolio_artifacts is not portfolio


@pytest.fixture
def baseline_evidence(evidence):
    journal, sentinels = evidence
    row, primary, portfolio = revision(sentinels, 0)
    command = '{"commands": [], "tensors": {}}'
    baseline = {
        "identity": {
            "baseline_sha256": digest("baseline"),
            "capsule_sha256": sentinels[0].capsule_sha256,
            "target": "synthetic",
        },
        "lowered_text": "baseline lowered",
        "lowered_sha256": digest("baseline lowered"),
        "command_buffer_text": command,
        "command_buffer_sha256": digest(command),
        "command_buffer": {"unbound_cached_object": True},
    }
    primary["baseline_artifacts"] = baseline
    row["analysis"]["emission"].update(
        baseline_lowered_sha256=baseline["lowered_sha256"],
        baseline_command_buffer_sha256=digest(command),
    )
    journal.publish_analysis(row, primary, portfolio)
    return journal, sentinels[0], row, baseline


def baseline_context(journal, sentinel, row):
    retained = journal.baseline_artifact_bytes(
        row,
        baseline_sha256=digest("baseline"),
        sentinel=sentinel,
        target="synthetic",
    )
    return journal.optimization_baseline_artifacts(
        row,
        retained,
        journal.current_artifacts(row),
        baseline_sha256=digest("baseline"),
        sentinel=sentinel,
    )


def test_baseline_decodes_checked_bytes_without_qualifying_them(baseline_evidence):
    journal, sentinel, row, baseline = baseline_evidence
    artifact = baseline_context(journal, sentinel, row)
    binding = journal.optimization_baseline_artifact_binding(
        row,
        artifact,
        baseline_sha256=digest("baseline"),
        baseline_dependencies={"synthetic": True},
        baseline_binding_sha256=digest("baseline binding"),
        sentinel=sentinel,
        target_sha256=digest("target"),
    )
    assert artifact["command_buffer"] == json.loads(baseline["command_buffer_text"])
    assert artifact["command_buffer"] != baseline["command_buffer"]
    for result in (artifact, binding):
        assert result["structural_plan_status"] == "UNVERIFIED"
        assert result["numerical_qualification"] == "UNPROVEN"
        assert not isinstance(result, ProbeBinding)
        assert "plan_digest" not in result
    assert binding["phase1_qualification_extended"] is False
    with pytest.raises(ValueError, match="two submitted"):
        journal.retained_previous_artifacts()


@pytest.mark.parametrize("mutation", ["identity", "lowered", "command", "analyzed_hash", "source", "declined"])
def test_baseline_refuses_changed_bytes_and_unusable_buffer(baseline_evidence, mutation):
    journal, sentinel, row, baseline = baseline_evidence
    if mutation == "identity":
        baseline["identity"]["baseline_sha256"] = digest("another compiler")
    elif mutation == "lowered":
        baseline["lowered_text"] += " changed"
    elif mutation == "command":
        baseline["command_buffer_text"] += " "
    elif mutation == "analyzed_hash":
        row["analysis"]["emission"]["baseline_lowered_sha256"] = digest("changed")
    elif mutation == "source":
        journal.artifacts["interface"].write_text("changed source")
    elif mutation == "declined":
        command = '{"declined": "synthetic refusal"}'
        baseline["command_buffer_text"] = command
        baseline["command_buffer_sha256"] = digest(command)
        row["analysis"]["emission"]["baseline_command_buffer_sha256"] = digest(command)
    with pytest.raises(ValueError):
        baseline_context(journal, sentinel, row)
