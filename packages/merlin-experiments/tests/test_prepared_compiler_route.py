"""Prepared ordinary routing under original gates; measurements here are synthetic."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import compiler_pass_preparation as P
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import paired_inputs as PI
from merlin_experiments.phase2 import paired_measurement as PM
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
from test_compiler_library_state import frozen
from test_compiler_pass_preparation import OriginalPreparation
from test_ordinary_search_policy import ordinary_inputs  # noqa: F401
from test_paired_measurement import _plan

from merlin.benchharness import hash_tree
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.perf.phase2_edit_contract import seal
from merlin.targetgen.compiler_library import active_compiler_library
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, active_compiler_passes


@pytest.fixture
def prepared(request, tmp_path):
    f = request.getfixturevalue("ordinary_inputs")
    host, binding, snapshot = frozen(tmp_path)
    evidence = tmp_path / "independent-preparation-evidence"
    evidence.write_text("neutral preparation only; no target or timing qualification\n")
    owner = OriginalPreparation(evidence)
    selected = PassCapabilityBinding("facts", "effects", "exact-i32", "fixture", host.contract.sha256)
    registry = ReviewedPassCapabilities((ReviewedPassCapability("composite", selected, owner),))
    passes = CompilerPassSelection(registry, selected, mechanisms=("composite",))
    choice = P.CompilerPassPreparation(
        "composite", baseline_policy={"enabled": False}, candidate_policy={"enabled": True}
    )
    record = P.prepare_pair(
        passes,
        choice,
        original=f.arguments["baseline"],
        functional_sha256=f.functional.digest,
        root=tmp_path / "prepared",
    )
    candidate = AW.fresh_round_workspace(
        Path(record["candidate_seed"]["path"]), tmp_path / "author", record["candidate_seed"]["sha256"]
    )
    authority_root = tmp_path / "prepared-authority"
    authority_root.mkdir()
    authority = FrozenEditAuthority(authority_root)
    authority.freeze(
        candidate,
        seal(
            dict(
                schema="compiler_edit_contract_v1",
                helper_extensions=[],
                existing_symbols=[dict(surface_id="policy", path="candidate_policy.py", symbol="choose")],
            )
        ),
        has_iterations=False,
    )
    arguments = {
        **f.arguments,
        "candidate": candidate,
        "baseline": Path(record["baseline"]["path"]),
        "edit_authority": authority,
        "compiler_pass_selection": passes,
        "compiler_pass_preparation": record,
        "compiler_library_selection": host,
    }
    return SimpleNamespace(
        f=f,
        record=record,
        passes=passes,
        choice=choice,
        owner=owner,
        arguments=arguments,
        library=host,
        library_binding=binding,
        library_snapshot=snapshot,
    )


def test_actual_prepared_seed_is_frozen_before_policy_edit_and_retained_feedback(prepared, tmp_path, monkeypatch):
    p = prepared
    original_execute = PM.run_execution
    seen = []

    def execute(spec, *args, **kwargs):
        assert active_compiler_passes() is p.passes
        assert active_compiler_library() is p.library
        seen.append(spec.arm)
        return original_execute(spec, *args, **kwargs)

    monkeypatch.setattr(PM, "run_execution", execute)
    policy = OS.prepare_ordinary_search_policy(**p.arguments)
    binding = policy.binding()
    assert binding["functional_origin"]["digest"] == p.f.functional.digest
    assert binding["baseline"]["sha256"] == p.record["baseline"]["sha256"] != p.f.functional.digest
    assert binding["compiler_pass_preparation"] == p.record
    assert policy.edit_authority.binding["initial_candidate_sha256"] == p.record["candidate_seed"]["sha256"]
    store = RetainedRevisionStore(
        tmp_path / "retained",
        edit_authority=policy.edit_authority,
        max_bytes=65536,
        max_files=32,
        max_total_bytes=262144,
    )
    baseline = store.freeze(policy.search_baseline)
    (policy.candidate / "candidate_policy.py").write_text("def choose(shape=(1,)):\n    return shape[0] > 7\n")
    candidate = store.freeze(policy.candidate, parent_revision=baseline.revision_id)
    result = policy.evaluate_retained(candidate, baseline, work_root=tmp_path / "job", call_index=0, timeout_s=10)
    assert result["summary"]["all_correct"]
    assert sorted(seen) == ["baseline", "candidate"]
    assert hash_tree(p.f.arguments["baseline"])["sha256"] == p.f.functional.digest
    assert not (p.f.arguments["baseline"] / "candidate_policy.py").exists()
    assert p.f.admission[0][2] == p.f.functional.digest
    assert active_compiler_passes() is None


def test_preparation_does_not_widen_existing_authority_or_remove_original_qualification(prepared, monkeypatch):
    p = prepared
    with pytest.raises(OS.StageGateError, match="edit authority"):
        OS.prepare_ordinary_search_policy(**{**p.arguments, "edit_authority": p.f.arguments["edit_authority"]})
    monkeypatch.setattr(
        OS.FI,
        "require_formal_qualification",
        lambda *_: (_ for _ in ()).throw(OS.StageGateError("original qualification pending")),
    )
    with pytest.raises(OS.StageGateError, match="original qualification pending"):
        OS.prepare_ordinary_search_policy(**p.arguments)
    assert not p.f.executions


def test_pass_selection_without_original_preparation_refuses_before_execution(prepared):
    p = prepared
    with pytest.raises(OS.StageGateError, match="pre-freeze preparation"):
        OS.prepare_ordinary_search_policy(**{**p.f.arguments, "compiler_pass_selection": p.passes})
    assert not p.f.executions


@pytest.mark.parametrize("width", [1, 2])
def test_paired_workers_keep_original_pass_policy_and_same_release(prepared, tmp_path, width):
    p = prepared
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    plan = _plan(plan_root)
    declaration = {
        **plan.declaration,
        "schema": "paired_arm4_measurement_plan_v5",
        "additional_compiler_library": p.library_binding,
        "compiler_pass_preparation": p.record,
    }
    plan = replace(
        plan,
        declaration=declaration,
        declaration_sha256=PM.sha256_bytes(PM.canonical_bytes(declaration)),
        compiler_library_binding=p.library_binding,
        compiler_library_selection=p.library_snapshot,
        compiler_library_host_selection=p.library,
        compiler_pass_preparation=p.record,
        compiler_pass_selection=p.passes,
        compiler_pass_preparation_selection=p.choice,
    )
    out = tmp_path / "results"
    out.mkdir()
    seen = []

    def executor(spec, *args, **kwargs):
        assert active_compiler_passes() is p.passes
        assert active_compiler_library() is p.library_snapshot
        seen.append(spec.arm)
        return {"execution": spec.as_dict(), "measurement": {"per_sim": {}}}

    PM.execute_schedule(
        plan,
        out,
        contract_root=tmp_path,
        timeout=1,
        target_experiment=None,
        rtl_identity={},
        hardware_counters=False,
        executor=executor,
        progress=lambda _: None,
        fanout={"effective": width},
    )
    assert sorted(seen) == ["baseline", "baseline", "candidate", "candidate"]
    for item in json.loads((out / "raw_results.index.json").read_bytes()):
        assert json.loads((out / item["path"]).read_bytes())["compiler_pass_preparation_sha256"] == p.record["sha256"]
    assert active_compiler_passes() is None
    changed = copy.deepcopy(p.record)
    changed["preparation"]["baseline_policy"]["enabled"] = True
    changed["sha256"] = P._sha({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(P.StageGateError, match="original host policy"):
        P.handoff_selection(replace(plan, compiler_pass_preparation=changed))


def test_original_fork_accessor_never_calls_prepared_bytes_original(prepared):
    p = prepared
    inputs = SimpleNamespace(
        functional=p.f.functional,
        baseline=p.arguments["baseline"],
        handoff=SimpleNamespace(functional_base_path=p.f.arguments["baseline"], compiler_pass_preparation=p.record),
    )
    assert PI.original_baseline(inputs) == p.f.arguments["baseline"] != inputs.baseline
