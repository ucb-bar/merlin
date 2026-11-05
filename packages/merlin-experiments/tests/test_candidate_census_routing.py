"""Fixed original census routing; synthetic observations qualify no engine."""

import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import candidate_census as C
from merlin_experiments.phase2 import ordinary_fast_feedback as F
from merlin_experiments.phase2.contracts import exact_tree_record
from test_census_launch import original as original

from merlin.benchharness import hash_tree


def test_actual_fixed_wrapper_import_and_original_link_producer_join(original, monkeypatch):
    from merlin.perf import dynamic_census as core

    calls = []
    def issue(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(closed=True, _issuer=object())
    monkeypatch.setattr(core, "_issue_launch", issue)
    monkeypatch.setattr(core, "profile_elf", lambda *args, **kwargs: core.unknown("synthetic joined route"))
    result = C.observe_passed_census(
        original.workspace, target="synthetic", compiler=original.package, policy=original.policy,
        remaining_s=10, numerical_pass=True,
    )
    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "synthetic joined route"
    assert len(calls) == 1
    assert calls[0]["elf"] == original.elf
    assert str(original.package) not in calls[0]["prefix"]
    assert str(original.workspace) not in calls[0]["prefix"]
    calls.clear()
    original.elf.write_bytes(b"changed original linked product")
    refused = C.observe_passed_census(
        original.workspace, target="synthetic", compiler=original.package, policy=original.policy,
        remaining_s=10, numerical_pass=True,
    )
    assert refused["status"] == "UNKNOWN" and not calls


def install_wrapper(monkeypatch, calls):
    def observe(elf, **kwargs):
        calls.append((elf, kwargs))
        return {"schema": "synthetic_dynamic_census", "status": "observed", "timing_cycles": None}
    monkeypatch.setitem(sys.modules, "merlin_experiments.phase2.census_launch",
                        SimpleNamespace(observe_dynamic_census=observe))


def test_failed_numeric_unknown_owner_and_ambiguous_images_execute_nothing(tmp_path, monkeypatch):
    calls = []
    install_wrapper(monkeypatch, calls)
    tmp_path.joinpath("package_kernel.elf").write_bytes(b"own image")
    assert C.observe_passed_census(tmp_path, target="selected", compiler=tmp_path,
                                   remaining_s=10, numerical_pass=False)["status"] == "UNKNOWN"
    assert C.observe_passed_census(tmp_path, target="selected", compiler=tmp_path,
                                   remaining_s=10, numerical_pass=True)["status"] == "UNKNOWN"
    other = tmp_path / "other"
    other.mkdir()
    other.joinpath("package_kernel.elf").write_bytes(b"second image")
    assert C.observe_passed_census(tmp_path, target="selected", compiler=tmp_path, policy=object(),
                                   remaining_s=10, numerical_pass=True)["status"] == "UNKNOWN"
    assert not calls


def test_selected_policy_and_exact_image_reach_only_fixed_boxed_wrapper(tmp_path, monkeypatch):
    calls = []
    install_wrapper(monkeypatch, calls)
    image = tmp_path / "package_kernel.elf"
    image.write_bytes(b"own image")
    policy = object()  # Lower boundary fixture; real wrapper separately requires exact live original policy.
    report = C.observe_passed_census(tmp_path, target="selected", compiler=tmp_path, policy=policy,
                                      remaining_s=7, numerical_pass=True)
    assert report["status"] == "observed"
    assert calls == [(image, {"target": "selected", "directory": tmp_path / "dynamic_census",
                             "remaining_s": 7, "policy": policy})]


def test_unresolved_real_cleanup_cannot_be_hidden_as_optional_unknown(tmp_path, monkeypatch):
    class CleanupUnknown(BaseException):
        pass
    def unknown_cleanup(*args, **kwargs):
        raise CleanupUnknown("synthetic retained live owner")
    monkeypatch.setitem(sys.modules, "merlin_experiments.phase2.census_launch",
                        SimpleNamespace(observe_dynamic_census=unknown_cleanup))
    tmp_path.joinpath("package_kernel.elf").write_bytes(b"own image")
    with pytest.raises(CleanupUnknown):
        C.observe_passed_census(tmp_path, target="selected", compiler=tmp_path, policy=object(),
                                remaining_s=10, numerical_pass=True)


@pytest.mark.parametrize("pass_l2", [False, True])
def test_normal_fast_uses_same_package_scope_and_original_numerical_result(tmp_path, monkeypatch, pass_l2):
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    compiler.joinpath("own.py").write_text("own compiler")
    source = tmp_path / "source"
    source.mkdir()
    source.joinpath("source.mlir").write_text("public source")
    revision = SimpleNamespace(source=compiler, source_sha256=str(hash_tree(compiler)["sha256"]))
    member = SimpleNamespace(source_dir=source, source_sha256=exact_tree_record(source)["sha256"])
    facts = tmp_path / "selected-facts.json"
    facts.write_text("{}")
    policy = SimpleNamespace(
        target_experiment=SimpleNamespace(target="selected"), _contract_root=tmp_path,
        feedback_evaluator=SimpleNamespace(rtl_identity={}, rtl_facts_path=facts),
    )
    selected = object()
    scopes, census = [], []
    monkeypatch.setattr(F.PC, "package_sandbox_policy", lambda *args: selected)
    def scope(actual):
        scopes.append(actual)
        return nullcontext()
    monkeypatch.setattr(F.PC, "boxed_entrypoints", scope)
    monkeypatch.setattr(F.CR, "load_capsule", lambda *args, **kw: {"required_oracle_tiers": ["L2", "L3"]})
    monkeypatch.setattr(F.CR, "simulator_adapter", lambda *args: "original_selected_spike")
    monkeypatch.setattr(F.CR, "run_capsule", lambda *args, **kw: {
        "tiers": {tier: {"status": "pass" if tier != "L2" or pass_l2 else "fail"} for tier in ("L0", "L1", "L2")}})
    monkeypatch.setattr(F, "compiled_observation", lambda *args, **kw: {"status": "UNKNOWN"})
    def observe(root, **kwargs):
        census.append(kwargs)
        return C.unknown("synthetic no execution")
    monkeypatch.setattr(F, "observe_passed_census", observe)
    report = F._member_arm(policy=policy, revision=revision, member=member, workspace=tmp_path / "job",
                           deadline=time.monotonic() + 10, workers=1)
    assert scopes == [selected]
    assert census[0]["policy"] is selected
    assert census[0]["numerical_pass"] is pass_l2
    assert report["functional_status"] == ("PASS" if pass_l2 else "UNKNOWN_OR_FAILURE")
    assert report["compiled"]["dynamic_census"]["status"] == "UNKNOWN"
    assert report["pending_tiers"] == ["L3"]
