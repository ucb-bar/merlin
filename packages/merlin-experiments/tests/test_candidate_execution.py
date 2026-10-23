"""Native operator selection, child propagation and drift refusal; no compiler launch."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import candidate_execution as E
from merlin_experiments.phase1 import context as C


@pytest.fixture
def selection(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import qualification_policy

    root, work = tmp_path / "frozen", tmp_path / "workers"
    root.mkdir()
    work.mkdir()
    binary = tmp_path / "sandbox"
    binary.write_bytes(b"explicit synthetic executable identity; never executed")
    policy = {"sha256": "a" * 64}
    observed = []

    def restore(owner, record):
        observed.append((owner, record))
        if owner != root or record != policy:
            raise ValueError("selected resource policy changed")
        return SimpleNamespace(record=policy)

    monkeypatch.setattr(qualification_policy, "restore", restore)
    record = dict(
        root=str(root),
        policy=policy,
        workspace_root=str(work),
        bwrap=dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(record))
    directories = (tmp_path / "runs", tmp_path / "reports", tmp_path / "bundles")
    context = C.InvocationContext(
        tmp_path, tmp_path / "target.yaml", tmp_path, "fixture", *directories, (), candidate_policy=path
    )
    return context, record, observed


def test_selection_reopens_resource_owner_and_binds_exact_operator_bytes(selection):
    context, record, observed = selection
    selected, identity = E.selection(context.candidate_policy)
    assert selected == record
    assert observed == [(Path(record["root"]), record["policy"])]
    assert identity["sha256"] == hashlib.sha256(context.candidate_policy.read_bytes()).hexdigest()
    E.verify_selection(context, identity)
    with pytest.raises(RuntimeError, match="selection changed or is absent"):
        E.verify_selection(replace(context, candidate_policy=None), identity)
    context.candidate_policy.write_text(json.dumps(record, indent=2))
    with pytest.raises(RuntimeError, match="selection changed or is absent"):
        E.verify_selection(context, identity)


@pytest.mark.parametrize("change", ["runtime", "binary", "extra", "alias", "parent_alias", "traversal"])
def test_selection_refuses_drift_and_indirect_owners(selection, change):
    context, record, _ = selection
    if change == "runtime":
        record["policy"] = {"sha256": "b" * 64}
    elif change == "binary":
        Path(record["bwrap"]["path"]).write_bytes(b"changed")
    elif change == "extra":
        record["compiler_import"] = "candidate.module"
    elif change == "alias":
        link = context.candidate_policy.parent / "alias.json"
        link.symlink_to(context.candidate_policy)
        context = replace(context, candidate_policy=link)
    elif change == "parent_alias":
        link = context.candidate_policy.parent / "alias-dir"
        link.symlink_to(Path(record["workspace_root"]), target_is_directory=True)
        record["workspace_root"] = str(link)
    else:
        context = replace(context, candidate_policy=context.candidate_policy.parent / "workers/../selection.json")
    if change not in {"alias", "traversal"}:
        context.candidate_policy.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        E.selection(context.candidate_policy)


def test_child_arguments_keep_operator_selection_and_oracle_binding_explicit(selection, monkeypatch):
    context, _, _ = selection
    transport = object()
    calls = []
    monkeypatch.setattr(E, "transport", lambda selected: calls.append(selected) or transport)
    assert C.oracle_kwargs(replace(context, candidate_policy=None)) == {}
    assert not calls
    assert C.oracle_kwargs(context) == {"candidate_transport": transport}
    assert calls == [context]
    args = C.context_argv(context)
    assert args[args.index("--candidate-policy") + 1] == str(context.candidate_policy)


def test_formal_cannot_enable_or_drop_transport_after_admission(selection, tmp_path):
    import yaml
    from merlin_experiments.phase1.feedback.formal import _verify_formal_readback_policy

    context, _, _ = selection
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(RuntimeError, match="selection changed or is absent"):
        _verify_formal_readback_policy(run, context)
    (run / "environment.yaml").write_text(yaml.safe_dump({"candidate_compilation": E.selection_record(context)}))
    _verify_formal_readback_policy(run, context)
    with pytest.raises(RuntimeError, match="selection changed or is absent"):
        _verify_formal_readback_policy(run, replace(context, candidate_policy=None))


def test_oracle_free_qa_and_formal_refuse_before_candidate_execution(selection):
    from merlin_experiments.phase1.feedback import formal, qa

    context, _, _ = selection
    with pytest.raises(ValueError, match="requires oracle evaluation"):
        qa.run("candidate", "corpus", Path("runs"), {"public"}, True, 1, context=context)
    with pytest.raises(ValueError, match="requires oracle evaluation"):
        formal._score("candidate", "corpus", "runs", {"hidden"}, True, context=context)


CHANGES = "unselected missing empty symlink malformed non_full_call tiers_missing L3 unknown no_L2".split()


@pytest.mark.parametrize("change", [None, *CHANGES])
def test_functional_codegen_scope_requires_verified_selection_and_both_complete_roots(selection, tmp_path, change):
    import yaml

    context, _, observed = selection
    public, hidden = tmp_path / "public", tmp_path / "hidden"
    capsule = {
        "operation": {"attributes": {"core_aten_contract": "full_call_v1"}},
        "required_oracle_tiers": ["L0", "L2"],
    }
    for root in (public, hidden):
        (root / "case").mkdir(parents=True)
        (root / "case/capsule.yaml").write_text(yaml.safe_dump(capsule))
    member = hidden / "case/capsule.yaml"
    if change == "unselected":
        context = replace(context, candidate_policy=None)
    elif change == "missing":
        hidden = None
    elif change == "empty":
        member.unlink()
    elif change == "symlink":
        member.unlink()
        member.symlink_to(public / "case/capsule.yaml")
    elif change == "malformed":
        member.write_text("operation: [")
    elif change == "non_full_call":
        capsule["operation"] = {"attributes": {}}
    elif change == "tiers_missing":
        del capsule["required_oracle_tiers"]
    elif change in {"L3", "unknown"}:
        capsule["required_oracle_tiers"] = ["L2", change]
    elif change == "no_L2":
        for path in (public / "case/capsule.yaml", member):
            path.write_text(yaml.safe_dump({**capsule, "required_oracle_tiers": ["L0"]}))
    if change in {"non_full_call", "tiers_missing", "L3", "unknown"}:
        member.write_text(yaml.safe_dump(capsule))
    scope = E.functional_codegen_scope(context, public_root=public, hidden_root=hidden)
    if change is not None:
        assert scope is None
    else:
        assert scope["status"] == "legacy_not_run"
        assert scope["selected_path"] == "source_product_full_call"
        assert scope["candidate_compilation"] == E.selection_record(context)
        assert scope["required_oracle_tiers"] == ["L0", "L2"]
        assert scope["n_capsules"] == 2
        assert len(scope["capsule_bytes_sha256"]) == 64
        assert str(hidden) not in json.dumps(scope)
    assert bool(observed) is (change != "unselected")
