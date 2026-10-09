"""Mutation checks for minimal views, isolation refusal and final comparison arithmetic."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_experiment as C
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen.compiler_library import freeze_compiler_library


def _view(tmp_path):
    root = tmp_path / "installed"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text("def lower(value):\n    return value\n")
    (root / "private-answer.txt").write_text("MUST_NOT_COPY")
    library = freeze_compiler_library(
        root,
        review_id="review",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    source = tmp_path / "generated.mlir"
    source.write_text("module {}\n")
    member = C.ApprovedInput(
        source, "generated_input/case.mlir", hashlib.sha256(source.read_bytes()).hexdigest(), "generated_input"
    )
    return C.materialize_component_view(
        tmp_path / "agent-view", library=library, library_root=root, inputs=(member,), generation_sha256="1" * 64
    )


def test_view_copies_only_explicit_reviewed_members_without_source_paths(tmp_path):
    view = _view(tmp_path)
    record = C.verify_component_view(view)
    assert len(record["members"]) == 3
    payload = (view.root / "manifest.json").read_text()
    assert str(tmp_path) not in payload
    assert not (view.root / "private-answer.txt").exists()


@pytest.mark.parametrize("writable", [True, False])
def test_explicit_readonly_execution_preserves_the_author_namespace(tmp_path, writable):
    view = _view(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    tool = tmp_path / "tool"
    tool.write_text("owned inventory member, not an execution proof")
    runtime = (C.RuntimeGrant(tool, "/usr/bin/tool", hashlib.sha256(tool.read_bytes()).hexdigest()),)
    options = dict(runtime=runtime, bwrap_binary=tmp_path / "selected-outer")
    writable_policy = C.strict_tool_policy(view, candidate, **options)
    policy = C.strict_tool_policy(view, candidate, candidate_writable=writable, **options)
    slot = writable_policy.index("--bind")
    assert policy[:slot] == writable_policy[:slot]
    assert policy[slot] == ("--bind" if writable else "--ro-bind")
    assert policy[slot + 1 :] == writable_policy[slot + 1 :]
    assert "--share-net" not in policy and "--unshare-all" in policy


@pytest.mark.parametrize("writable", [None, 0, 1, "false", {}])
def test_saved_or_coerced_write_selections_do_not_grant_a_namespace(tmp_path, writable):
    with pytest.raises(StageGateError, match="explicit bool"):
        C.strict_tool_policy(None, tmp_path, runtime=(), candidate_writable=writable)


@pytest.mark.parametrize("selection", [None, 0, 1, "false", {}])
def test_process_filesystem_selection_is_not_coerced(tmp_path, selection):
    with pytest.raises(StageGateError, match="explicit bool"):
        C.strict_tool_policy(None, tmp_path, runtime=(), mount_proc=selection)


def test_view_refreezes_exact_approved_library_selection(tmp_path, monkeypatch):
    root = tmp_path / "installed"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text("def lower(value):\n    return value\n")
    sources = (("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable"))
    library = freeze_compiler_library(root, review_id="review", public_modules=("merlin.portable",), sources=sources)
    generated = tmp_path / "generated.mlir"
    generated.write_text("module {}\n")
    member = C.ApprovedInput(
        generated, "generated_input/case.mlir", hashlib.sha256(generated.read_bytes()).hexdigest(), "generated_input"
    )
    selected = []

    def checked_freeze(selected_root, *, review_id, public_modules, sources):
        selected.append((selected_root, review_id, public_modules, sources))
        return freeze_compiler_library(
            selected_root, review_id=review_id, public_modules=public_modules, sources=sources
        )

    monkeypatch.setattr(C, "freeze_compiler_library", checked_freeze, raising=False)
    view = C.materialize_component_view(
        tmp_path / "agent-view", library=library, library_root=root, inputs=(member,), generation_sha256="1" * 64
    )
    assert C.verify_component_view(view)["library_sha256"] == library.sha256
    assert selected == [(root, "review", ("merlin.portable",), sources)] * 2


def test_view_refuses_refrozen_library_identity_drift(tmp_path, monkeypatch):
    root = tmp_path / "installed"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text("def lower(value):\n    return value\n")
    sources = (("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable"))
    library = freeze_compiler_library(root, review_id="review", public_modules=("merlin.portable",), sources=sources)
    generated = tmp_path / "generated.mlir"
    generated.write_text("module {}\n")
    member = C.ApprovedInput(
        generated, "generated_input/case.mlir", hashlib.sha256(generated.read_bytes()).hexdigest(), "generated_input"
    )

    def changed_identity(selected_root, *, review_id, public_modules, sources):
        return freeze_compiler_library(
            selected_root, review_id="different-review", public_modules=public_modules, sources=sources
        )

    monkeypatch.setattr(C, "freeze_compiler_library", changed_identity, raising=False)
    with pytest.raises(StageGateError, match="library identity"):
        C.materialize_component_view(
            tmp_path / "agent-view", library=library, library_root=root, inputs=(member,), generation_sha256="1" * 64
        )
    assert not (tmp_path / "agent-view").exists()


@pytest.mark.parametrize("mutation", ["history", "bytes", "link", "empty-directory"])
def test_added_history_or_changed_member_invalidates_view(tmp_path, mutation):
    view = _view(tmp_path)
    source = view.root / "generated_input/case.mlir"
    if mutation == "history":
        (view.root / ".git").mkdir()
        (view.root / ".git/config").write_text("remote private-repository")
    elif mutation == "bytes":
        source.chmod(0o644)
        source.write_text("changed")
    elif mutation == "link":
        source.unlink()
        source.symlink_to(tmp_path / "generated.mlir")
    else:
        (view.root / "unreviewed").mkdir()
    with pytest.raises(StageGateError):
        C.verify_component_view(view)


def test_strict_policy_does_not_inherit_network_home_or_entire_checkout(tmp_path):
    view = _view(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    tool = tmp_path / "tool"
    tool.write_bytes(b"synthetic-runtime")
    grant = C.RuntimeGrant(tool, "/usr/bin/tool", hashlib.sha256(tool.read_bytes()).hexdigest())
    argv = C.strict_tool_policy(view, candidate, runtime=(grant,))
    assert "--unshare-all" in argv and "--clearenv" in argv
    assert "--share-net" not in argv
    assert str(tmp_path / "installed") not in argv
    assert not any(value in argv for value in ("/home", "/scratch", ".git"))
    with pytest.raises(StageGateError, match="overlaps"):
        C.strict_tool_policy(view, view.root, runtime=(grant,))


def test_unavailable_namespace_probe_is_not_pass(monkeypatch):
    monkeypatch.setattr(
        C.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b"not permitted")
    )
    with pytest.raises(StageGateError, match="unavailable"):
        C.run_isolation_probe(("bwrap", "--unshare-all"), ("/usr/bin/true",))


def _comparisons():
    return tuple(
        C.FinalMemberComparison(name, 1000, 1050, "1" * 64, "2" * 64, "3" * 64, True, True, True)
        for name in ("heldout-a", "heldout-b", "heldout-c")
    )


def _gate(rows=None, **kw):
    values = {
        "expected_members": ("heldout-a", "heldout-b", "heldout-c"),
        "phase12_wall_s": 10,
        "handwritten_wall_s": 200,
    }
    values.update(kw)
    return C.final_component_campaign_gate(_comparisons() if rows is None else rows, **values)


def test_final_parity_is_per_member_and_uses_exact_boundary_arithmetic():
    assert _gate()["status"] == "pass"
    rows = list(_comparisons())
    rows[0] = replace(rows[0], candidate_cycles=1051)
    rows[1] = replace(rows[1], candidate_cycles=1)
    assert _gate(tuple(rows))["status"] == "fail"
    assert _gate(phase12_wall_s=10.01)["status"] == "fail"


def test_unknown_hardware_or_historical_time_cannot_pass():
    rows = list(_comparisons())
    rows[0] = replace(rows[0], hardware_verified=False)
    assert _gate(tuple(rows))["status"] == "unknown"
    assert _gate(handwritten_wall_s=None)["status"] == "unknown"
    rows[0] = replace(rows[0], accuracy_passed=False)
    assert _gate(tuple(rows))["status"] == "fail"


def test_final_membership_and_numeric_evidence_are_not_coerced():
    with pytest.raises(StageGateError, match="membership"):
        _gate(_comparisons()[:2])
    rows = list(_comparisons())
    rows[0] = replace(rows[0], candidate_cycles=True)
    with pytest.raises(StageGateError, match="integers"):
        _gate(tuple(rows))
    with pytest.raises(StageGateError, match="finite"):
        _gate(phase12_wall_s=float("nan"))
