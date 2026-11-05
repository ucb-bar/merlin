"""Absent declarations retain an effective deny under original mount replay."""

import importlib
from types import SimpleNamespace

import pytest

from merlin.targetgen.sandbox import bwrap as BW

AS = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")


@pytest.fixture
def selected(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(AS, "artifacts_dir", lambda: artifacts)
    monkeypatch.setattr(AS, "prior_backend_exemptions", lambda _te: ())
    monkeypatch.setattr(AS, "_evicted_oracle_modules_result", lambda: ([], None))
    target = SimpleNamespace(target="selected_unit", prior_backends=("historical",))
    namespace = artifacts / "targets" / target.target
    return target, namespace


def test_absence_is_explicit_and_emits_no_pointless_mount(selected):
    target, namespace = selected
    surfaces = AS.prior_backend_surfaces(target)
    assert [s.path for s in surfaces] == [namespace]
    assert AS.dropped_declarations(target) == []
    assert AS.absent_declaration_coverage(target) == [
        {
            "origin": "prior_backend",
            "declared": "historical",
            "path": str(namespace / "historical"),
            "namespace": str(namespace),
            "status": "absent_with_retained_namespace_deny",
        }
    ]
    assert BW.apply_answer_masks([], surfaces) == []
    assert BW.coverage_gap([], surfaces) == []


@pytest.mark.parametrize("where", ["namespace", "child", "alias"])
def test_later_bound_bytes_are_masked_even_with_previously_derived_surfaces(selected, tmp_path, where):
    target, namespace = selected
    surfaces = AS.prior_backend_surfaces(target)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "answer.py").write_text("private reference implementation")
    destination = {"namespace": namespace, "child": namespace / "new_unlisted", "alias": tmp_path / "alias"}[where]
    if where == "alias":
        # Original projection follows a declared private source to another destination.
        namespace.mkdir(parents=True)
        source = namespace / "historical"
        source.mkdir()
        (source / "answer.py").write_text("private reference implementation")
    else:
        source = payload
    argv = ["--ro-bind", str(source), str(destination)]
    assert BW.coverage_gap(argv, surfaces)
    masked = BW.apply_answer_masks(argv, surfaces)
    assert BW.coverage_gap(masked, surfaces) == []
    assert not BW.is_exposed(masked, destination / "answer.py")


def test_late_directory_creation_uses_the_same_fresh_mount_replay(selected):
    target, namespace = selected
    surfaces = AS.prior_backend_surfaces(target)
    assert BW.apply_answer_masks([], surfaces) == []
    package = namespace / "historical"
    package.mkdir(parents=True)
    (package / "answer.py").write_text("private")
    assert AS.absent_declaration_coverage(target) == []
    assert AS.dropped_declarations(target) == []
    argv = ["--ro-bind", str(namespace.parent), str(namespace.parent)]
    assert BW.coverage_gap(argv, surfaces)
    assert BW.coverage_gap(BW.apply_answer_masks(argv, surfaces), surfaces) == []
    assert [s.path for s in AS.prior_backend_surfaces(target)] == [namespace, package]


def test_absent_exemption_and_discovery_failures_still_refuse(selected, monkeypatch):
    target, _ = selected
    monkeypatch.setattr(AS, "prior_backend_exemptions", lambda _te: ("missing_public_exception",))
    monkeypatch.setattr(AS, "_evicted_oracle_modules_result", lambda: ([], "ImportError: registry unavailable"))
    drops = AS.dropped_declarations(target)
    assert {d.origin for d in drops} == {"prior_backend_exemption", "evicted_oracle"}
    with pytest.raises(AS.DroppedDeclarations):
        AS.require_declarations_in_force(target)


@pytest.mark.parametrize("name", ["../foreign", "/foreign"])
def test_invalid_or_uncovered_names_do_not_receive_absence_coverage(selected, name):
    target, _ = selected
    target.prior_backends = (name,)
    assert AS.absent_declaration_coverage(target) == []
    assert AS.dropped_declarations(target)[0].origin == "prior_backend"


def test_dangling_alias_does_not_receive_absence_coverage(selected, tmp_path):
    target, namespace = selected
    namespace.mkdir(parents=True)
    (namespace / "historical").symlink_to(tmp_path / "missing")
    assert AS.absent_declaration_coverage(target) == []
    assert AS.dropped_declarations(target)[0].origin == "prior_backend"


def test_missing_logical_deny_cannot_issue_absence_coverage(selected, monkeypatch):
    target, _ = selected
    monkeypatch.setattr(AS, "prior_backend_surfaces", lambda _te: [])
    assert AS.absent_declaration_coverage(target) == []
    assert AS.dropped_declarations(target)[0].origin == "prior_backend"
