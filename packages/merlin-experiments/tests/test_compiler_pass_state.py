"""Operator CLI custody controls; neutral factories qualify no target."""

import hashlib
import json
from pathlib import Path

import pytest
from merlin_experiments.phase2 import compiler_pass_state as S
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_libraries import selected as selected_library

from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, OriginalPassFactory


class NeutralOwner:
    def __init__(self, path):
        self.path = Path(path)
        self.identity = sha256_file(self.path)

    def revalidate(self):
        if sha256_file(self.path) != self.identity:
            raise ValueError("original neutral evidence changed")


def original_factory(inputs, *, work_root):
    evidence = Path(inputs["evidence"])
    if evidence.read_text() != "independent neutral check\n":
        raise ValueError("independent neutral check refused")
    work_root.mkdir(exist_ok=False)
    (work_root / "original-check-ran").write_text(sha256_file(evidence))
    return ReviewedPassCapabilities(
        (
            ReviewedPassCapability(
                "configuration_cleanup", PassCapabilityBinding(**inputs["binding"]), NeutralOwner(evidence)
            ),
        )
    )


@pytest.fixture
def original(tmp_path):
    from dataclasses import asdict

    library, _ = selected_library(tmp_path)
    evidence = tmp_path / "neutral-evidence"
    evidence.write_text("independent neutral check\n")
    binding = PassCapabilityBinding("hardware", "software", "integer", "functional", library.contract.sha256)
    factory = OriginalPassFactory(original_factory, {"evidence": str(evidence), "binding": asdict(binding)})
    selection = CompilerPassSelection(
        factory.create(work_root=tmp_path / "first-check"),
        binding,
        mechanisms=("configuration_cleanup",),
        factory=factory,
    )
    path = tmp_path / "operator-selection.json"
    path.write_text(json.dumps(selection.record()))
    return dict(
        path=path,
        sha256=sha256_file(path),
        work_root=tmp_path / "new-check",
        library=library,
        stage_root=tmp_path / "stage",
    ), selection


def test_explicit_operator_selection_reruns_original_factory(original):
    args, first = original
    live = S.load_selection(**args)
    assert live.record() == first.record()
    assert (args["work_root"] / "original-check-ran").is_file()
    assert not args["stage_root"].exists()


def test_absent_selection_preserves_legacy(original):
    args, _ = original
    assert S.load_selection(**{**args, "path": None, "sha256": None, "work_root": None, "library": None}) is None


@pytest.mark.parametrize(
    "damage",
    [
        "partial",
        "digest",
        "library",
        "wrong_release",
        "nested_root",
        "existing",
        "alias",
        "large",
        "evidence",
        "owner",
        "factory",
    ],
)
def test_incomplete_changed_or_saved_success_cannot_select_passes(original, damage):
    args, _ = original
    path = args["path"]
    if damage == "partial":
        args["work_root"] = None
    elif damage == "digest":
        args["sha256"] = "0" * 64
    elif damage == "library":
        args["library"] = None
    elif damage == "nested_root":
        args["work_root"] = args["stage_root"] / "inside"
    elif damage == "existing":
        args["work_root"].mkdir()
    elif damage == "alias":
        alias = path.with_name("alias.json")
        alias.symlink_to(path)
        args["path"] = alias
    elif damage == "large":
        path.write_bytes(b" " * ((2 << 20) + 1))
    else:
        record = json.loads(path.read_bytes())
        if damage == "wrong_release":
            record["binding"]["library"] = "other release"
        elif damage == "evidence":
            Path(record["factory"]["inputs"]["evidence"]).write_text("copied success\n")
        elif damage == "owner":
            record["registry"]["capabilities"][0]["owner_identity"] = "copied success"
        else:
            record["factory"] = None
        path.write_text(json.dumps(record))
        args["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(StageGateError):
        S.load_selection(**args)
    assert not args["stage_root"].exists()


@pytest.mark.parametrize("damage", ["stage_alias", "existing_stage", "library_child"])
def test_original_namespaces_are_checked_before_any_factory_side_effect(original, monkeypatch, damage):
    args, _ = original
    if damage == "stage_alias":
        real = args["stage_root"].with_name("canonical")
        real.mkdir()
        alias = real.with_name("alias")
        alias.symlink_to(real, target_is_directory=True)
        args["stage_root"] = alias / "new-stage"
        args["work_root"] = real / "new-stage" / "admission"
    elif damage == "existing_stage":
        args["stage_root"].mkdir()
    else:
        args["work_root"] = args["library"].root / "admission"

    def forbidden(*args, **kwargs):
        pytest.fail("invalid original namespace reached the admission callback")

    monkeypatch.setattr(S, "restore_original_selection", forbidden)
    with pytest.raises(StageGateError):
        S.load_selection(**args)
    assert not args["work_root"].exists()


def test_original_factory_runs_under_selected_resource_context(original, monkeypatch):
    from contextlib import contextmanager

    from merlin_experiments.phase2 import normal_resource_launch as NR

    args, first = original
    boundary, calls = object(), []

    @contextmanager
    def selected(value, *, writable_roots):
        assert value is boundary and args["work_root"] in writable_roots
        calls.append("enter")
        yield
        assert (args["work_root"] / "original-check-ran").is_file()
        calls.append("exit")

    monkeypatch.setattr(NR, "selected_boundary", selected)
    monkeypatch.setenv("TMPDIR", str(args["work_root"].parent))
    live = S.load_selection(**args, external_boundary=boundary)
    assert live.record() == first.record() and calls == ["enter", "exit"]
