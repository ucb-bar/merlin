"""Formal private source checks retain the reviewed public-semantics exception."""

import shutil
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import formal, private_source_freeze
from test_private_source_freeze import _fixture, _sha

from merlin.targetgen.sandbox import bwrap


def _prepared(tmp_path, monkeypatch, *, grant=True, mode="ro", note="admitted software semantics",
              reviewed=True, mask_host=True, mask_root=True):
    from merlin_experiments.corpus import numeric_policy

    from merlin.targetgen import software_spec, target_experiment, tool_profile

    spec, software, host, _ = _fixture(tmp_path)
    run = tmp_path / "run"
    root = run / "private_full_model_input/sources"
    record = private_source_freeze.stage(spec, root, target="fixture")
    copied_spec = run / "private_full_model_input/spec.yaml"
    shutil.copy2(spec, copied_spec)
    masks = []
    if mask_root:
        masks.append({"path": str(root), "kind": "dir"})
    if mask_host:
        masks.append({"path": str(host), "kind": "file"})
    effective = {
        "allowed": [{"path": str(software), "mode": mode, "note": note}] if grant else [],
        "host_inputs": [{"path": str(root)}],
        "private_validation_paths": masks,
    }
    effective_raw = yaml.safe_dump(effective).encode()
    (run / "input_bundle_manifest.yaml").write_bytes(effective_raw)
    private = {
        "source_sha256": _sha(spec.read_bytes()), "path": str(copied_spec),
        "sha256": _sha(copied_spec.read_bytes()), "frozen_path": str(copied_spec),
        "frozen_sha256": _sha(copied_spec.read_bytes()), "masked_paths": masks,
        "source_freeze": record,
    }
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bwrap.materialize_bundle_inputs(workspace, effective, repo=tmp_path)
    environment = {"private_full_model_spec": private,
                   "bundle_manifest_sha256": _sha(effective_raw),
                   "bundle_input_snapshot": bwrap.snapshot_record(workspace)}
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    selected = SimpleNamespace(target="fixture", numeric_profile="selected", _source_root=lambda: tmp_path)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _: selected)
    monkeypatch.setattr(tool_profile, "verify_tool_profile",
                        lambda _: {"common_substrate": tool_profile.COMMON_SUBSTRATE_V3})
    monkeypatch.setattr(numeric_policy, "numeric_profile_path", lambda *a, **k: tmp_path / "recipe.yaml")
    monkeypatch.setattr(software_spec, "software_spec_path_for_recipe", lambda _: software)
    monkeypatch.setattr(software_spec, "validate_software_spec",
                        lambda *a, **k: {"status": "reviewed" if reviewed else "draft"})
    return run, copied_spec, workspace, software, record


def _consume(prepared, *, descriptor=True):
    run, spec, workspace, _, _ = prepared
    return formal._private_source_freeze_for_formal(
        run, spec, "fixture", workspace=workspace, repo=run.parent,
        **({"descriptor": run.parent / "descriptor.yaml"} if descriptor else {}),
    )


def test_formal_accepts_only_verified_public_semantics_with_private_copies_masked(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch)
    assert _consume(prepared) == prepared[-1]
    with pytest.raises(ValueError, match="not frozen and masked"):
        _consume(prepared, descriptor=False)


@pytest.mark.parametrize("selection", [
    {"grant": False}, {"mode": "rw"}, {"note": "unreviewed grant"}, {"reviewed": False},
    {"mask_host": False}, {"mask_root": False},
])
def test_formal_refuses_unqualified_public_exception_or_unmasked_private_sources(
    tmp_path, monkeypatch, selection,
):
    with pytest.raises(ValueError):
        _consume(_prepared(tmp_path, monkeypatch, **selection))


def test_formal_refuses_public_semantics_changed_after_source_freeze(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch)
    prepared[3].write_bytes(b"changed semantics\n")
    with pytest.raises(ValueError, match="differs from authenticated private source"):
        _consume(prepared)
