"""Reviewed semantics grants never publish private model dependency directories."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import corpus_inputs as CI
from merlin_experiments.phase1.feedback import private_source_freeze as SF

from merlin.targetgen import tool_profile


@pytest.fixture
def joined(tmp_path, monkeypatch):
    source = tmp_path / "software.yaml"
    template = Path(__file__).resolve().parents[3] / "examples/gemmini/target/software-spec.yaml"
    source.write_bytes(template.read_bytes())
    frozen = tmp_path / "private/frozen/software.bin"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(source.read_bytes())
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"software_spec": str(source)}))
    te = SimpleNamespace(target="gemmini", numeric_profile=str(recipe), _source_root=lambda: tmp_path)
    bundle = {"allowed": [{"path": str(source), "mode": "ro", "note": "admitted software semantics"}]}
    paths = [
        {"path": str(source), "kind": "file"},
        {"path": str(frozen.parent), "kind": "dir"},
        {"path": str(tmp_path / "private/capture"), "kind": "dir"},
        {"path": str(tmp_path / "private/goldens.npz"), "kind": "file"},
    ]
    resolved = {(str(source), "software_spec"): frozen}
    monkeypatch.setattr(
        tool_profile, "verify_tool_profile", lambda _: {"common_substrate": tool_profile.COMMON_SUBSTRATE_V2}
    )
    monkeypatch.setattr(SF, "verify", lambda *a, **k: resolved)

    def run():
        return CI._reconcile_public_software_denial(
            te, bundle, paths, spec=tmp_path / "spec.yaml", source_freeze={}, source_root=frozen.parent
        )

    return source, frozen, bundle, paths, resolved, run


def test_exact_reviewed_software_join_preserves_all_private_dependencies(joined):
    source, frozen, _, paths, _, run = joined
    original = source.read_bytes(), frozen.read_bytes()
    assert run() == paths[1:]
    assert (source.read_bytes(), frozen.read_bytes()) == original


def test_changed_frozen_software_cannot_become_public(joined):
    _, frozen, _, _, _, run = joined
    frozen.write_bytes(frozen.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="authenticated private source"):
        run()


def test_non_semantic_grant_never_removes_a_private_denial(joined):
    _, _, bundle, paths, _, run = joined
    bundle["allowed"][0]["note"] = "other input"
    assert run() == paths


def test_software_with_private_nonsoftware_role_is_refused(joined):
    source, frozen, _, _, resolved, run = joined
    resolved[(str(source), "host_capabilities")] = frozen
    with pytest.raises(ValueError, match="non-software role"):
        run()


def test_unreviewed_software_cannot_become_public(joined):
    source, frozen, _, _, _, run = joined
    changed = yaml.safe_load(source.read_bytes())
    changed["status"] = "unreviewed"
    raw = yaml.safe_dump(changed).encode()
    source.write_bytes(raw)
    frozen.write_bytes(raw)
    with pytest.raises(ValueError, match="not reviewed"):
        run()
