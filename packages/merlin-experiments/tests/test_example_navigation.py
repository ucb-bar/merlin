"""Public workflow maps follow the catalog without copying definitions or running engines."""

import re
import subprocess

import pytest
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.spec import SpecError, catalog, load_spec

from merlin.common.paths import repo_root


@pytest.mark.parametrize("folder", ["gemmini", "gemmini_universal", "atlas", "radiance", "mx_gemmini"])
def test_workflow_map_matches_canonical_definition_and_local_links(folder, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("navigation inspection launched a process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    root = repo_root()
    example = root / "examples" / folder
    spec = load_spec(example / "experiment.yaml")
    assert catalog()[spec.id] == spec.path
    guide = example / "README.md"
    text = guide.read_text()
    assert f"merlin experiment inspect {spec.id} --phase 1" in text
    assert f"merlin experiment runs --target {spec.target}" in text
    assert f"{folder}/README.md" in (root / "examples/README.md").read_text()
    assert "experiment.yaml" in text
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0]
        if target and "://" not in target:
            assert (guide.parent / target).exists(), f"broken workflow link: {target}"
    if "0" in spec.document["phases"]:
        assert spec.resolve(spec.document["phases"]["0"]["config"]["recipe"]) == example / "phase0/recipe.yaml"
    else:
        assert folder == "gemmini_universal"
        assert "no reviewed Universal-specific" in (example / "phase0/README.md").read_text()
    assert spec.resolve(spec.document["phases"]["1"]["config"]["descriptor"]) == example / "target/descriptor.yaml"


def test_gemmini_guide_and_catalog_select_installed_rtlchecks(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("route inspection launched a process"))
    root = repo_root()
    spec = load_spec(root / "examples/gemmini/experiment.yaml")
    config = spec.document["phases"]["1"]["config"]
    adapter = ADAPTERS["capsule_bench"]
    adapter.validate(config)
    command = adapter.resolve(spec, config, root, tmp_path / "run")
    assert config["treatment"] == "rtlchecks"
    assert command["module"] == "merlin_experiments.phase1"
    assert command["argv"][1:3] == ["-m", "merlin_experiments.phase1"]
    assert command["argv"][command["argv"].index("--treatment") + 1] == "rtlchecks"
    for field in ("bundle", "bundle_manifest", "oracle_timing"):
        missing = {name: value for name, value in config.items() if name != field}
        with pytest.raises(SpecError, match="installed Phase 1 requires explicit inputs"):
            adapter.validate(missing)
    guide = " ".join((root / "examples/gemmini/phase1/README.md").read_text().split())
    assert "not the command produced by that catalog definition" not in guide
    assert "python -m merlin_experiments.phase1" in guide
    assert "baseline-functional-template" in guide


def test_measured_template_and_guide_follow_installed_managed_route(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("navigation launched a process"))
    root = repo_root()
    spec = load_spec(root / "experiments/definitions/measured-claims-template.yaml")
    assert spec.document["kind"] == "template"
    adapter = ADAPTERS["measured_claims"]
    assert adapter.module == "merlin_experiments.phase2.chia_envelope_cli"
    assert adapter.resume == "native_chain"
    config = spec.document["phases"]["2"]["config"]
    adapter.validate(config)
    assert {name for name, option in adapter.options.items() if option.required} <= config.keys()
    assert "managed_native_endpoint" in config
    assert "chia_wrapper" not in config  # The installed adapter owns executable selection.
    guide = " ".join((root / "examples/gemmini/phase2/README.md").read_text().split())
    assert "installed managed Chia envelope" in guide
    assert "model-portfolio adapter retains its native" in guide
    details = (root / "examples/gemmini/phase2/installed-measured-claims.md").read_text()
    assert "merlin experiment resume RUN" in details
    assert "no separate `--resume` flag" in details
