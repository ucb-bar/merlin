"""New release selections retain exact old defaults and bind an actual SDK export."""

from pathlib import Path

import pytest
from merlin_experiments.corpus import preparation, release
from merlin_experiments.spec import SpecError

from merlin.common.paths import python_source_dir, repo_root
from merlin.targetgen import tool_profile as TP
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release


def test_omitted_selection_keeps_original_v3_source():
    assert preparation.compiler_source_selection() == (TP.COMMON_SUBSTRATE_V3, python_source_dir())


def test_real_v6_export_selects_exact_public_inventory_and_mutation_refuses(tmp_path):
    sdk = tmp_path / "sdk"
    tools = tuple(TR.ARM_TOOLS["merlin_assisted"])
    exported = prepare_shared_source_release(
        sdk, python_source_root=python_source_dir(), license_directory=repo_root(), tools=tools, source_binding=True
    )
    assert preparation.compiler_source_selection(TP.COMMON_SUBSTRATE_V6, sdk) == (TP.COMMON_SUBSTRATE_V6, sdk)
    bundle = {"arm": "merlin_assisted", "tools": list(tools), "allowed": [], "denied": [], "host_inputs": []}
    TP.attach_common_substrate(bundle, version=TP.COMMON_SUBSTRATE_V6, python_source_root=sdk)
    profile = bundle["compiler_tool_profile"]
    assert TP.verify_tool_profile(bundle) == profile
    assert profile["shared_scheduling_library"] == exported
    assert "merlin.compile.scheduling.configuration_ir" in profile["common_public_modules"]
    member = sdk / exported["members"][0]["path"]
    member.write_bytes(member.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="bytes differ"):
        preparation.compiler_source_selection(TP.COMMON_SUBSTRATE_V6, sdk)


@pytest.mark.parametrize("version,root", [("unknown", None), (TP.COMMON_SUBSTRATE_V6, Path("relative"))])
def test_invalid_sdk_selection_refuses_before_release_output(tmp_path, version, root):
    output = tmp_path / "output"
    with pytest.raises((ValueError, SpecError)):
        release.prepare(tmp_path / "missing-run", output=output, common_substrate=version, python_source_root=root)
    assert not output.exists()


def test_release_forwards_explicit_selection_to_actual_scaffold(release_fixture, capsys, monkeypatch):
    from test_corpus_release import _prepare

    # The existing source/run fixture executes ordinary release assembly and
    # real scaffold; observe its call without replacing the implementation.
    original = release.scaffold
    selected = []

    def observed(*args, **kwargs):
        selected.append((kwargs.get("common_substrate"), kwargs.get("python_source_root")))
        return original(*args, **kwargs)

    monkeypatch.setattr(release, "scaffold", observed)
    _prepare(release_fixture, capsys)
    assert selected == [(None, None)]


def test_release_cli_materializes_explicit_v6_sdk(release_fixture, capsys):
    import json

    from merlin_experiments.cli import main

    sdk = release_fixture["root"] / "sdk"
    prepare_shared_source_release(
        sdk,
        python_source_root=Path(TP.__file__).resolve().parents[2],
        license_directory=Path(TP.__file__).resolve().parents[3],
        tools=tuple(TR.ARM_TOOLS["merlin_assisted"]),
        source_binding=True,
    )
    assert (
        main(["run", str(release_fixture["definition"]), "--phase", "0", "--run-dir", str(release_fixture["run"])]) == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(release_fixture["run"]),
                "--output",
                str(release_fixture["release"]),
                "--common-substrate",
                TP.COMMON_SUBSTRATE_V6,
                "--python-source-root",
                str(sdk),
            ]
        )
        == 0
    )
    capsys.readouterr()
    prepared = json.loads((release_fixture["release"] / "private/preparation.json").read_text())
    assert prepared["scaffolding"]["bundle_generation"]["compiler_source"] == {
        "common_substrate": TP.COMMON_SUBSTRATE_V6,
        "python_source_root": str(sdk),
    }
    manifests = list(
        (release_fixture["release"] / "payload/experiment/input_bundles").glob("*/input_bundle_manifest.yaml")
    )
    assert manifests
    import yaml

    bundle = yaml.safe_load(manifests[0].read_text())
    assert TP.verify_tool_profile(bundle)["common_substrate"] == TP.COMMON_SUBSTRATE_V6


# Reuse the ordinary harmless fixture, with no hardware, tool or candidate authority.
pytest_plugins = ("test_corpus_release",)
