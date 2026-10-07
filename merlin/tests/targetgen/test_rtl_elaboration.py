"""A selected configuration must produce fresh, repeatable FIRRTL bytes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.targetgen.rtl import elaboration, source_selection


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _source(tmp_path: Path, *, nondeterministic: bool = False) -> tuple[Path, str, list[str]]:
    root = tmp_path / "selected-source"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "configs.py").write_text("class SelectedConfig:\n    pass\n")
    (root / "emit.py").write_text(
        "import pathlib, sys\n"
        "assert sys.argv[1] == 'SelectedConfig'\n"
        "text = 'FIRRTL version 3.3.0\\ncircuit Top :%[[]]\\n  module Top :\\n'\n"
        + ("text += pathlib.Path(sys.argv[2]).parent.name\n" if nondeterministic else "")
        + "pathlib.Path(sys.argv[2]).write_text(text)\n"
    )
    _git(root, "add", "configs.py", "emit.py")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "source")
    return root, _git(root, "rev-parse", "HEAD"), [sys.executable, str(root / "emit.py"), "SelectedConfig", "{firrtl}"]


def test_two_fresh_git_bound_runs_feed_selected_hw_production(tmp_path: Path) -> None:
    root, revision, command = _source(tmp_path)
    receipt = elaboration.issue(
        source_root=root,
        revision=revision,
        config_file="configs.py",
        config="SelectedConfig",
        command=command,
        output=tmp_path / "elaboration",
    )
    selected_firrtl = receipt.parent / "run1" / "selected.fir"
    observed = elaboration.verify(receipt, firrtl=selected_firrtl, config="SelectedConfig")
    assert observed["status"] == "reproduced_exact_firrtl"
    assert observed["runs"][0]["firrtl_sha256"] == observed["runs"][1]["firrtl_sha256"]

    firtool = tmp_path / "firtool"
    firtool.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[-1]).write_text('module {\\n  hw.module @Top() {\\n    hw.output\\n  }\\n}\\n')\n"
    )
    firtool.chmod(0o755)
    selection_path = source_selection.produce_selection(
        target="fixture",
        firrtl=selected_firrtl,
        generator="fixture",
        config="SelectedConfig",
        core_root="Top",
        firtool=firtool,
        output=tmp_path / "selection",
        elaboration_receipt=receipt,
    )
    selection = source_selection.load_selection(selection_path, target="fixture")
    consistency = source_selection.production_consistency(selection)
    assert consistency["status"] == "verified"
    assert consistency["elaboration"]["status"] == "reproduced_exact_firrtl"
    assert consistency["elaboration"]["source_revision"] == revision

    (root / "configs.py").write_text("class OtherConfig:\n    pass\n")
    assert source_selection.production_consistency(selection)["status"] == "unverified"


def test_elaboration_refuses_stale_or_mutated_output(tmp_path: Path) -> None:
    root, revision, command = _source(tmp_path)
    with pytest.raises(ValueError, match="one \\{firrtl\\} output"):
        elaboration.issue(
            source_root=root,
            revision=revision,
            config_file="configs.py",
            config="SelectedConfig",
            command=command[:-1],
            output=tmp_path / "bad-command",
        )
    receipt = elaboration.issue(
        source_root=root,
        revision=revision,
        config_file="configs.py",
        config="SelectedConfig",
        command=command,
        output=tmp_path / "good-command",
    )
    firrtl = receipt.parent / "run1" / "selected.fir"
    firrtl.write_text("stale")
    with pytest.raises(ValueError, match="run 1 bytes"):
        elaboration.verify(receipt, firrtl=firrtl)
    with pytest.raises(ValueError, match="changed tracked files"):
        # The selected source file cannot be an edited working-tree copy.
        (root / "configs.py").write_text("class SelectedConfig: pass\n")
        elaboration.issue(
            source_root=root,
            revision=revision,
            config_file="configs.py",
            config="SelectedConfig",
            command=command,
            output=tmp_path / "changed-source",
        )


def test_fresh_nondeterministic_firrtl_never_receives_success_receipt(tmp_path: Path) -> None:
    root, revision, command = _source(tmp_path, nondeterministic=True)
    output = tmp_path / "nondeterministic"
    with pytest.raises(RuntimeError, match="different FIRRTL bytes"):
        elaboration.issue(
            source_root=root,
            revision=revision,
            config_file="configs.py",
            config="SelectedConfig",
            command=command,
            output=output,
        )
    assert not (output / "elaboration.json").exists()
    assert (output / "run1" / "stdout.log").is_file()
    assert (output / "run2" / "stdout.log").is_file()


def test_elaboration_receipt_rejects_unselected_config_and_submodule_pin(tmp_path: Path) -> None:
    root, revision, command = _source(tmp_path)
    with pytest.raises(ValueError, match="submodule Git link differs"):
        elaboration.issue(
            source_root=root,
            revision=revision,
            config_file="configs.py",
            config="SelectedConfig",
            command=command,
            submodules={"missing": "a" * 40},
            output=tmp_path / "missing-submodule",
        )
    receipt = elaboration.issue(
        source_root=root,
        revision=revision,
        config_file="configs.py",
        config="SelectedConfig",
        command=command,
        output=tmp_path / "good",
    )
    with pytest.raises(ValueError, match="configuration changed"):
        elaboration.verify(receipt, firrtl=receipt.parent / "run1" / "selected.fir", config="OtherConfig")
    document = json.loads(receipt.read_text())
    document["runs"][1]["argv"][2] = "OtherConfig"
    receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="run 2 bytes or command"):
        elaboration.verify(receipt, firrtl=receipt.parent / "run1" / "selected.fir")


def test_elaboration_attests_nested_selected_git_links(tmp_path: Path) -> None:
    root, _, command = _source(tmp_path)
    nested = tmp_path / "nested-source"
    nested.mkdir()
    _git(nested, "init", "-q")
    (nested / "arithmetic.scala").write_text("object Arithmetic {}\n")
    _git(nested, "add", "arithmetic.scala")
    _git(nested, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "nested")
    nested_revision = _git(nested, "rev-parse", "HEAD")

    child = tmp_path / "child-source"
    child.mkdir()
    _git(child, "init", "-q")
    (child / "unit.scala").write_text("object Unit {}\n")
    _git(child, "add", "unit.scala")
    _git(child, "update-index", "--add", "--cacheinfo", f"160000,{nested_revision},dependencies/fp")
    _git(child, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "child")
    child_revision = _git(child, "rev-parse", "HEAD")

    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{child_revision},generators/unit")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "selected links")
    revision = _git(root, "rev-parse", "HEAD")
    subprocess.run(["git", "clone", "-q", str(child), str(root / "generators" / "unit")], check=True)
    subprocess.run(
        ["git", "clone", "-q", str(nested), str(root / "generators" / "unit" / "dependencies" / "fp")],
        check=True,
    )
    pins = {
        "generators/unit": child_revision,
        "generators/unit/dependencies/fp": nested_revision,
    }
    receipt = elaboration.issue(
        source_root=root,
        revision=revision,
        config_file="configs.py",
        config="SelectedConfig",
        command=command,
        submodules=pins,
        output=tmp_path / "nested-elaboration",
    )
    assert elaboration.verify(receipt, firrtl=receipt.parent / "run1" / "selected.fir")["source"]["submodules"] == pins
    (root / "generators" / "unit" / "dependencies" / "fp" / "arithmetic.scala").write_text("changed\n")
    with pytest.raises(ValueError, match="changed tracked files|selected submodule checkout differs"):
        elaboration.verify(receipt, firrtl=receipt.parent / "run1" / "selected.fir")


def test_elaboration_attests_external_generator_artifact(tmp_path: Path) -> None:
    root, revision, command = _source(tmp_path)
    generator_artifact = tmp_path / "compiled-generator.jar"
    generator_artifact.write_bytes(b"selected compiled tool")
    receipt = elaboration.issue(
        source_root=root,
        revision=revision,
        config_file="configs.py",
        config="SelectedConfig",
        command=command,
        output=tmp_path / "compiled-generator-elaboration",
        tool_inputs=[generator_artifact],
    )
    firrtl = receipt.parent / "run1" / "selected.fir"
    observed = elaboration.verify(receipt, firrtl=firrtl)
    assert observed["tool"]["inputs"][0]["path"] == str(generator_artifact)
    generator_artifact.write_bytes(b"changed compiled tool")
    with pytest.raises(ValueError, match="elaboration tool changed"):
        elaboration.verify(receipt, firrtl=firrtl)


def test_elaboration_cli_uses_explicit_command_file_and_reports_status(tmp_path: Path, capsys) -> None:
    root, revision, command = _source(tmp_path)
    command_file = tmp_path / "command.json"
    command_file.write_text(json.dumps(command))
    output = tmp_path / "cli-elaboration"
    status = elaboration.main(
        [
            "--source-root",
            str(root),
            "--revision",
            revision,
            "--config-file",
            "configs.py",
            "--config",
            "SelectedConfig",
            "--command-json",
            str(command_file),
            "--output",
            str(output),
        ]
    )
    assert status == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "REPRODUCED",
        "receipt": str(output / "elaboration.json"),
    }
    assert (
        elaboration.main(
            [
                "--source-root",
                str(root),
                "--revision",
                revision,
                "--config-file",
                "configs.py",
                "--config",
                "OtherConfig",
                "--command-json",
                str(command_file),
                "--output",
                str(tmp_path / "refused"),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
