"""An uncertified publish must SAY SO, and a root-rooted package must arrive runnable.

Two regressions, both of which produced an artifact that looked correct and was not:

1. A package whose entrypoint sits at the package ROOT was published with its tool MISSING. The
   export hoists ``mlir_oot/`` to the repo root; for this shape that dissolved the very package the
   tool imports and dropped every root-level file, so the assembled tree had no ``<target>-opt`` at
   all while ``manifest.yaml`` still named one.
2. ``--no-gate`` wrote its warning to stderr and shipped the ordinary "certified champion" README
   and commit subject. The only reader who saw the warning was the operator who already knew; the
   person who clones the repo saw a certified champion.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from merlin.common.yaml import dump_yaml
from merlin.targetgen import publish as pub


def _write(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _make_root_rooted_python_package(targets_root: Path, target: str = "gemmini",
                                     package_id: str = "graded_only_v0",
                                     status: str = "capsule_graded_l3_partial") -> Path:
    """An interpreted package shaped like a capsule-bench submission: the tool at the package root,
    the importable tree beside it, docs and a report alongside."""
    d = targets_root / target / package_id
    tool = f"{target}-opt"
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": target,
        "package_id": package_id,
        "family": "tensor_resident",
        "language": "python",
        "status": status,
        "version": 0,
        "integrity_exempt": False,
        "authoring": {"mode": "agent_generated_from_rtl_facts", "author": "fixture",
                      "generated_by_agent": True},
        "entrypoints": {"tool": tool},
        "commands": {
            "parse": {"argv": ["{tool}", "--verify-diagnostics", "{input_mlir}"]},
            "lower_interface_to_target": {"argv": ["{tool}", f"--convert-iface-to-{target}",
                                                   "{input_mlir}"]},
            "emit_command_buffer": {"argv": ["{tool}", "--emit-command-buffer={output_json}",
                                             "{input_mlir}"]},
            "lower_target_to_llvm": {"argv": ["{tool}", "--emit-target-artifact", "{input_mlir}"]},
        },
        "publication": {"champion": False, "certification": "not_certified"},
    }
    _write(d / "manifest.yaml", dump_yaml(manifest))
    _write(d / tool,
           "#!/usr/bin/env python3\n"
           "import sys\n"
           "from pathlib import Path\n"
           "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
           "from mlir_oot.opt import main\n"
           "raise SystemExit(main())\n",
           mode=0o755)
    _write(d / "mlir_oot" / "__init__.py", "")
    _write(d / "mlir_oot" / "tables.py", "OK = 'derived'\n")
    _write(d / "mlir_oot" / "opt.py",
           "from .tables import OK\n\n\n"
           "def main():\n"
           "    print(OK)\n"
           "    return 0\n")
    _write(d / "REPORT.md", "# report\n")
    _write(d / "docs" / "notes.md", "# notes\n")
    return d


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    (root / "artifacts" / "targets").mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    return root


def test_a_root_rooted_package_keeps_its_entrypoint_and_stays_importable(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    _make_root_rooted_python_package(troot)
    monkeypatch.setenv("MERLIN_PUBLISH_REMOTE_GEMMINI", "file:///dev/null")

    res = pub.publish("gemmini", dry_run=True, package_id="graded_only_v0", gate=False)
    repo = res.repo_dir

    tool = repo / "gemmini-opt"
    assert tool.is_file(), "the published tree lost the entrypoint its manifest declares"
    assert tool.stat().st_mode & 0o111, "the entrypoint lost its executable bit"
    # the tree the tool imports must still be a PACKAGE, not scattered to the top level
    assert (repo / "mlir_oot" / "opt.py").is_file()
    assert not (repo / "opt.py").exists()
    # nothing root-level is silently dropped
    assert (repo / "REPORT.md").is_file()
    assert (repo / "docs" / "notes.md").is_file()
    # and the manifest still points at the tool that is actually there
    from merlin.common.yaml import load_yaml
    assert load_yaml(repo / "manifest.yaml")["entrypoints"]["tool"] == "gemmini-opt"
    # the strongest check: it RUNS from the assembled tree
    proc = subprocess.run([str(tool)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "derived" in proc.stdout


def test_an_interpreted_package_is_not_told_to_run_cmake(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    _make_root_rooted_python_package(troot)
    monkeypatch.setenv("MERLIN_PUBLISH_REMOTE_GEMMINI", "file:///dev/null")
    res = pub.publish("gemmini", dry_run=True, package_id="graded_only_v0", gate=False)
    readme = (res.repo_dir / "README.md").read_text(encoding="utf-8")
    assert "cmake" not in readme.lower(), "a python package has no CMakeLists.txt to build"
    assert "./gemmini-opt" in readme
    assert not (res.repo_dir / "CMakeLists.txt").exists()


def test_a_no_gate_publish_says_uncertified_in_the_readme(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    _make_root_rooted_python_package(troot)
    monkeypatch.setenv("MERLIN_PUBLISH_REMOTE_GEMMINI", "file:///dev/null")
    res = pub.publish("gemmini", dry_run=True, package_id="graded_only_v0", gate=False)

    readme = (res.repo_dir / "README.md").read_text(encoding="utf-8")
    assert "NOT CERTIFIED" in readme
    assert "--no-gate" in readme
    assert "capsule_graded_l3_partial" in readme
    assert "certified champion codegen package" not in readme
    assert "Certified against" not in readme


def test_a_certified_publish_keeps_the_champion_wording(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    d = _make_root_rooted_python_package(troot, package_id="certified_v0", status="rtl_certified")
    monkeypatch.setenv("MERLIN_PUBLISH_REMOTE_GEMMINI", "file:///dev/null")
    res = pub.publish("gemmini", dry_run=True, package_id="certified_v0", gate=True)
    readme = (res.repo_dir / "README.md").read_text(encoding="utf-8")
    assert "NOT CERTIFIED" not in readme
    assert "certified champion codegen package" in readme
    assert d.is_dir()


def test_the_uncertified_warning_reaches_the_published_history(out_root, monkeypatch, tmp_path):
    """The commit subject and the tag must not call an ungated package a champion."""
    troot = out_root / "artifacts" / "targets"
    _make_root_rooted_python_package(troot)
    bare = out_root / "build" / "publish" / "_fake_remotes" / "gemmini.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    monkeypatch.setenv("MERLIN_PUBLISH_REMOTE_GEMMINI", f"file://{bare}")

    res = pub.publish("gemmini", dry_run=False, package_id="graded_only_v0", gate=False,
                      branch="profiling/graded_only_v0")
    assert res.committed

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "-b", "profiling/graded_only_v0", f"file://{bare}",
                    str(clone)], check=True)
    msg = subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%B"],
                         capture_output=True, text=True, check=True).stdout
    assert "UNCERTIFIED" in msg.splitlines()[0]
    assert "champion" not in msg.splitlines()[0]
    assert "--no-gate" in msg
    assert "Gate-Refusal:" in msg
    # and the clone is runnable, which is the whole point of publishing it
    assert (clone / "gemmini-opt").is_file()
    proc = subprocess.run([str(clone / "gemmini-opt")], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
