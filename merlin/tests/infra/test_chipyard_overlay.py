"""An overlay build may write anywhere in the overlay and nowhere in the shared checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root

_TOOL = repo_root() / "build_tools/firesim/make_chipyard_overlay.py"


def _base(root: Path) -> Path:
    for relative in (
        "build.sbt",
        "env.sh",
        "sims/firesim/sourceme-manager.sh",
        "sims/firesim-staging/Makefile",
        "sims/firesim/deploy/config_hwdb.yaml",
        "sims/firesim/sim/Makefile",
        "sims/firesim/deploy/firesim",
        "generators/unit/.git",
        "generators/unit/include/params.h",
        "generators/other/.git",
        "sims/firesim-staging/generated-src/old/design.fir",
        ".bloop/state",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (root / "build.sbt").write_text(
        'lazy val unit = (project in file("generators/unit"))\n'
        'lazy val other = (project in file("./generators/other"))\n',
        encoding="utf-8",
    )
    nested = root / "sims/firesim/sim"
    (nested / "build.sbt").write_text('lazy val lib = (project in file("lib"))\n', encoding="utf-8")
    (nested / "lib/src").mkdir(parents=True)
    (root / "tools/odd/src/target").mkdir(parents=True)  # declared as file("tools/odd") / "src"
    (root / "generators/unit/target").mkdir()
    (root / "generators/unit/target/classes").write_text("base build", encoding="utf-8")
    return root


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_TOOL), *arguments], capture_output=True, text=True, check=False)


def test_reads_are_links_and_writes_are_private(tmp_path: Path) -> None:
    base, overlay = _base(tmp_path / "base"), tmp_path / "overlay"
    before = sorted(str(path) for path in base.rglob("*"))
    done = _run(
        "--base",
        str(base),
        "--overlay",
        str(overlay),
        "--exclude-generator",
        "other",
        "--private-dir",
        "generators/unit/include",
    )
    assert done.returncode == 0, done.stderr

    assert (overlay / "build.sbt").is_symlink()
    assert (overlay / "sims/firesim-staging/Makefile").is_symlink()
    for private in (
        "target",
        "sims/firesim-staging/generated-src",
        "sims/firesim/sim/output",
        "sims/firesim/deploy/results-build",
    ):
        directory = overlay / private
        assert directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir())
    # The build's own rule skips a generator whose directory has no `.git`.
    assert list((overlay / "generators/other").iterdir()) == []
    assert (overlay / "generators/unit/.git").is_symlink()
    # sbt writes a project's classes beside its sources: that directory must not be the base's.
    project_target = overlay / "generators/unit/target"
    assert project_target.is_dir() and not project_target.is_symlink()
    assert not any(project_target.iterdir())
    (project_target / "classes").write_text("overlay build", encoding="utf-8")
    assert not (overlay / "generators/other/target").exists()
    # A second sbt build inside the tree, and a project only visible as an existing target.
    for private in ("sims/firesim/sim/target", "sims/firesim/sim/lib/target", "tools/odd/src/target"):
        directory = overlay / private
        assert directory.is_dir() and not directory.is_symlink(), private
    header = overlay / "generators/unit/include/params.h"
    assert header.is_file() and not header.is_symlink()
    config = overlay / "sims/firesim/deploy/config_hwdb.yaml"
    assert config.is_file() and not config.is_symlink()
    assert not (overlay / ".bloop").exists()  # two builds must not share one build server

    header.write_text("regenerated", encoding="utf-8")
    (overlay / "sims/firesim/sim/output/driver").write_text("built", encoding="utf-8")
    assert sorted(str(path) for path in base.rglob("*")) == before
    assert (base / "generators/unit/target/classes").read_text(encoding="utf-8") == "base build"
    assert (base / "generators/unit/include/params.h").read_text(encoding="utf-8") == "generators/unit/include/params.h"


def test_an_existing_overlay_is_not_merged_into(tmp_path: Path) -> None:
    base, overlay = _base(tmp_path / "base"), tmp_path / "overlay"
    overlay.mkdir()
    done = _run("--base", str(base), "--overlay", str(overlay))
    assert done.returncode == 2 and "refusing" in done.stderr


def test_a_bitstream_builds_per_design_platform_copy_lands_in_the_overlay(tmp_path: Path) -> None:
    # The third place a build wrote into the base: `platforms/<platform>/` was a link, so the
    # per-design copy a bitstream build makes beside the platform template was created in the
    # shared checkout.
    base = _base(tmp_path / "base")
    template = base / "sims/firesim/platforms/board_x/cl_template/scripts/run.tcl"
    template.parent.mkdir(parents=True)
    template.write_text("template", encoding="utf-8")
    overlay = tmp_path / "overlay"
    assert _run("--base", str(base), "--overlay", str(overlay)).returncode == 0

    platform = overlay / "sims/firesim/platforms/board_x"
    assert platform.is_dir() and not platform.is_symlink()
    assert (platform / "cl_template").is_symlink()  # the template is read
    assert (platform / "cl_template/scripts/run.tcl").read_text() == "template"
    (platform / "cl_my_design").mkdir()  # what the build does
    (platform / "cl_my_design/stamp").write_text("", encoding="utf-8")
    assert not (base / "sims/firesim/platforms/board_x/cl_my_design").exists()


def test_the_manager_entry_script_is_a_copy_so_a_run_roots_in_the_overlay(tmp_path: Path) -> None:
    # The queue launcher resolves `deploy/firesim` to find the checkout. As a link it resolved to
    # the base, and a job looked for this overlay's host driver in the base's output directory.
    base = _base(tmp_path / "base")
    overlay = tmp_path / "overlay"
    assert _run("--base", str(base), "--overlay", str(overlay)).returncode == 0
    script = overlay / "sims/firesim/deploy/firesim"
    assert script.is_file() and not script.is_symlink()
    assert script.resolve().parents[3] == overlay.resolve()


def test_the_overlays_sudo_scripts_can_be_made_to_match_the_hosts(tmp_path: Path) -> None:
    base = _base(tmp_path / "base")
    scripts = base / "sims/firesim/deploy/sudo-scripts"
    scripts.mkdir(parents=True)
    (scripts / "set-perms").write_text("chmod a+rw dev", encoding="utf-8")
    (scripts / "load-module").write_text("modprobe x", encoding="utf-8")
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "set-perms").write_text("chgrp runner dev", encoding="utf-8")
    (installed / "unrelated").write_text("not a FireSim script", encoding="utf-8")
    overlay = tmp_path / "overlay"
    assert (
        _run("--base", str(base), "--overlay", str(overlay), "--installed-sudo-scripts", str(installed)).returncode == 0
    )
    mine = overlay / "sims/firesim/deploy/sudo-scripts"
    assert mine.is_dir() and not mine.is_symlink()
    assert (mine / "set-perms").read_text() == "chgrp runner dev"  # matches the host
    assert (mine / "load-module").read_text() == "modprobe x"  # the host has none: kept
    assert not (mine / "unrelated").exists()
    assert (scripts / "set-perms").read_text() == "chmod a+rw dev"  # the base is untouched
