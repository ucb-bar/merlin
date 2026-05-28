"""Build packaging helpers — `package_dist`, `maybe_install`.

Bundles the host install tree into a release tarball and runs cmake's
`--install` step.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tarfile

import utils


def package_dist(
    build_dir: pathlib.Path,
    install_dir: pathlib.Path,
    dist_dir: pathlib.Path,
    dist_name: str,
    include_runtime_samples: bool,
) -> pathlib.Path:
    if not install_dir.exists():
        raise FileNotFoundError(f"Install tree not found: {install_dir}")

    dist_dir.mkdir(parents=True, exist_ok=True)

    stage_dir = dist_dir / dist_name
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    if include_runtime_samples:
        # Runtime target package layout:
        #   <artifact>/install/...
        #   <artifact>/runtime/plugins/merlin-samples/...
        shutil.copytree(install_dir, stage_dir / "install")

        runtime_samples_dir = build_dir / "runtime" / "plugins" / "merlin-samples"
        if not runtime_samples_dir.exists():
            raise FileNotFoundError(f"Runtime sample tree not found: {runtime_samples_dir}")

        shutil.copytree(
            runtime_samples_dir,
            stage_dir / "runtime" / "plugins" / "merlin-samples",
        )
    else:
        # Host package layout:
        #   <artifact>/bin
        #   <artifact>/lib
        #   ...
        shutil.copytree(install_dir, stage_dir)

    archive_path = dist_dir / f"{dist_name}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(stage_dir, arcname=dist_name)

    shutil.rmtree(stage_dir)
    return archive_path


def maybe_install(
    cmake_bin: str,
    build_dir: pathlib.Path,
    strip_install: bool,
    dry_run: bool,
    env: dict[str, str],
) -> int:
    cmd = [cmake_bin, "--install", str(build_dir)]
    if strip_install:
        cmd.append("--strip")
    return utils.run(cmd, dry_run=dry_run, env=env)
