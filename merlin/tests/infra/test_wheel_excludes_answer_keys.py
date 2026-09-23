"""Build real release archives with hostile on-disk canaries; never scan private corpora."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


@pytest.fixture(scope="module")
def release_archives(tmp_path_factory):
    root = repo_root()
    checkout = tmp_path_factory.mktemp("release-canary")
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md"):
        shutil.copyfile(root / name, checkout / name)
    manifest = json.loads((root / "build_tools/package_resources.json").read_text())
    for name in ["build_tools/package_resources.json", *manifest["files"]]:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    # A minimal real package avoids compiling or importing unrelated optional toolchains.
    package = checkout / "src/merlin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    stale = package / "_data/contract/hidden/secret.yaml"
    stale.parent.mkdir(parents=True)
    stale.write_text("PRIVATE_RELEASE_CANARY")
    for name in (
        "hidden/secret.yaml",
        "golden/secret.yaml",
        "x/golden.yaml",
        "x/capsule.hidden.yaml",
        "unreviewed.yaml",
    ):
        path = checkout / "merlin/contract" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PRIVATE_RELEASE_CANARY")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", "out/build/dist"],
        cwd=checkout,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "MERLIN_REPO_ROOT"}},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert stale.read_text() == "PRIVATE_RELEASE_CANARY", "release build mutated the source package"
    assert list((package / "_data").rglob("*.yaml")) == [stale]
    assert not list((checkout / "src").glob("*.egg-info")), "metadata must live under out/build"
    wheel = next((checkout / "out/build/dist").glob("*.whl"))
    sdist = next((checkout / "out/build/dist").glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    with tarfile.open(sdist) as archive:
        sdist_files = {
            member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()
        }
    return manifest, wheel_files, sdist_files


def test_archives_never_include_unreviewed_answer_files(release_archives):
    _, wheel, sdist = release_archives
    for files in (wheel, sdist):
        assert files
        assert all(b"PRIVATE_RELEASE_CANARY" not in contents for contents in files.values())


def test_every_declared_public_resource_survives_both_builds(release_archives):
    manifest, wheel, sdist = release_archives
    for name in manifest["files"]:
        relative = Path(name).relative_to("merlin").as_posix()
        assert wheel["merlin/_data/" + relative] == (repo_root() / name).read_bytes()
        assert any(path.endswith("/" + name) for path in sdist), name


def test_resource_manifest_never_names_private_corpus_files():
    files = json.loads((repo_root() / "build_tools/package_resources.json").read_text())["files"]
    assert "merlin/contract/schemas/command_buffer.schema.json" in files
    assert len(files) == len(set(files))
    for name in files:
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert not any(part.startswith(("golden", "hidden")) for part in path.parts)
        assert not name.endswith(".hidden.yaml")
        assert (repo_root() / name).is_file()
