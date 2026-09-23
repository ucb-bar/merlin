"""Shared setuptools build routing for optional distributions (linked as their setup.py).

PEP 517 invokes the backend in the project directory. Resolve that directory, not this file's
symlink target. Checkout builds stay beneath out/build/python/<distribution>; an extracted sdist
has no workspace layout and uses its own out/build/python/<distribution> instead.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import tomllib
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.egg_info import egg_info as _egg_info
from setuptools.command.sdist import sdist as _sdist

_PROJECT = Path.cwd().resolve()
_METADATA = tomllib.loads((_PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
_NAME = _METADATA["project"]["name"]
_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
if (
    not isinstance(_NAME, str)
    or not _NAME
    or _NAME[0] not in _ALNUM
    or _NAME[-1] not in _ALNUM
    or any(char not in _ALNUM + "-_." for char in _NAME)
):
    raise ValueError("invalid extension distribution name")
_WORKSPACE = _PROJECT.parent.parent
if _PROJECT.parent.name != "packages" or not (_WORKSPACE / "build_tools" / "extension_setup.py").is_file():
    _WORKSPACE = _PROJECT
_OUTPUT = _WORKSPACE / "out" / "build" / "python" / _NAME


def _source_files(names: list[str]) -> list[str]:
    """Keep source manifests project-relative; external build metadata is not source."""
    selected = []
    for name in names:
        path = _PROJECT / name
        # The sole intentional external source is this linked build hook. The sdist materializes
        # its bytes as a regular setup.py, so rebuilding never needs the parent checkout.
        if path == _PROJECT / "setup.py":
            selected.append("setup.py")
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(_PROJECT) or resolved.is_relative_to(_OUTPUT):
            continue
        relative = resolved.relative_to(_PROJECT)
        if any(part.endswith(".egg-info") or part == "__pycache__" for part in relative.parts):
            continue
        selected.append(relative.as_posix())
    return sorted(set(selected))


class Build(_build):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.build_base = str(_OUTPUT / "build")


class BuildPy(_build_py):
    def run(self) -> None:
        # A moved or removed module must not survive a later wheel through stale staging.
        # Only this distribution's dedicated build-lib subtree can be cleared; never source,
        # metadata, archives, the output root, or a caller-provided location outside staging.
        staged = Path(self.build_lib)
        build_root = _OUTPUT / "build"
        resolved = staged.resolve()
        if (
            staged.is_symlink()
            or build_root.is_symlink()
            or not build_root.resolve().is_relative_to(_OUTPUT.resolve())
            or resolved == build_root.resolve()
            or not resolved.is_relative_to(build_root.resolve())
        ):
            raise ValueError(f"extension package staging must be below {build_root}: {staged}")
        if staged.exists():
            shutil.rmtree(staged)
        super().run()


class Wheel(_bdist_wheel):
    def run(self) -> None:
        # setuptools removes this directory only after a successful archive. An interrupted older
        # build can otherwise inject stale files despite build_lib having been refreshed correctly.
        staged = Path(self.bdist_dir)
        base = Path(self.get_finalized_command("bdist").bdist_base)
        allowed = (_OUTPUT / "build").resolve()
        if (
            staged.is_symlink()
            or base.is_symlink()
            or staged.resolve() != (base / "wheel").resolve()
            or base.parent.resolve() != allowed
            or not base.name.startswith("bdist.")
        ):
            raise ValueError(f"extension wheel staging must be its dedicated directory below {allowed}: {staged}")
        if staged.exists():
            shutil.rmtree(staged)
        super().run()


class EggInfo(_egg_info):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.egg_base = str(_OUTPUT / "metadata")

    def finalize_options(self) -> None:
        Path(self.egg_base).mkdir(parents=True, exist_ok=True)
        super().finalize_options()

    def find_sources(self) -> None:
        super().find_sources()
        # setuptools appends the egg-info directory to SOURCES.txt. With output metadata outside
        # the source tree that would embed absolute paths and break a wheel rebuilt from sdist.
        self.filelist.files = _source_files(self.filelist.files)
        self.write_file(
            "manifest file", os.path.join(self.egg_info, "SOURCES.txt"), "\n".join(self.filelist.files) + "\n"
        )


class SourceDistribution(_sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, _source_files(files))
        script = Path(base_dir) / "setup.py"
        # A hard-linked symlink would point outside the extracted archive. Always package bytes.
        script.unlink(missing_ok=True)
        shutil.copyfile(_PROJECT / "setup.py", script)

    def make_distribution(self) -> None:
        _OUTPUT.mkdir(parents=True, exist_ok=True)
        dist_dir = Path(self.dist_dir).resolve()
        dist_dir.mkdir(parents=True, exist_ok=True)
        # The default sdist command stages <name-version>/ in the source directory. Keep even
        # transient staging out of source, while preserving --keep-temp for build debugging.
        with tempfile.TemporaryDirectory(prefix="sdist-", dir=_OUTPUT, delete=not self.keep_temp) as stage:
            name = self.distribution.get_fullname()
            self.make_release_tree(str(Path(stage) / name), self.filelist.files)
            self.archive_files = [
                self.make_archive(
                    str(dist_dir / name),
                    fmt,
                    root_dir=stage,
                    base_dir=name,
                    owner=self.owner,
                    group=self.group,
                )
                for fmt in self.formats
            ]


setup(
    cmdclass={
        "build": Build,
        "build_py": BuildPy,
        "bdist_wheel": Wheel,
        "egg_info": EggInfo,
        "sdist": SourceDistribution,
    }
)
