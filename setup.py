"""Stage explicitly public resources in the build directory, never in the source package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build import build
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist

_ROOT = Path(__file__).resolve().parent

_SKILL_PACKAGE = Path("merlin/development/backend/author_skills/pass-library")
_SKILL_OUTPUT = Path("out/build/author-skills/pass-library")


def generated_pass_library() -> dict[str, bytes]:
    """Render in a checkout; retain the exact generated payload in source releases."""
    generator = _ROOT / "build_tools/scripts/gen_author_pass_library.py"
    if generator.is_file():
        output = _ROOT / _SKILL_OUTPUT
        environment = dict(os.environ, PYTHONPATH=str(_ROOT / "src"))
        subprocess.run(
            [sys.executable, str(generator), "--output", str(output)],
            cwd=_ROOT,
            env=environment,
            check=True,
            text=True,
        )
    else:
        # A source release already carries the original generated build payload.
        # Its pruned test/registry inputs cannot reproduce the checkout renderer.
        output = _ROOT / "src" / _SKILL_PACKAGE
    catalog = json.loads((_ROOT / "src/merlin/resources/optimizations/authoring.json").read_text(encoding="utf-8"))
    facets = {row["facet"] for row in catalog["mechanisms"]}
    if any(not isinstance(facet, str) or Path(facet).name != facet or facet in {".", ".."} for facet in facets):
        raise ValueError("invalid pass-library facet path")
    expected = {"SKILL.md", *(f"reference/{facet}.md" for facet in facets)}
    payload = {}
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("generated pass-library payload may not contain links")
        if path.is_file():
            payload[path.relative_to(output).as_posix()] = path.read_bytes()
    if payload.keys() != expected:
        raise ValueError("generated pass-library payload is missing or has unexpected members")
    return payload


def write_pass_library(directory: Path, payload: dict[str, bytes]) -> None:
    """Stage only the closed renderer payload into a build or source release."""
    for relative, data in payload.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def public_resources() -> list[str]:
    """Reviewed inclusion manifest; untracked build-host files cannot enter a release."""
    manifest = json.loads((_ROOT / "build_tools" / "package_resources.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise ValueError("unsupported public-resource manifest version")
    result = manifest["files"]
    if not isinstance(result, list) or not result or len(result) != len(set(result)):
        raise ValueError("public-resource paths must be a nonempty unique list")
    for name in result:
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or not name.startswith(("src/merlin/resources/", "src/merlin/runtime/native/"))
        ):
            raise ValueError(f"invalid public-resource path: {name}")
        if any(p in {"golden", "hidden"} for p in path.parts) or name.endswith(".hidden.yaml"):
            raise ValueError(f"private resource cannot be packaged: {name}")
        if not (_ROOT / path).is_file():
            raise FileNotFoundError(f"declared public resource is missing: {name}")
    return result


class BuildInOutput(build):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.build_base = "out/build/python"


class MetadataInOutput(egg_info):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.egg_base = "out/build/python/metadata"
        (_ROOT / self.egg_base).mkdir(parents=True, exist_ok=True)


class BuildPyWithData(build_py):
    def run(self) -> None:
        if self.editable_mode:
            return super().run()
        # An older build may have bundled private corpus files. Reusing its build
        # directory must not silently carry them into a new release.
        # These import roots have one distribution owner. Clear each owned root
        # so an interrupted earlier build cannot retain removed workflow code.
        # Retired namespace staging is cleared too; these are not installed packages.
        for package in ("merlin", "merlin_experiments", "merlin_dse"):
            staged = Path(self.build_lib) / package
            if staged.is_symlink() or not staged.resolve().is_relative_to((_ROOT / "out/build").resolve()):
                raise ValueError(f"package staging must be under out/build: {staged}")
            if staged.exists():
                shutil.rmtree(staged)
        payload = generated_pass_library()
        super().run()
        for name in public_resources():
            destination = Path(self.build_lib) / Path(name).relative_to("src")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / name, destination)
        write_pass_library(Path(self.build_lib) / _SKILL_PACKAGE, payload)
        self._skill_outputs = [str(Path(self.build_lib) / _SKILL_PACKAGE / relative) for relative in payload]

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        if self.editable_mode:
            return super().get_outputs(include_bytecode)
        return (
            super().get_outputs(include_bytecode)
            + [str(Path(self.build_lib) / Path(name).relative_to("src")) for name in public_resources()]
            + getattr(self, "_skill_outputs", [])
        )


class EditableWithGeneratedSkills(editable_wheel):
    def run(self) -> None:
        generated_pass_library()
        super().run()


class WheelWithFreshStaging(bdist_wheel):
    def run(self) -> None:
        # An interrupted wheel build leaves install staging behind. Refreshing only build_lib does
        # not remove those files: setuptools archives everything below bdist_dir on the next run.
        staged = Path(self.bdist_dir)
        base = Path(self.get_finalized_command("bdist").bdist_base)
        allowed = (_ROOT / "out/build/python").resolve()
        if (
            staged.is_symlink()
            or base.is_symlink()
            or staged.resolve() != (base / "wheel").resolve()
            or base.parent.resolve() != allowed
            or not base.name.startswith("bdist.")
        ):
            raise ValueError(f"core wheel staging must be its dedicated directory below {allowed}: {staged}")
        if staged.exists():
            shutil.rmtree(staged)
        super().run()


class SourceDistributionInOutput(sdist):
    def make_distribution(self) -> None:
        output = _ROOT / "out/build/python"
        output.mkdir(parents=True, exist_ok=True)
        dist_dir = Path(self.dist_dir).resolve()
        dist_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sdist-", dir=output, delete=not self.keep_temp) as stage:
            name = self.distribution.get_fullname()
            # The same reviewed manifest owns wheel and source resources. A new
            # runtime dependency must not need a second MANIFEST.in declaration
            # before a wheel rebuilt from the source archive can use it.
            public = set(public_resources())
            files = sorted(
                {
                    name
                    for name in self.filelist.files
                    if not name.startswith(("src/merlin/resources/", "src/merlin/runtime/native/"))
                    or name == "src/merlin/resources/__init__.py"
                }
                | public
            )
            release = Path(stage) / name
            self.make_release_tree(str(release), files)
            write_pass_library(release / "src" / _SKILL_PACKAGE, generated_pass_library())
            self.archive_files = [
                self.make_archive(
                    str(dist_dir / name), fmt, root_dir=stage, base_dir=name, owner=self.owner, group=self.group
                )
                for fmt in self.formats
            ]


setup(
    cmdclass={
        "build": BuildInOutput,
        "build_py": BuildPyWithData,
        "bdist_wheel": WheelWithFreshStaging,
        "egg_info": MetadataInOutput,
        "editable_wheel": EditableWithGeneratedSkills,
        "sdist": SourceDistributionInOutput,
    }
)
