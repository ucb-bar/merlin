"""Exercise shared build commands without requiring an external frontend or a network."""

from __future__ import annotations

import runpy
import shutil
import tarfile
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest

from merlin.common.paths import repo_root

setuptools = pytest.importorskip("setuptools")


def _project(tmp_path, monkeypatch, *, workspace: bool, core: bool = False):
    # pytest may immediately reuse a cleaned passing tmp_path. setuptools caches mkdir calls for
    # the interpreter lifetime; distinct synthetic projects must not inherit that cache identity.
    tmp_path = tmp_path / uuid4().hex
    project = tmp_path / "packages" / "merlin-example" if workspace else tmp_path / "merlin-example"
    project.mkdir(parents=True)
    helper = repo_root() / ("setup.py" if core else "build_tools/extension_setup.py")
    if workspace:
        shared = tmp_path / "build_tools" / "extension_setup.py"
        shared.parent.mkdir()
        shutil.copyfile(helper, shared)
        (project / "setup.py").symlink_to("../../build_tools/extension_setup.py")
    else:
        shutil.copyfile(helper, project / "setup.py")
    if core:
        resources = project / "build_tools" / "package_resources.json"
        resources.parent.mkdir()
        resources.write_text('{"version": 1, "files": []}')
    (project / "pyproject.toml").write_text('[project]\nname = "merlin-example"\nversion = "0.1.0"\n')
    (project / "README.md").write_text("Build routing fixture.\n")
    package = project / "src" / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('VALUE = "example"\n')
    captured = {}
    monkeypatch.chdir(project)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    namespace = runpy.run_path(str(project / "setup.py"))
    distribution = setuptools.Distribution(
        {
            "name": "merlin-example",
            "version": "0.1.0",
            "packages": ["example"],
            "package_dir": {"": "src"},
            "cmdclass": captured["cmdclass"],
        }
    )
    distribution.script_name = "setup.py"
    output = (tmp_path if workspace else project) / "out" / "build" / "python"
    if not core:
        output /= "merlin-example"
    return project, output, namespace, distribution


@pytest.mark.parametrize("workspace", [True, False], ids=["workspace", "standalone-sdist"])
def test_build_and_metadata_stay_in_managed_output(tmp_path, monkeypatch, workspace):
    project, output, _, distribution = _project(tmp_path, monkeypatch, workspace=workspace)
    distribution.run_command("build")
    distribution.run_command("egg_info")
    assert Path(distribution.get_command_obj("build").build_base) == output / "build"
    metadata = Path(distribution.get_command_obj("egg_info").egg_info)
    assert metadata.is_relative_to(output / "metadata")
    assert (metadata / "PKG-INFO").is_file()
    assert not (project / "build").exists()
    assert not list((project / "src").glob("*.egg-info"))
    sources = (metadata / "SOURCES.txt").read_text().splitlines()
    assert "src/example/__init__.py" in sources
    assert all(not Path(name).is_absolute() for name in sources)
    assert all(".egg-info" not in name and not name.startswith("out/") for name in sources)


def test_source_manifest_excludes_external_and_generated_paths(tmp_path, monkeypatch):
    project, output, namespace, _ = _project(tmp_path, monkeypatch, workspace=True)
    names = [
        "setup.py",
        "src/example/__init__.py",
        str(project / "src" / "example" / "__init__.py"),
        str(output / "metadata" / "merlin_example.egg-info" / "PKG-INFO"),
        "src/stale.egg-info/PKG-INFO",
        "src/example/__pycache__/stale.pyc",
        "../external.py",
    ]
    assert namespace["_source_files"](names) == ["setup.py", "src/example/__init__.py"]


def test_rebuild_drops_stale_staging_but_preserves_other_outputs(tmp_path, monkeypatch):
    _, output, _, distribution = _project(tmp_path, monkeypatch, workspace=True)
    build = distribution.get_command_obj("build_py")
    build.ensure_finalized()
    stale = Path(build.build_lib) / "removed" / "module.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("obsolete = True\n")
    evidence = output / "dist" / "prior-release.whl"
    evidence.parent.mkdir()
    evidence.write_bytes(b"prior build evidence")
    build.run()
    assert not stale.exists()
    assert (Path(build.build_lib) / "example" / "__init__.py").is_file()
    assert evidence.read_bytes() == b"prior build evidence"


@pytest.mark.parametrize("location", ["source", "build-root", "symlink"])
def test_build_refuses_to_clear_unmanaged_or_symlink_staging(tmp_path, monkeypatch, location):
    project, output, _, distribution = _project(tmp_path, monkeypatch, workspace=True)
    build = distribution.get_command_obj("build_py")
    build.ensure_finalized()
    preserved = project / "src" / "example" / "__init__.py"
    if location == "source":
        build.build_lib = str(project / "src")
    elif location == "build-root":
        build.build_lib = str(output / "build")
    else:
        staged = Path(build.build_lib)
        staged.parent.mkdir(parents=True)
        staged.symlink_to(project / "src", target_is_directory=True)
    with pytest.raises(ValueError, match="extension package staging"):
        build.run()
    assert preserved.read_text() == 'VALUE = "example"\n'


@pytest.mark.parametrize("core", [False, True], ids=["extension", "core"])
def test_sdist_materializes_hook_and_does_not_stage_in_source(tmp_path, monkeypatch, core):
    project, output, _, distribution = _project(tmp_path, monkeypatch, workspace=not core, core=core)
    sdist = distribution.get_command_obj("sdist")
    sdist.dist_dir = str(output / "dist")
    sdist.formats = ["gztar"]
    release_tree = sdist.make_release_tree
    stages = []

    def capture_stage(base_dir, files):
        stages.append(Path(base_dir))
        return release_tree(base_dir, files)

    monkeypatch.setattr(sdist, "make_release_tree", capture_stage)
    distribution.run_command("sdist")
    assert (project / "setup.py").is_symlink() is not core
    assert stages and all(stage.is_relative_to(output) for stage in stages)
    assert not (project / distribution.get_fullname()).exists()
    with tarfile.open(sdist.archive_files[0]) as archive:
        scripts = [member for member in archive.getmembers() if member.name.endswith("/setup.py")]
        assert len(scripts) == 1
        assert scripts[0].isfile() and not scripts[0].issym() and not scripts[0].islnk()
        assert archive.extractfile(scripts[0]).read() == (project / "setup.py").read_bytes()
        assert all(not Path(member.name).is_absolute() and ".." not in Path(member.name).parts for member in archive)


@pytest.mark.parametrize("core", [False, True], ids=["extension", "core"])
def test_failed_sdist_leaves_no_generated_source_tree(tmp_path, monkeypatch, core):
    project, output, _, distribution = _project(tmp_path, monkeypatch, workspace=False, core=core)
    sdist = distribution.get_command_obj("sdist")
    sdist.dist_dir = str(output / "dist")
    stages = []

    def fail_after_staging(base_dir, files):
        stage = Path(base_dir)
        stage.mkdir()
        (stage / "partial.py").write_text("partial = True\n")
        stages.append(stage)
        raise RuntimeError("interrupted fixture build")

    monkeypatch.setattr(sdist, "make_release_tree", fail_after_staging)
    with pytest.raises(RuntimeError, match="interrupted fixture build"):
        distribution.run_command("sdist")
    assert stages and all(stage.is_relative_to(output) and not stage.exists() for stage in stages)
    assert not (project / distribution.get_fullname()).exists()


@pytest.mark.parametrize("core", [False, True], ids=["extension", "core"])
@pytest.mark.parametrize("protected", [True, False], ids=["fresh-staging", "legacy-negative-control"])
def test_direct_wheel_rebuild_drops_interrupted_private_staging(tmp_path, monkeypatch, core, protected):
    _, output, _, distribution = _project(tmp_path, monkeypatch, workspace=False, core=core)
    wheel = distribution.get_command_obj("bdist_wheel")
    wheel.dist_dir = str(output / "dist")
    wheel.ensure_finalized()
    stale = Path(wheel.bdist_dir) / "merlin" / "removed_private_grader.py"
    stale.parent.mkdir(parents=True)
    stale.write_text('ANSWER = "must never ship"\n')
    if protected:
        wheel.run()
    else:
        super(type(wheel), wheel).run()
    with zipfile.ZipFile(next((output / "dist").glob("*.whl"))) as archive:
        assert "example/__init__.py" in archive.namelist()
        assert ("merlin/removed_private_grader.py" in archive.namelist()) is not protected
        assert any(b"must never ship" in archive.read(name) for name in archive.namelist()) is not protected


@pytest.mark.parametrize("core", [False, True], ids=["extension", "core"])
@pytest.mark.parametrize("location", ["source", "output-root", "symlink"])
def test_wheel_refuses_escaping_or_symlink_staging(tmp_path, monkeypatch, core, location):
    project, output, _, distribution = _project(tmp_path, monkeypatch, workspace=False, core=core)
    wheel = distribution.get_command_obj("bdist_wheel")
    wheel.ensure_finalized()
    if location == "source":
        wheel.bdist_dir = str(project / "src")
    elif location == "output-root":
        wheel.bdist_dir = str(output)
    else:
        staged = Path(wheel.bdist_dir)
        staged.parent.mkdir(parents=True)
        staged.symlink_to(project / "src", target_is_directory=True)
    with pytest.raises(ValueError, match="wheel staging"):
        wheel.run()
    assert (project / "src" / "example" / "__init__.py").read_text() == 'VALUE = "example"\n'


@pytest.mark.parametrize("name", ["..", "/escape", "name/escape", "-name", "name-"])
def test_distribution_names_cannot_escape_output_root(tmp_path, monkeypatch, name):
    project, _, _, _ = _project(tmp_path, monkeypatch, workspace=False)
    (project / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.1.0"\n')
    with pytest.raises(ValueError, match="invalid extension distribution name"):
        runpy.run_path(str(project / "setup.py"))


def test_all_optional_distributions_share_one_hook():
    root = repo_root()
    for name in ("analysis", "dse", "experiments", "mining"):
        script = root / "packages" / f"merlin-{name}" / "setup.py"
        assert script.is_symlink()
        assert script.resolve() == root / "build_tools" / "extension_setup.py"
