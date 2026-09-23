"""Release ownership checks fail on real overlapping wheel archives."""

from __future__ import annotations

import importlib.util
import tomllib
import zipfile

import pytest

from merlin.common.paths import repo_root


def _gate():
    path = repo_root() / "build_tools/scripts/check_distribution_layout.py"
    spec = importlib.util.spec_from_file_location("distribution_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(tmp_path, distribution, files):
    path = tmp_path / f"{distribution}-0.1-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def test_duplicate_namespace_files_are_rejected(tmp_path):
    core = _wheel(tmp_path, "merlin", {"merlin/__init__.py": ""})
    extension = _wheel(tmp_path, "merlin_dse", {"merlin/__init__.py": ""})
    errors = _gate().audit([core, extension])
    assert any("duplicate file" in error for error in errors)
    assert any("overwrites core namespace" in error for error in errors)


def test_moved_modules_cannot_survive_a_stale_core_build(tmp_path):
    core = _wheel(tmp_path, "merlin", {"merlin/dse/cli.py": ""})
    assert any("research shipped in core" in error for error in _gate().audit([core]))


def test_evaluator_cannot_survive_a_stale_core_build(tmp_path):
    core = _wheel(tmp_path, "merlin", {"merlin/targetgen/capsule_runner.py": ""})
    assert any("research shipped in core" in error for error in _gate().audit([core]))


@pytest.mark.parametrize(
    "module",
    [
        "merlin/agentreport/tokens.py",
        "merlin/verify/plots.py",
        "merlin/verify/replay.py",
        "merlin/verify/replay_layers.py",
        "merlin/perf/recovery.py",
        "merlin/perf/source_program_pair_provider.py",
        "merlin/kernels/ceiling_drivers/run_expert_gemm.py",
        "merlin/kernels/ceiling_drivers/multishape_compare.py",
        "merlin/targetgen/agent/claude_cli.py",
        "merlin/targetgen/aet_bridge.py",
        "merlin/targetgen/experiment_tokens.py",
        "merlin/targetgen/heavy_oracles.py",
        "merlin/targetgen/model_slice_export.py",
        "merlin/targetgen/group_capsules.py",
        "merlin/targetgen/store_probe.py",
        "merlin/targetgen/evaluation_cohort.py",
        "merlin/targetgen/numeric_falsifiability.py",
        "merlin/targetgen/rtl/gen_rocc_replay.py",
    ],
)
def test_research_adapters_cannot_survive_a_stale_core_build(tmp_path, module):
    core = _wheel(tmp_path, "merlin", {module: ""})
    assert any("research shipped in core" in error for error in _gate().audit([core]))


def test_console_script_has_one_owner(tmp_path):
    wheels = [
        _wheel(
            tmp_path,
            dist,
            {f"{dist}.dist-info/entry_points.txt": "[console_scripts]\nmerlin-study = merlin.study:main\n"},
        )
        for dist in ("merlin", "merlin_dse")
    ]
    assert any("duplicate command" in error for error in _gate().audit(wheels))


def test_disjoint_shared_namespace_contributions_are_valid(tmp_path):
    core = _wheel(tmp_path, "merlin", {"merlin/__init__.py": "", "merlin/common/paths.py": ""})
    extension = _wheel(tmp_path, "merlin_dse", {"merlin/dse/cli.py": ""})
    assert _gate().audit([core, extension]) == []


def test_source_parity_checks_actual_wheel_bytes_and_owner(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n')
    source = tmp_path / "src/merlin/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    wheel = _wheel(tmp_path, "merlin", {"merlin/example.py": source.read_bytes()})
    assert _gate().audit_sources([wheel], tmp_path) == []
    source.write_text("VALUE = 2\n")
    assert _gate().audit_sources([wheel], tmp_path) == [
        "Python member differs from canonical source: merlin:merlin/example.py"
    ]
    wheel = _wheel(tmp_path, "merlin", {"merlin/removed.py": ""})
    assert _gate().audit_sources([wheel], tmp_path) == [
        "required Python member missing: merlin:merlin/example.py",
        "Python member has no canonical source: merlin:merlin/removed.py",
    ]


def test_source_parity_resolves_optional_owner_and_refuses_unknown_distribution(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n')
    (tmp_path / "src").mkdir()
    extension = tmp_path / "packages/merlin-analysis"
    extension.mkdir(parents=True)
    (extension / "pyproject.toml").write_text('[project]\nname = "merlin-analysis"\n')
    source = extension / "src/merlin/verify/replay.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic analysis\n")
    wheel = _wheel(tmp_path, "merlin_analysis", {"merlin/verify/replay.py": source.read_bytes()})
    assert _gate().audit_sources([wheel], tmp_path) == []
    unknown = _wheel(tmp_path, "unrelated", {"unrelated/example.py": ""})
    assert _gate().audit_sources([unknown], tmp_path) == ["unknown source distribution: unrelated"]


def test_source_parity_refuses_escaping_archive_members(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n')
    (tmp_path / "src").mkdir()
    wheel = _wheel(tmp_path, "merlin", {"../outside.py": "", "/absolute.py": ""})
    assert _gate().audit_sources([wheel], tmp_path) == [
        "escaping Python member: merlin:../outside.py",
        "escaping Python member: merlin:/absolute.py",
    ]


@pytest.mark.parametrize("namespace", ["verify", "perf", "kernels", "kernels/ceiling_drivers", "targetgen/rtl"])
def test_analysis_cannot_replace_core_namespace_initializers(tmp_path, namespace):
    extension = _wheel(tmp_path, "merlin_analysis", {f"merlin/{namespace}/__init__.py": ""})
    assert any("overwrites core namespace" in error for error in _gate().audit([extension]))


@pytest.mark.parametrize("shipped", [{}, {"merlin/__init__.py": ""}])
def test_empty_and_partial_wheels_refuse_missing_canonical_modules(tmp_path, shipped):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n')
    source = tmp_path / "src/merlin"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "required.py").write_text("REQUIRED = True\n")
    errors = _gate().audit_sources([_wheel(tmp_path, "merlin", shipped)], tmp_path)
    assert "required Python member missing: merlin:merlin/required.py" in errors
    assert len(errors) == 2 - len(shipped)


def test_unreadable_source_inventory_cannot_validate_an_empty_wheel(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n')
    (tmp_path / "src").mkdir()
    gate = _gate()

    def unreadable(source, *, onerror):
        onerror(PermissionError("source directory unreadable"))

    monkeypatch.setattr(gate.os, "walk", unreadable)
    assert gate.audit_sources([_wheel(tmp_path, "merlin", {})], tmp_path) == [
        "cannot inventory source distribution: merlin: source directory unreadable"
    ]


def test_inventory_respects_includes_excludes_and_namespace_initializers(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "merlin-analysis"\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\n'
        'include = ["merlin.verify*"]\nexclude = ["merlin.verify.private*"]\nnamespaces = true\n'
    )
    files = {
        "merlin/__init__.py": "# owned by core, not selected\n",
        "merlin/verify/replay.py": "# required namespace contribution\n",
        "merlin/verify/private/secret.py": "# excluded\n",
        "merlin/other/ignored.py": "# not included\n",
    }
    for relative, contents in files.items():
        path = tmp_path / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    selected = {"merlin/verify/replay.py": files["merlin/verify/replay.py"]}
    gate = _gate()
    assert gate.audit_sources([_wheel(tmp_path, "merlin_analysis", selected)], tmp_path) == []
    assert gate.audit_sources([_wheel(tmp_path, "merlin_analysis", {})], tmp_path) == [
        "required Python member missing: merlin_analysis:merlin/verify/replay.py"
    ]
    selected["merlin/__init__.py"] = files["merlin/__init__.py"]
    assert gate.audit_sources([_wheel(tmp_path, "merlin_analysis", selected)], tmp_path) == [
        "Python member excluded by package configuration: merlin_analysis:merlin/__init__.py"
    ]


def test_non_namespace_discovery_requires_initializers(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "merlin"\n[tool.setuptools.packages.find]\nwhere = ["src"]\nnamespaces = false\n'
    )
    for relative in ("merlin/__init__.py", "merlin/regular/__init__.py", "merlin/implicit/module.py"):
        path = tmp_path / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    wheel = _wheel(tmp_path, "merlin", {"merlin/__init__.py": "", "merlin/regular/__init__.py": ""})
    assert _gate().audit_sources([wheel], tmp_path) == []


def test_actual_project_inventory_matches_setuptools_discovery():
    setuptools = pytest.importorskip("setuptools")
    root = repo_root()
    projects = [root / "pyproject.toml", *sorted((root / "packages").glob("*/pyproject.toml"))]
    for project in projects:
        metadata = tomllib.loads(project.read_text())
        config = metadata["tool"]["setuptools"]["packages"]["find"]
        source = project.parent / "src"
        finder = setuptools.find_namespace_packages if config.get("namespaces", True) else setuptools.find_packages
        packages = finder(where=str(source), include=config.get("include", ["*"]), exclude=config.get("exclude", []))
        expected = {
            module.relative_to(source).as_posix(): module
            for package in packages
            for module in (source / package.replace(".", "/")).glob("*.py")
            if module.is_file()
        }
        assert expected, project
        assert _gate()._python_sources(project, metadata) == expected


def test_inventory_refuses_unsupported_packaging_layout(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n[tool.setuptools]\npackages = ["merlin"]\n')
    errors = _gate().audit_sources([_wheel(tmp_path, "merlin", {})], tmp_path)
    assert errors == ["cannot inventory source distribution: merlin: requires setuptools packages.find"]


def test_missing_source_root_cannot_validate_empty_wheel(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n')
    assert _gate().audit_sources([_wheel(tmp_path, "merlin", {})], tmp_path) == [
        "cannot inventory source distribution: merlin: source root must be an existing unlinked directory"
    ]


@pytest.mark.parametrize("link_kind", ["cycle", "escape", "module"])
def test_linked_sources_refuse_without_traversal(tmp_path, link_kind):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "merlin"\n')
    package = tmp_path / "src/merlin"
    package.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    if link_kind == "module":
        (outside / "module.py").write_text("")
        (package / "linked.py").symlink_to(outside / "module.py")
    else:
        (package / "linked").symlink_to(package if link_kind == "cycle" else outside, target_is_directory=True)
    errors = _gate().audit_sources([_wheel(tmp_path, "merlin", {})], tmp_path)
    assert len(errors) == 1
    assert "linked" in errors[0]


@pytest.mark.parametrize(
    "configuration",
    ['find = "src"', 'find.include = "merlin*"', "find.exclude = [42]", 'find.namespaces = "false"'],
)
def test_invalid_discovery_configuration_refuses(tmp_path, configuration):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "merlin"\n[tool.setuptools.packages]\n' + configuration + "\n"
    )
    errors = _gate().audit_sources([_wheel(tmp_path, "merlin", {})], tmp_path)
    assert len(errors) == 1
    assert errors[0].startswith("cannot inventory source distribution:")
