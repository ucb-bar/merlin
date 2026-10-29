"""Seed preflight compiles with only the Merlin import roots a candidate sandbox grants."""

import hashlib
import importlib.util
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent

TOOL = """#!/usr/bin/env python3
import sys
from pathlib import Path
PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE / "compiler_library")]
from merlin.seed_probe_member import VALUE
print(VALUE)
"""


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("seed_compile_probe_test", HERE / "seed_compile_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package(root: Path, *, member: bool) -> SimpleNamespace:
    """A staged submission whose tool imports one compiler_library member (a namespace portion)."""
    library = root / "submission/compiler_library/merlin"
    library.mkdir(parents=True)
    if member:
        (library / "seed_probe_member.py").write_text("VALUE = 'staged'\n")
    tool = root / "submission/opt"
    tool.write_text(TOOL)
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return SimpleNamespace(directory=root / "submission", tool=tool, language="python",
                           manifest={"commands": {"parse": {"argv": ["{tool}", "{input_mlir}"]}}})


def _foreign_root(root: Path) -> Path:
    """A Merlin source root (an export's src/) that provides the member the staged compiler lacks."""
    foreign = root / "export-src"
    (foreign / "merlin").mkdir(parents=True)
    (foreign / "merlin/__init__.py").write_text("from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n")
    (foreign / "merlin/seed_probe_member.py").write_text("VALUE = 'export'\n")
    return foreign


def _run(probe, pkg, granted, tmp_path, monkeypatch, *, ambient=None):
    if ambient is not None:
        monkeypatch.setenv("PYTHONPATH", str(ambient))
    source = tmp_path / "case.mlir"
    source.write_text("module {}\n")
    return probe.invoker(granted)(pkg, "parse", source)


def test_staged_member_imports(probe, tmp_path, monkeypatch):
    pkg = _package(tmp_path, member=True)
    done = _run(probe, pkg, probe.granted_sources({}, tmp_path, tmp_path / "granted"), tmp_path, monkeypatch,
                ambient=_foreign_root(tmp_path))
    assert done.returncode == 0 and done.stdout.strip() == "staged"


def test_a_member_missing_from_the_staged_compiler_is_not_masked_by_the_export(probe, tmp_path, monkeypatch):
    pkg = _package(tmp_path, member=False)
    done = _run(probe, pkg, probe.granted_sources({}, tmp_path, tmp_path / "granted"), tmp_path, monkeypatch,
                ambient=_foreign_root(tmp_path))
    assert done.returncode != 0 and "ModuleNotFoundError" in done.stderr


def test_a_member_the_bundle_grants_publicly_imports(probe, tmp_path, monkeypatch):
    pkg = _package(tmp_path, member=False)
    repo = tmp_path / "repo"
    (repo / "src/merlin").mkdir(parents=True)
    granted_files = {"merlin/__init__.py": "", "merlin/seed_probe_member.py": "VALUE = 'granted'\n"}
    rows = []
    for relative, text in granted_files.items():
        (repo / "src" / relative).write_text(text)
        rows.append({"path": "src/" + relative, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    bundle = {"allowed": [{"path": str(repo / row["path"]), "mode": "ro"} for row in rows],
              "compiler_tool_profile": {"common_sources": [
                  dict(row, path="merlin/python/" + row["path"].removeprefix("src/")) for row in rows]}}
    granted = probe.granted_sources(bundle, repo, tmp_path / "granted")
    done = _run(probe, pkg, granted, tmp_path, monkeypatch, ambient=_foreign_root(tmp_path))
    assert done.returncode == 0 and done.stdout.strip() == "granted"
    (repo / "src/merlin/seed_probe_member.py").write_text("VALUE = 'changed'\n")
    with pytest.raises(probe.SeedCompileError, match="differs from the bundle pin"):
        probe.granted_sources(bundle, repo, tmp_path / "granted-again")
    bundle["allowed"] = bundle["allowed"][:1]
    with pytest.raises(probe.SeedCompileError, match="no unique bundle grant"):
        probe.granted_sources(bundle, repo, tmp_path / "granted-third")


def test_one_compiled_case_passes_and_none_refuses(probe, tmp_path, monkeypatch):
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import package_runtime as P

    monkeypatch.setattr(P, "load_package", lambda submission, contract=None: SimpleNamespace(directory=submission))
    monkeypatch.setattr(P, "build_package", lambda pkg: None)
    cases = [{"name": name, "interface": tmp_path / f"{name}.mlir"} for name in ("refusal", "dense")]

    def lower(pkg, interface, generated, **kw):
        if interface.stem == "refusal":
            raise ValueError("declined shape")
        return {"commands": [1]}, "llvm"

    monkeypatch.setattr(CC, "lower_interface", lower)
    result = probe.compile_one(tmp_path, cases, granted=tmp_path, work=tmp_path / "work")
    assert result["status"] == "compiled" and result["case"] == "dense" and len(result["refused_first"]) == 1
    monkeypatch.setattr(CC, "lower_interface", lambda *a, **kw: (_ for _ in ()).throw(
        ModuleNotFoundError("No module named 'merlin.targetgen.oot_starterkit.iface'")))
    with pytest.raises(probe.SeedCompileError, match="oot_starterkit"):
        probe.compile_one(tmp_path, cases, granted=tmp_path, work=tmp_path / "work2")


def test_smallest_public_cases_come_from_the_sealed_corpus(probe, tmp_path, monkeypatch):
    from merlin.targetgen import capsule_common as CC

    rows = []
    for name, size, label in (("big", 90, "public"), ("small", 10, "public"), ("hidden", 1, "hidden")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "capsule.interface.mlir").write_text("x" * size)
        rows.append({"name": name, "__dir__": str(directory), "label": label})
    monkeypatch.setattr(CC, "discover_capsules", lambda roots, labels=None, contract=None: [
        row for row in rows if labels is None or row["label"] in labels])
    assert [case["name"] for case in probe.smallest_public_cases([tmp_path], count=2)] == ["small", "big"]
    monkeypatch.setattr(CC, "discover_capsules", lambda *a, **kw: [])
    with pytest.raises(probe.SeedCompileError, match="no public interface"):
        probe.smallest_public_cases([tmp_path])
