"""Additional reviewed source leaves execute in the normal package policy."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import compiler_libraries as L
from merlin_experiments.phase2.campaign import PackageSandboxPolicy
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library, scoped_compiler_library


def selected(tmp_path):
    root = tmp_path / "library"
    (root / "merlin").mkdir(parents=True)
    (root / "merlin/__init__.py").write_text("")
    (root / "merlin/portable.py").write_text("def lower(value):\n return max(-16, min(15, value * 3 + 1))\n")
    (root / "private-answer.txt").write_text("must not enter the compiler namespace")
    contract = freeze_compiler_library(
        root,
        review_id="test-independent-library",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    return CompilerLibrarySelection(contract, root), candidate


def policy(candidate, argv=()):
    return PackageSandboxPolicy(
        tuple(argv),
        (),
        (),
        candidate.parent / "scratch",
        candidate,
        SimpleNamespace(),
        env_prefix="export OMP_NUM_THREADS=1; ",
    )


def test_default_unchanged_and_exact_member_mounts_are_reopened(tmp_path):
    selection, candidate = selected(tmp_path)
    original = policy(candidate, ("box",))
    assert L.apply(original) is original
    with scoped_compiler_library(selection):
        result = L.apply(original)
        assert len([part for part in result.argv if part == "--ro-bind"]) == 2
        assert str(selection.root) not in result.argv
        assert str(selection.root / "private-answer.txt") not in result.argv
        assert str(L.guest_root(selection)) in result.env_prefix
        assert L.apply(result) == result
        suffix = replace(result, argv=(*result.argv, "--ro-bind", "/input", "/input"))
        assert L.apply(suffix) == suffix
        with pytest.raises(StageGateError, match="changed"):
            L.apply(replace(result, argv=(*result.argv[:-1], "/wrong")))
        with pytest.raises(StageGateError, match="foreign or overriding"):
            L.apply(
                replace(
                    result,
                    argv=(*result.argv, "--ro-bind", "/other", str(L.guest_root(selection) / "merlin/portable.py")),
                )
            )
        with pytest.raises(StageGateError, match="foreign or overriding"):
            L.apply(replace(original, argv=(*original.argv, "--dir", "/perf-compiler-library/other/python")))


def test_candidate_owned_library_refuses(tmp_path):
    selection, _candidate = selected(tmp_path)
    with scoped_compiler_library(selection):
        with pytest.raises(StageGateError, match="candidate-owned"):
            L.apply(policy(selection.root))


def test_real_box_runs_selected_library_with_original_leaves_and_hides_siblings(tmp_path, monkeypatch):
    binary = shutil.which("bwrap")
    python = Path("/usr/bin/python3")
    if binary is None or not python.is_file():
        pytest.skip("requires selected local bwrap and system Python")
    selection, candidate = selected(tmp_path)
    previous = tmp_path / "original.py"
    previous.write_text("def adjust(value):\n return value - 2\n")
    monkeypatch.setattr(L, "_original_sources", lambda owner: {"merlin/previous.py": previous})
    (candidate / "compiler.py").write_text(
        "import json,os\nfrom merlin.portable import lower\nfrom merlin.previous import adjust\n"
        f"assert not os.path.exists({str(selection.root / 'private-answer.txt')!r})\n"
        "print(json.dumps([lower(adjust(x)) for x in range(-19,20)]))\n"
    )
    argv = [binary, "--unshare-all", "--die-with-parent", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for name in ("/lib", "/lib64"):
        if Path(name).exists():
            argv += ["--ro-bind", name, name]
    argv += ["--tmpfs", "/tmp", "--ro-bind", str(candidate), "/candidate"]
    with scoped_compiler_library(selection):
        boxed = L.apply(policy(candidate, argv))
        with shared_compute_pool(timeout_s=30):
            result = subprocess.run(
                [*boxed.argv, "bash", "-c", boxed.env_prefix + "exec /usr/bin/python3 -B /candidate/compiler.py"],
                capture_output=True,
                text=True,
                timeout=15,
            )
    if result.returncode and "Operation not permitted" in result.stderr:
        pytest.skip("kernel user namespaces unavailable for real local sandbox check")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [max(-16, min(15, (x - 2) * 3 + 1)) for x in range(-19, 20)]


def test_real_box_v2_startup_uses_admitted_union_without_host_aliases(tmp_path, monkeypatch):
    from merlin.common.paths import module_source_path

    binary = shutil.which("bwrap")
    python = Path("/usr/bin/python3")
    if binary is None or not python.is_file():
        pytest.skip("requires selected local bwrap and system Python")
    selection, candidate = selected(tmp_path)
    previous = tmp_path / "original.py"
    previous.write_text("def adjust(value):\n return value - 2\n")
    monkeypatch.setattr(L, "_original_sources", lambda owner: {"merlin/previous.py": previous})
    shutil.copyfile(module_source_path("merlin.targetgen.selected_library_imports"), candidate / "startup.py")
    (candidate / "selection.json").write_text(json.dumps(selection.record()))
    (candidate / "compiler.py").write_text(
        "import json,os\nfrom pathlib import Path\nfrom startup import activate_selected_sources_v2\n"
        f"assert not os.path.exists({str(selection.root)!r})\n"
        "selection=json.loads(Path('/candidate/selection.json').read_text())\n"
        f"report=activate_selected_sources_v2(selection, original_host_roots=({str(tmp_path / 'unmounted-host')!r},))\n"
        "from merlin.portable import lower\nfrom merlin.previous import adjust\n"
        f"assert not os.path.exists({str(selection.root / 'private-answer.txt')!r})\n"
        "print(json.dumps({'report':report,'result':[lower(adjust(x)) for x in range(-19,20)]}))\n"
    )
    argv = [binary, "--unshare-all", "--die-with-parent", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for name in ("/lib", "/lib64"):
        if Path(name).exists():
            argv += ["--ro-bind", name, name]
    argv += ["--tmpfs", "/tmp", "--ro-bind", str(candidate), "/candidate"]
    with scoped_compiler_library(selection):
        boxed = L.apply(policy(candidate, argv))
        with shared_compute_pool(timeout_s=30):
            result = subprocess.run(
                [*boxed.argv, "bash", "-c", boxed.env_prefix + "exec /usr/bin/python3 -B /candidate/compiler.py"],
                capture_output=True,
                text=True,
                timeout=15,
            )
    if result.returncode and "Operation not permitted" in result.stderr:
        pytest.skip("kernel user namespaces unavailable for real local sandbox check")
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["result"] == [max(-16, min(15, (x - 2) * 3 + 1)) for x in range(-19, 20)]
    assert value["report"]["root"] == str(L.guest_root(selection))
    assert value["report"]["schema"] == "merlin.selected_compiler_source_startup.v2"
    assert value["report"]["contract_sha256"] == selection.contract.sha256
    assert str(selection.root / "private-answer.txt") not in boxed.argv


def test_mount_operation_reopens_boundaries_without_redundant_formatting_checks(tmp_path, monkeypatch):
    selection, candidate = selected(tmp_path)
    original_record = CompilerLibrarySelection.record
    calls = []

    def count(owner):
        calls.append(owner)
        return original_record(owner)

    monkeypatch.setattr(CompilerLibrarySelection, 'record', count)
    with scoped_compiler_library(selection):
        calls.clear()
        result = L.apply(policy(candidate))
        assert calls == [selection, selection, selection]
        assert f'/perf-compiler-library/{selection.contract.sha256}/python' in result.env_prefix
        calls.clear()
        assert L.apply(result) == result
        assert calls == [selection, selection, selection]


def test_mount_operation_rejects_mutation_between_original_boundaries(tmp_path, monkeypatch):
    from merlin.targetgen.compiler_library import CompilerLibraryError

    selection, candidate = selected(tmp_path)
    source = selection.root / 'merlin/portable.py'
    original = source.read_bytes()

    def mutate(_owner):
        source.write_text('def lower(value):\n return 999\n')
        return {}

    with scoped_compiler_library(selection):
        monkeypatch.setattr(L, '_original_sources', mutate)
        try:
            with pytest.raises(CompilerLibraryError, match='bytes changed'):
                L.apply(policy(candidate))
        finally:
            source.write_bytes(original)
