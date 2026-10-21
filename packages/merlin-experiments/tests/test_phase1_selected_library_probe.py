"""Selected public imports are checked in the actual guest namespace."""

from __future__ import annotations

import hashlib
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import canary
from merlin_experiments.phase1 import tooling_readiness as R
from merlin_experiments.phase2 import compiler_libraries as L
from merlin_experiments.phase2 import compiler_library_state as S
from test_compiler_libraries import policy

from merlin.targetgen.compiler_library import (
    CompilerLibraryError,
    CompilerLibrarySelection,
    freeze_compiler_library,
    scoped_compiler_library,
)


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "reviewed"
    files = {
        "merlin/__init__.py": "",
        "merlin/alpha.py": "def select(value):\n return value + 1\n",
        "merlin/zeta.py": "def arrange(value):\n return value[::-1]\n",
        "merlin/_detail.py": "raise AssertionError('nonpublic module was imported')\n",
    }
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    contract = freeze_compiler_library(
        root,
        review_id="selected-import-control",
        public_modules=("merlin.alpha", "merlin.zeta"),
        sources=tuple(
            (path, "merlin" if path.endswith("__init__.py") else path[:-3].replace("/", ".")) for path in files
        ),
    )
    host = CompilerLibrarySelection(contract, root)
    binding = S.freeze_selection(host, tmp_path / "frozen/python")
    return S.verify_binding(binding, host)


def test_no_selection_keeps_original_probe():
    assert R._selected_library_probe() == ""
    assert "SELECTED_PUBLIC_LIBRARY_IMPORTS_OK" not in R._sandbox_probe("synthetic", (), "")


def test_probe_uses_exact_frozen_public_roster_without_host_or_private_paths(tmp_path, monkeypatch, library):
    monkeypatch.setattr(R.BW, "frozen_selected_rtl_facts", lambda *a, **kw: None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = SimpleNamespace(target="synthetic", repo=tmp_path)
    with scoped_compiler_library(library):
        with R.public_probe_session(context, workspace, {}, ()) as script:
            assert "SELECTED_PUBLIC_LIBRARY_IMPORTS_OK" in script
        for name in library.contract.public_modules:
            assert repr(name) in script
        assert library.contract.sha256 in script
        assert str(L.guest_root(library)) in script
    assert "merlin._detail" not in script and str(library.root) not in script
    assert "sys.path.insert" not in script and "sys.path[:0]" not in script
    compile(script, "<selected-public-library>", "exec")


def test_changed_library_bytes_refuse_probe_generation(library):
    leaf = library.root / "merlin/alpha.py"
    original = leaf.read_bytes()
    with scoped_compiler_library(library):
        try:
            leaf.chmod(0o644)
            leaf.write_text("changed = True\n")
            with pytest.raises(CompilerLibraryError, match="bytes changed"):
                R._selected_library_probe()
        finally:
            leaf.write_bytes(original)
            leaf.chmod(0o444)


@pytest.mark.parametrize("failure", [None, "missing_last_public", "wrong_bytes", "wrong_origin"])
@pytest.mark.parametrize("execution", ["readiness", "isolated_canary"])
def test_real_guest_imports_all_selected_public_modules_and_checks_origin(
    tmp_path, monkeypatch, library, failure, execution
):
    binary = shutil.which("bwrap")
    if binary is None or not Path("/usr/bin/python3").is_file():
        pytest.skip("requires local bwrap and Python")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    monkeypatch.setattr(L, "_original_sources", lambda owner: {})
    argv = [binary, "--unshare-all", "--die-with-parent", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for path in ("/lib", "/lib64"):
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    argv += ["--tmpfs", "/tmp", "--bind", str(candidate), "/candidate", "--chdir", "/candidate"]
    with scoped_compiler_library(library):
        script = R._sandbox_probe("synthetic", (), "")
        selected = L.apply(policy(candidate, argv))
        guest = L.guest_root(library)
    argv = list(selected.argv)
    environment = selected.env_prefix
    if failure == "missing_last_public":
        position = argv.index(str(library.root / "merlin/zeta.py"))
        del argv[position - 1 : position + 2]
    elif failure == "wrong_bytes":
        changed = candidate / "changed.py"
        changed.write_text("def select(value):\n return value - 1\n")
        position = argv.index(str(library.root / "merlin/alpha.py"))
        argv[position] = str(changed)
    elif failure == "wrong_origin":
        shadow = candidate / "shadow/merlin"
        shadow.mkdir(parents=True)
        for relative in ("__init__.py", "alpha.py", "zeta.py"):
            shutil.copyfile(library.root / "merlin" / relative, shadow / relative)
        environment += "export PYTHONPATH=/candidate/shadow:$PYTHONPATH; "
    if execution == "isolated_canary":
        (candidate / "control").write_bytes(b"public control")
        paths = (str(guest),)
        if failure == "wrong_origin":
            paths = ("/candidate/shadow", *paths)
        body = canary._probe(
            ("/candidate/control", hashlib.sha256(b"public control").hexdigest()), (), script, paths
        ).decode()
        environment += "export CODEX_HOME=/absent-client-private-home; "
        command = "exec /usr/bin/python3 -I -B -c " + shlex.quote(body)
    else:
        command = "exec /usr/bin/python3 -B -c " + shlex.quote(script)
    result = subprocess.run(
        [*argv, "bash", "-c", environment + command],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode and "Operation not permitted" in result.stderr:
        pytest.skip("kernel user namespaces unavailable")
    if failure is None:
        assert result.returncode == 0, result.stderr
        assert f"SELECTED_PUBLIC_LIBRARY_IMPORTS_OK {library.contract.sha256} 2" in result.stdout
        assert "AUTHORING_AND_BROKER_ROUNDTRIPS_OK" in result.stdout
        if execution == "isolated_canary":
            assert (candidate / canary._REPORT).read_text() == canary._PUBLIC + canary._DONE
    else:
        assert result.returncode != 0, (result.stdout, result.stderr)
        assert "AUTHORING_AND_BROKER_ROUNDTRIPS_OK" not in result.stdout
        assert "SELECTED_PUBLIC_LIBRARY_IMPORTS_OK" not in result.stdout
        if failure == "wrong_origin":
            assert "/candidate/shadow/merlin/alpha.py" in result.stderr
        elif failure == "wrong_bytes":
            assert "AssertionError: merlin.alpha" in result.stderr
        else:
            assert "merlin.zeta" in result.stderr
    assert str(guest) in script
