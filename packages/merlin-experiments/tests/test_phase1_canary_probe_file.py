"""Exact canary-file transport controls; no provider or target execution."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import canary as C


def _transport(path, probe):
    runtime = SimpleNamespace(
        round_kwargs=lambda: {"candidate_read_paths": ("/existing-public",), "runtime_binds": lambda home: []}
    )
    return C._probe_call_kwargs(
        runtime, context=None, run_dir=path.parent, model="synthetic", probe_path=path, probe=probe
    )


@pytest.mark.parametrize("mutation", ["bytes", "writable", "symlink"])
def test_probe_mutation_refuses_before_runtime_bind(tmp_path, mutation):
    path = tmp_path / "probe.py"
    probe = b"raise AssertionError('must not execute')\n"
    path.write_bytes(probe)
    path.chmod(0o444)
    kwargs = _transport(path, probe)
    assert kwargs["candidate_read_paths"] == ("/existing-public", C._PROBE_GUEST)
    if mutation == "bytes":
        path.chmod(0o644)
        path.write_bytes(b"changed")
        path.chmod(0o444)
    elif mutation == "writable":
        path.chmod(0o644)
    else:
        target = tmp_path / "other.py"
        target.write_bytes(probe)
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(ValueError):
        kwargs["runtime_binds"](tmp_path / "home")


@pytest.mark.parametrize("failure", [None, "changed_guest", "missing_leaf"])
def test_real_bwrap_readonly_probe_exact_hash_and_private_sibling_hidden(tmp_path, failure):
    binary = shutil.which("bwrap")
    if binary is None or not Path("/usr/bin/python3").exists():
        pytest.skip("requires local bwrap and Python")
    private, workspace = tmp_path / "private", tmp_path / "workspace"
    private.mkdir()
    workspace.mkdir()
    (private / "answer").write_text("withheld synthetic sibling")
    path = private / "probe.py"
    probe = (
        "from pathlib import Path\n"
        f"assert not Path({str(private / 'answer')!r}).exists()\n"
        "try:\n"
        f"    Path({C._PROBE_GUEST!r}).write_text('changed')\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('probe mount is writable')\n"
        "Path('result').write_text('original λ source')\n"
    ).encode()
    path.write_bytes(probe)
    path.chmod(0o444)
    binds = _transport(path, probe)["runtime_binds"](tmp_path / "home")
    assert binds == ["--ro-bind", str(path), C._PROBE_GUEST]
    if failure == "changed_guest":
        replacement = private / "replacement.py"
        replacement.write_text("from pathlib import Path;Path('forged').write_text('executed')")
        binds[1] = str(replacement)
    elif failure == "missing_leaf":
        binds = []
    argv = [binary, "--unshare-all", "--die-with-parent", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for directory in ("/lib", "/lib64"):
        if Path(directory).exists():
            argv += ["--ro-bind", directory, directory]
    argv += ["--bind", str(workspace), "/candidate", "--chdir", "/candidate", *binds]
    command = C._probe_command(probe).removeprefix("python3 ")
    result = subprocess.run(
        [*argv, "/usr/bin/python3", *shlex.split(command)], capture_output=True, text=True, timeout=10
    )
    if result.returncode and "Operation not permitted" in result.stderr:
        pytest.skip("kernel user namespaces unavailable")
    if failure is None:
        assert result.returncode == 0, result.stderr
        assert (workspace / "result").read_text() == "original λ source"
    else:
        assert result.returncode != 0, result.stdout
        assert not (workspace / "result").exists() and not (workspace / "forged").exists()
    assert path.read_bytes() == probe
