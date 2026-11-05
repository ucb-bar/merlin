"""Actual clean-environment compiler launches use their existing writable scope.

This exercises process transport only; it grants no compiler or target verdict.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R

from merlin.common.cpu_pool import shared_compute_pool


def test_actual_original_launch_uses_scoped_scratch_after_clearenv(tmp_path, monkeypatch):
    binary = shutil.which("bwrap")
    python = Path("/usr/bin/python3")
    if binary is None or not python.is_file():
        pytest.skip("requires selected local bwrap and system Python")
    workspace = tmp_path / "compiler scratch's directory"
    workspace.mkdir()
    package = tmp_path / "immutable-package"
    package.mkdir()
    forbidden = tmp_path / "ambient-host-tmp"
    forbidden.mkdir()
    monkeypatch.setenv("TMPDIR", str(forbidden))
    argv = [binary, "--unshare-all", "--die-with-parent", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for name in ("/lib", "/lib64"):
        if Path(name).exists():
            argv += ["--ro-bind", name, name]
    argv += ["--tmpfs", "/tmp", "--bind", str(workspace), str(workspace),
             "--ro-bind", str(package), str(package)]
    policy = SimpleNamespace(argv=tuple(argv), workspace=workspace, env_prefix="")
    script = (
        "import json,os,tempfile; from pathlib import Path; "
        "root=Path(os.environ['TMPDIR']); "
        "handle=tempfile.NamedTemporaryFile(dir=root); "
        "handle.write(b'current compiler scratch');handle.flush(); "
        "print(json.dumps(dict(root=str(root),leaf=handle.name,bytes=Path(handle.name).read_text()))); "
        "handle.close()"
    )
    executor = object.__new__(R._Executor)
    with shared_compute_pool(timeout_s=30):
        result = executor._invoke(policy, package, [str(python), "-I", "-B", "-c", script], timeout=15)
    if result.returncode and "Operation not permitted" in result.stderr:
        pytest.skip("kernel user namespaces unavailable for real local sandbox check")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["root"] == str(workspace)
    assert Path(report["leaf"]).parent == workspace
    assert report["bytes"] == "current compiler scratch"
    assert not Path(report["leaf"]).exists()
    assert not tuple(forbidden.iterdir())
