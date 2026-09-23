"""Cold installed resume over real V4 sources and synthetic static-plan evidence.

Only three isolated Python children are launched. Each denies further processes and
listeners before importing production owners. No compiler, sandbox or engine runs.
"""

import json
import shutil
import socket
import subprocess
import sys
import sysconfig
from pathlib import Path

import merlin_experiments
from merlin_experiments import frozen_python as FP
from merlin_experiments import source_snapshot as SNAP
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import portfolio_resume as R

import merlin

GUARD = """
import socket, subprocess
def forbidden(*args, **kwargs):
    raise AssertionError('cold checkpoint verification forbids subprocesses and listeners')
subprocess.Popen = forbidden
socket.socket.bind = forbidden
"""

PREPARE = (
    GUARD
    + """
import importlib.util, json, sys
from pathlib import Path
from merlin_experiments import source_snapshot as SNAP
from merlin_experiments.phase2 import contracts as C, host_policy as HP, portfolio_resume as R
root, output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('checkpoint_fixture', root/'fixtures/checkpoint.py')
fixture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)
controller = root/'experiments/merlin_experiments/phase2/global_experiment.py'
contract, shared = root/'contract', root/'shared'
policy = HP.build_record(controller_source=controller, contract_root=contract)
checkpoint = fixture.build_checkpoint(output, shared_source_root=shared, host_policy=policy)
receipt = SNAP.verify(root)
checkpoint.document.update(source_snapshot=str(root), source_snapshot_files_sha256=C.document_sha256(receipt['files']))
checkpoint.save()
dispatch = R.installed_dispatch(snapshot=root, controller_source=controller,
    contract_root=contract, compiler_shared_source_root=shared)
fixture.write(output/'launch.json', {'source_snapshot': str(root), 'checkpoint_verifier': dispatch})
print(json.dumps({'checkpoint': str(checkpoint.path)}))
"""
)

CONSUME = (
    GUARD
    + """
import sys
from merlin_experiments.phase2 import portfolio_resume as R
raise SystemExit(R.main(['--checkpoint', sys.argv[1]]))
"""
)


def _copy_python(source, destination):
    # Copy selected package code only, not site-packages, resources or private corpora.
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*.py"):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def test_cold_installed_ready_checkpoint_resume(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    allowed = []

    def bounded_popen(command, *args, **kwargs):
        assert not kwargs.get("shell")
        assert command in allowed, "only the exact frozen Python commands are allowed"
        assert kwargs.get("cwd") == tmp_path
        return real_popen(command, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("listeners forbidden")

    monkeypatch.setattr(subprocess, "Popen", bounded_popen)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    # Avoid a separate discovery subprocess; the actual frozen bootstrap/resolver
    # still runs cold and uses this interpreter's ordinary dependency directories.
    monkeypatch.setattr(
        FP,
        "_interpreter_dependencies",
        lambda executable: tuple(sysconfig.get_path(key) for key in ("purelib", "platlib")),
    )
    stage = tmp_path / "stage"
    roots = []
    for index, path in enumerate(dict.fromkeys(Path(value).resolve() for value in merlin.__path__)):
        relative = f"owner-{index}"
        _copy_python(path, stage / relative / "merlin")
        roots.append(relative)
    _copy_python(Path(merlin_experiments.__file__).resolve().parent, stage / "experiments/merlin_experiments")
    roots.append("experiments")
    for relative in HP.RESOURCE_FILES:
        path = stage / "contract" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (stage / "shared").mkdir()
    (stage / "shared/__init__.py").write_text("")
    (stage / "shared/helper.py").write_text("VALUE = 1\n")
    (stage / "fixtures").mkdir()
    shutil.copyfile(Path(__file__).with_name("portfolio_checkpoint_fixtures.py"), stage / "fixtures/checkpoint.py")
    snapshot = tmp_path / "snapshot"
    SNAP.create(
        stage,
        snapshot,
        output_root=tmp_path / "unused-output",
        source_roots=(*roots, "contract", "shared", "fixtures"),
        python_roots=tuple(roots),
        legacy_roots=(),
    )
    output = tmp_path / "case"
    output.mkdir()

    def run(code, *arguments):
        command = FP.python_command(
            snapshot, [sys.executable, "-c", code, *map(str, arguments)], verifier_source=Path(SNAP.__file__)
        )
        allowed.append(command)
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        allowed.remove(command)
        return result

    try:
        prepared = run(PREPARE, snapshot, output)
        assert prepared.returncode == 0, prepared.stderr
        checkpoint = Path(json.loads(prepared.stdout)["checkpoint"])
        selected = R.resolve(checkpoint)
        assert selected.argv == (sys.executable, "-m", R.MODULE, "--checkpoint", str(checkpoint))
        checked = run(CONSUME, checkpoint)
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout) == {
            "original_checkpoint_verified": True,
            "source_snapshot": str(snapshot),
        }
        document = json.loads(checkpoint.read_text())
        candidate = Path(document["candidate_path"]) / "compiler.py"
        candidate.write_text('raise RuntimeError("must never execute")\n')
        refused = run(CONSUME, checkpoint)
        assert refused.returncode != 0
        assert "ValueError: sealed global candidate bytes changed" in refused.stderr
    finally:
        # Pytest cleanup needs writable directories; immutable source files remain
        # untouched until all cold verification has finished.
        for path in (snapshot, *snapshot.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
