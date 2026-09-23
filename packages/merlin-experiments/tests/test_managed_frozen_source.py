"""Real guarded service/guardian processes reuse the existing native snapshot seal."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from merlin_experiments import frozen_python
from merlin_experiments.execution.chia_native import CleanupIncomplete, Session, cleanup, run, setup

from merlin.common.paths import module_source_path, python_import_roots


@pytest.fixture
def frozen_service_source(tmp_path):
    from merlin_experiments import source_snapshot as snapshot_module

    verifier = module_source_path("merlin_experiments.source_snapshot")
    source = tmp_path / "source"
    root = source / "packages/merlin-experiments/src/merlin_experiments"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "frozen_python.py").write_text(Path(frozen_python.__file__).read_text())
    execution = root / "execution"
    execution.mkdir()
    for item in module_source_path("merlin_experiments.execution").parent.glob("*.py"):
        (execution / item.name).write_bytes(item.read_bytes())

    def create(name):
        snapshot = tmp_path / name
        seal = snapshot_module.create(
            source,
            snapshot,
            output_root=tmp_path / "output",
            source_roots=("packages/merlin-experiments/src",),
            python_roots=("packages/merlin-experiments/src",),
            legacy_roots=(),
        )
        identity = {"path": str(seal), "sha256": hashlib.sha256(seal.read_bytes()).hexdigest()}
        return snapshot, identity

    snapshot, identity = create("snapshot")

    def command(argv):
        return frozen_python.python_command(snapshot, argv, verifier_source=verifier)

    yield snapshot, identity, command, create
    for path in (tmp_path, *tmp_path.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)


@contextmanager
def _running_service(command, *, environment=None):
    with tempfile.TemporaryDirectory(prefix="mfs-", dir="/tmp") as runtime:
        endpoint = Path(runtime) / "service.sock"
        process = subprocess.Popen(
            command(
                [sys.executable, "-m", "merlin_experiments.execution.native_supervisor", "--endpoint", str(endpoint)]
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        try:
            deadline = time.monotonic() + 10
            while not endpoint.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("frozen service startup failed: " + process.communicate(timeout=1)[1])
                time.sleep(0.02)
            yield endpoint
        finally:
            if process.poll() is None:
                process.terminate()
            _, error = process.communicate(timeout=12)
            assert process.returncode == 0, error


@pytest.fixture
def frozen_service(frozen_service_source):
    snapshot, identity, command, create = frozen_service_source
    with _running_service(command) as endpoint:
        yield endpoint, snapshot, identity, command, create


def test_identity_requires_guarded_bootstrap_not_environment(frozen_service_source, monkeypatch):
    _, identity, command, _ = frozen_service_source
    monkeypatch.setenv(frozen_python.CONTEXT, json.dumps({"seal": identity}))
    assert frozen_python.active_source_identity() is None
    # Build before installing bogus ambient context; run with it to prove bootstrap
    # independently installs its already-verified process-local identity.
    monkeypatch.delenv(frozen_python.CONTEXT)
    argv = command(
        [
            sys.executable,
            "-c",
            "import json; from merlin_experiments.frozen_python import active_source_identity; "
            "print(json.dumps(active_source_identity()))",
        ]
    )
    environment = {**os.environ, frozen_python.CONTEXT: json.dumps({"seal": {"sha256": "0" * 64}})}
    result = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == identity


def test_frozen_service_and_guardian_share_existing_seal(frozen_service, tmp_path):
    endpoint, _, identity, _, _ = frozen_service
    with Session(endpoint, expected_source=identity) as session:
        invitation = session.reserve()
        setup(invitation)
        try:
            assert run([sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path) == 7
        finally:
            cleanup(invitation)
        receipt = session.receipt(invitation)
        assert receipt["source"] == identity
        assert receipt["guardian_reaped"] and receipt["cleanup_complete"]


def test_stale_service_rejects_new_seal_and_unqualified_client(frozen_service):
    endpoint, _, identity, _, create = frozen_service
    _, newer = create("newer")
    assert newer != identity
    for expected in (None, newer, {**identity, "sha256": "0" * 64}):
        with pytest.raises(RuntimeError, match="PermissionError"):
            Session(endpoint, expected_source=expected)
    # Mismatches did not mutate or upgrade the resident service's startup identity.
    with Session(endpoint, expected_source=identity) as session:
        invitation = session.reserve()
        session.cancel(invitation)
        receipt = session.receipt(invitation)
        assert receipt["source"] == identity and not receipt["guardian_created"]


@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_guardian_revalidates_frozen_bytes_after_service_start(frozen_service, damage):
    endpoint, snapshot, identity, _, _ = frozen_service
    guardian = snapshot / "packages/merlin-experiments/src/merlin_experiments/execution/native_guardian.py"
    if damage == "missing":
        guardian.parent.chmod(0o700)
        guardian.unlink()
    else:
        guardian.chmod(0o600)
        guardian.write_text(guardian.read_text() + "\n# changed after supervisor startup\n")
        guardian.chmod(0o444)
    session = Session(endpoint, expected_source=identity)
    invitation = session.reserve()
    with pytest.raises((RuntimeError, EOFError)):
        setup(invitation)
    receipt = session.receipt(invitation)
    assert receipt["source"] == identity
    assert receipt["guardian_created"] and receipt["guardian_reaped"]
    assert receipt["guardian"] is None and not receipt["cleanup_complete"]
    with pytest.raises(CleanupIncomplete):
        session.close()


def test_frozen_caller_cannot_implicitly_downgrade(frozen_service):
    endpoint, _, identity, command, _ = frozen_service
    program = (
        "import json; from merlin_experiments.execution.chia_native import Session; "
        f"session=Session({str(endpoint)!r}); print(json.dumps(session.source)); session.close()"
    )
    result = subprocess.run(command([sys.executable, "-c", program]), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == identity


def test_environment_only_service_cannot_claim_frozen_identity(frozen_service_source):
    _, identity, command, _ = frozen_service_source
    context = json.loads(command([sys.executable, "-c", "pass"])[-1])["context"]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
        frozen_python.CONTEXT: json.dumps(context),
    }
    # A full valid inherited context is still not proof this service was guarded.
    with _running_service(lambda argv: argv, environment=environment) as endpoint:
        with pytest.raises(RuntimeError, match="PermissionError"):
            Session(endpoint, expected_source=identity)
        with pytest.raises(ValueError, match="snapshot seal pin"):
            Session(endpoint, expected_source={})
        program = (
            "from merlin_experiments.execution.chia_native import Session; "
            f"Session({str(endpoint)!r}); print('downgraded')"
        )
        result = subprocess.run(command([sys.executable, "-c", program]), capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and "PermissionError" in result.stderr
        assert "downgraded" not in result.stdout
        # Refused frozen admission did not prevent ordinary non-frozen sessions.
        with Session(endpoint) as session:
            invitation = session.reserve()
            session.cancel(invitation)
            assert session.receipt(invitation)["source"] is None


@pytest.mark.parametrize("damage", ["changed", "added", "removed", "initializer"])
def test_execution_package_membership_and_bytes_are_bound(tmp_path, monkeypatch, damage):
    from merlin_experiments.phase1 import source_inputs
    from merlin_experiments.spec import SpecError

    package = tmp_path / "execution"
    shutil.copytree(
        module_source_path("merlin_experiments.execution").parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    original = source_inputs._source

    def copied_source(module):
        if module == "merlin_experiments.execution":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.execution."):
            return package / (module.rsplit(".", 1)[-1] + ".py")
        return original(module)

    monkeypatch.setattr(source_inputs, "_source", copied_source)
    arguments = {"repo": tmp_path, "entrypoint": tmp_path / "transport.py"}
    record = source_inputs.record(**arguments)
    for name in ("__init__", "_protocol", "chia_native", "native_supervisor", "native_guardian"):
        assert record["inputs"][f"phase1:startup:execution:{name}.py"]["path"] == str(package / f"{name}.py")
    source_inputs.verify(record, **arguments)
    member = package / ("__init__.py" if damage == "initializer" else "native_guardian.py")
    if damage == "removed":
        member.unlink()
    elif damage == "added":
        (package / "new_owner.py").write_text("# new implementation member\n")
    else:
        member.write_text(member.read_text() + "\n# source changed\n")
    with pytest.raises(SpecError, match="source identity changed|missing or foreign source owner"):
        source_inputs.verify(record, **arguments)
