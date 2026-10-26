"""The attestation replay may run beside issuance, with the same comparison and guarantee.

A full-model capture takes hours, and its attestation replay re-executes it in full. Run after
issuance, the replay doubles the wall time. Run beside it, on a disjoint CPU set, the replay is still
a fresh sandboxed execution of the identical command on the same sealed snapshot, into its own
empty output, and it is compared byte for byte by the same replay_verify code.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments import cli
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_m2m import SealedM2MError
from merlin_experiments.phase0 import capture_selection

CHILD = r"""
import json, os, sys, time
out = sys.argv[1]
start = time.time()
time.sleep(float(sys.argv[2]))
with open(os.path.join(out, "result.json"), "w") as stream:
    json.dump({"start": start, "end": time.time(), "cpus": sorted(os.sched_getaffinity(0))}, stream)
print("same stdout")
sys.exit(int(sys.argv[3]))
"""


def _two_cpu_sets():
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("two CPUs are needed to pin two executions apart")
    return frozenset(cpus[:1]), frozenset(cpus[1:2])


def _child_argv(monkeypatch, tmp_path, timings):
    script = tmp_path / "child.py"
    script.write_text(CHILD)

    def argv(_bwrap, _runtime, _source, output, *_args, **_kwargs):
        sleep, code = timings[Path(output).name]
        return [sys.executable, "-I", str(script), str(output), str(sleep), str(code)]

    monkeypatch.setattr(sealed_m2m, "_sandbox_argv", argv)


def test_both_executions_run_at_once_each_on_its_own_cpus(tmp_path, monkeypatch):
    cpus = _two_cpu_sets()
    outputs = (tmp_path / "issued", tmp_path / "replayed")
    for path in outputs:
        path.mkdir()
    _child_argv(monkeypatch, tmp_path, {"issued": (1.0, 0), "replayed": (1.0, 0)})
    issued, replayed = sealed_m2m._execute_beside(
        Path("bwrap"), Path("root"), Path("source"), outputs, ("cmd",), Path("/out"), cpus=cpus, timeout_seconds=60
    )
    assert issued == replayed and issued["returncode"] == 0
    first, second = (json.loads((path / "result.json").read_text()) for path in outputs)
    assert first["start"] < second["end"] and second["start"] < first["end"], "the two executions overlap"
    assert first["cpus"] == sorted(cpus[0]) and second["cpus"] == sorted(cpus[1])
    assert os.sched_getaffinity(0) >= cpus[0] | cpus[1], "the issuer's own CPU set is unchanged"


def test_a_failed_execution_stops_the_other_and_fails_the_issue(tmp_path, monkeypatch):
    cpus = _two_cpu_sets()
    outputs = (tmp_path / "issued", tmp_path / "replayed")
    for path in outputs:
        path.mkdir()
    _child_argv(monkeypatch, tmp_path, {"issued": (60.0, 0), "replayed": (0.2, 3)})
    start = time.monotonic()
    with pytest.raises(SealedM2MError, match="exited 3"):
        sealed_m2m._execute_beside(
            Path("bwrap"), Path("root"), Path("source"), outputs, ("cmd",), Path("/out"), cpus=cpus, timeout_seconds=120
        )
    assert time.monotonic() - start < 30
    assert not (outputs[0] / "result.json").exists(), "the still-running execution was stopped"


def test_cpu_sets_are_equal_disjoint_halves():
    assert sealed_m2m.replay_cpu_sets({31, 24, 25, 26, 27, 28, 29, 30}) == (
        frozenset({24, 25, 26, 27}),
        frozenset({28, 29, 30, 31}),
    )
    assert sealed_m2m.replay_cpu_sets([5, 3, 4]) == (frozenset({3}), frozenset({4}))
    with pytest.raises(SealedM2MError, match="two CPUs"):
        sealed_m2m.replay_cpu_sets([7])
    with pytest.raises(SealedM2MError, match="disjoint"):
        sealed_m2m.issue_and_replay({}, Path("/unused"), cpus=(frozenset({1, 2}), frozenset({2, 3})))
    with pytest.raises(SealedM2MError, match="disjoint"):
        sealed_m2m.issue_and_replay({}, Path("/unused"), cpus=(frozenset(), frozenset({3})))


def _pending_v3(tmp_path, monkeypatch):
    """A v3 pending receipt whose identities replay_verify accepts (cf. the receipt-policy test)."""
    run = tmp_path / "run"
    source, runtime, output = run / "snapshots/source", run / "snapshots/guest-root", run / "capture"
    for path in (source, runtime, output):
        path.mkdir(parents=True)
    (source / "workload").mkdir()
    (source / "workload/loader.py").write_text("def get_model_and_inputs(): pass\n")
    identity, output_identity = {"sha256": "input"}, {"sha256": "output"}
    process, materialized = {"returncode": 0, "stdout": "same"}, {"status": "verified_materialized"}
    command = sealed_m2m._command_v2(output, dtype="int8", recipe=True)
    plan = {
        "schema": sealed_m2m.SCHEMA_V3,
        "status": "plan_only",
        "command_template_sha256": sealed_m2m._digest(
            sealed_m2m._command_v2(Path("/capture-out"), dtype="int8", recipe=True)[-1].encode()
        ),
        "dtype": "int8",
        "recipe": {"sha256": "selected"},
        "m2m_commit": "a" * 40,
        "base": "/usr",
        "merlin_root": "/selected/merlin",
        "schemas_root": "/selected/merlin/_data/schemas",
        "worker_sha256": "receipt",
        "selected_trees": {name: identity for name in ("venv", "base", "m2m", "workload", "merlin", "schemas")},
        "execution_timeout_seconds": 3600,
        "loader_env": {},
        "loader_env_reads": [],
        "selected_inputs": [{"role": "checkpoint", "guest_member": "weights.bin"}],
    }
    receipt = {
        "schema": sealed_m2m.SCHEMA_V3,
        "status": "pending_replay",
        "issuer_sha256": "current-issuer",
        "nonce": "0" * 32,
        "plan": plan,
        "command": list(command),
        "policy_sha256": sealed_m2m._policy(command, output, loader_env={}, timeout_seconds=3600),
        "scope": sealed_m2m._V3_SCOPE,
        "source": identity,
        "guest_root": identity,
        "schemas": identity,
        "output": output_identity,
        "process": process,
        "materialized": materialized,
        "bwrap_sha256": "bwrap",
    }
    (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
    replays = []
    monkeypatch.setattr(sealed_m2m, "_validate_snapshots", lambda *_, **__: None)
    monkeypatch.setattr(
        sealed_m2m,
        "_snapshot_tree",
        lambda path, **_: (
            identity
            if Path(path).is_relative_to(source) or Path(path).is_relative_to(runtime)
            else ({"sha256": "tampered"} if (Path(path) / "tampered").exists() else output_identity)
        ),
    )
    monkeypatch.setattr(sealed_m2m, "_tree", lambda path, **_: {"owner": Path(path)})
    monkeypatch.setattr(sealed_m2m, "_snapshot_rows", lambda rows, relative=".": identity)
    monkeypatch.setattr(sealed_m2m, "_materialized_v3", lambda *_: materialized)
    monkeypatch.setattr(sealed_m2m, "_declared_loader_env", lambda *_: ({}, []))
    monkeypatch.setattr(sealed_m2m, "_verify_selected_input", lambda *_: None)
    monkeypatch.setattr(sealed_m2m, "_execute", lambda *args, **kwargs: replays.append(args) or dict(process))
    monkeypatch.setattr(sealed_m2m, "_bwrap_binary", lambda *_: tmp_path / "bwrap")
    monkeypatch.setattr(
        sealed_m2m,
        "_file_digest",
        lambda path: (
            "bwrap"
            if Path(path).name == "bwrap"
            else "current-issuer"
            if Path(path).name == "sealed_m2m.py"
            else "receipt"
        ),
    )
    return run, receipt, process, replays


def test_a_concurrent_replay_is_compared_exactly_as_a_replay_run_now(tmp_path, monkeypatch):
    run, receipt, process, replays = _pending_v3(tmp_path, monkeypatch)
    assert sealed_m2m.replay_verify(run)["status"] == "verified_sandbox_replay"
    assert len(replays) == 1, "without a concurrent replay, replay_verify executes one now"
    replay = run / "m2m-replay-beside"
    replay.mkdir()
    result = sealed_m2m.replay_verify(run, _executed=(dict(process), replay))
    assert result["status"] == "verified_sandbox_replay" and result["sealed_source_closure_replayed"] is True
    assert len(replays) == 1, "the concurrent replay is not executed again"
    with pytest.raises(SealedM2MError, match="process output differs"):
        sealed_m2m.replay_verify(run, _executed=({**process, "stdout": "other"}, replay))
    (replay / "tampered").write_text("x")
    with pytest.raises(SealedM2MError, match="fresh M2M replay differs: unexpected member 'tampered'"):
        sealed_m2m.replay_verify(run, _executed=(dict(process), replay))
    (replay / "tampered").unlink()
    for elsewhere in (tmp_path / "m2m-replay-outside", run / "capture"):
        elsewhere.mkdir(exist_ok=True)
        with pytest.raises(SealedM2MError, match="fresh replay directory"):
            sealed_m2m.replay_verify(run, _executed=(dict(process), elsewhere))
    # Every receipt policy check still applies before the comparison.
    receipt["issuer_sha256"] = "f" * 64
    (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
    with pytest.raises(SealedM2MError, match="unsupported policy"):
        sealed_m2m.replay_verify(run, _executed=(dict(process), replay))


@pytest.mark.parametrize(
    "issuer",
    [
        "_PRE_CONCURRENT_REPLAY_ISSUER_SHA256",
        "_PRE_SELECTED_THREADS_ISSUER_SHA256",
        "_PRE_PUBLICATION_DIAGNOSTICS_ISSUER_SHA256",
    ],
)
def test_receipts_of_the_previous_issuers_still_replay(tmp_path, monkeypatch, issuer):
    run, receipt, _process, _replays = _pending_v3(tmp_path, monkeypatch)
    receipt["issuer_sha256"] = getattr(sealed_m2m, issuer)
    (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
    assert sealed_m2m.replay_verify(run)["status"] == "verified_sandbox_replay"


def test_issue_and_attest_hands_the_concurrent_replay_to_the_attestation(tmp_path, monkeypatch):
    run = tmp_path / "run"
    selection = tmp_path / "selection/capture-selection.json"
    selection.parent.mkdir()
    selection.write_text("{}")
    output = tmp_path / "attestations/capture.json"
    output.parent.mkdir()
    monkeypatch.setattr(capture_selection, "load", lambda *_args, **_kwargs: {"run_dir": str(run)})
    replay = {"status": "verified_sandbox_replay", "receipt_sha256": "r" * 64}
    calls = []

    def fake_issue(path, *, expected_sha256, cpus=None):
        calls.append(("issue", cpus))
        return run / "sealed_m2m_pending.json", replay

    def fake_attest(path, *, expected_sha256, output, _replay=None):
        calls.append(("attest", _replay))
        return {"capture_execution_attestation": str(output), "issuer": "sealed", "capture": {}}

    monkeypatch.setattr(capture_selection, "_issue", fake_issue)
    monkeypatch.setattr(capture_selection, "attest", fake_attest)
    cpus = (frozenset({0}), frozenset({1}))
    result = capture_selection.issue_and_attest(selection, expected_sha256="s" * 64, output=output, cpus=cpus)
    assert calls == [("issue", cpus), ("attest", replay)]
    assert result["replay_execution"] == {"mode": "concurrent_disjoint_cpus", "issue_cpus": [0], "replay_cpus": [1]}
    # A destination the attestation could not be written to is refused before anything executes.
    calls.clear()
    output.write_text("{}")
    with pytest.raises(ValueError, match="fresh"):
        capture_selection.issue_and_attest(selection, expected_sha256="s" * 64, output=output, cpus=cpus)
    assert calls == []


def test_concurrent_replay_flag_requires_an_attestation_output(tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(capture_selection, "issue", lambda *a, **k: called.append(a))
    monkeypatch.setattr(capture_selection, "issue_and_attest", lambda *a, **k: called.append(a))
    argv = ["corpus", "capture", "issue", "--selection", str(tmp_path / "s.json"), "--expected-sha256", "a" * 64]
    assert cli.main([*argv, "--concurrent-replay"]) != 0
    assert "--concurrent-replay needs --attestation-output" in capsys.readouterr().err
    assert called == []


def test_concurrent_replay_flag_splits_this_process_cpus(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(path, *, expected_sha256, output, cpus):
        calls.append(cpus)
        return {"sealed_receipt": "r", "capture_execution_attestation": str(output)}

    monkeypatch.setattr(capture_selection, "issue_and_attest", fake)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {24, 25, 26, 27, 28, 29, 30, 31})
    output = tmp_path / "attestation.json"
    argv = ["corpus", "capture", "issue", "--selection", str(tmp_path / "s.json"), "--expected-sha256", "a" * 64]
    assert cli.main([*argv, "--attestation-output", str(output), "--concurrent-replay"]) == 0
    assert calls == [(frozenset({24, 25, 26, 27}), frozenset({28, 29, 30, 31}))]
    assert json.loads(capsys.readouterr().out)["phase0_admission"] == "not_granted"


def _minimal_root(root: Path, mount: Path) -> None:
    """A guest root holding only ``/bin/sh`` (dash), its libraries and the sandbox mount points."""
    import subprocess

    shell = Path(shutil.which("dash") or "/usr/bin/dash")
    if not shell.is_file():
        pytest.skip("dash is unavailable for a minimal guest root")
    libraries = [
        Path(word)
        for line in subprocess.run(["ldd", str(shell)], capture_output=True, text=True, check=True).stdout.splitlines()
        for word in line.split()
        if word.startswith("/")
    ]
    for path in (shell, *libraries):
        target = root / path.resolve().relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path.resolve(), target)
        if path != path.resolve():
            alias = root / path.relative_to("/")
            alias.parent.mkdir(parents=True, exist_ok=True)
            if not alias.exists():
                alias.symlink_to(path.resolve())
    (root / "bin").mkdir(exist_ok=True)
    if not (root / "bin/sh").exists():
        shutil.copy2(shell.resolve(), root / "bin/sh")
    for name in ("source", "tmp", "dev", "proc", *mount.relative_to("/").parts[:1]):
        (root / name).mkdir(exist_ok=True)
    (root / mount.relative_to("/")).mkdir(parents=True, exist_ok=True)


def test_real_sandboxes_each_see_only_their_own_empty_output(tmp_path):
    bwrap = os.environ.get("MERLIN_BWRAP") or shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is unavailable (MERLIN_BWRAP selects one)")
    cpus = _two_cpu_sets()
    mount = Path("/capture-out/run/capture")
    root, source = tmp_path / "guest-root", tmp_path / "source"
    outputs = (tmp_path / "issued", tmp_path / "replayed")
    for path in (root, source, *outputs):
        path.mkdir()
    _minimal_root(root, mount)
    # Shell builtins only: refuse a non-empty output, write one file, stay alive so both overlap.
    script = (
        f'set -- {mount}/*; [ "$1" = "{mount}/*" ] || exit 7; echo fresh > {mount}/out; '
        "i=0; while [ $i -lt 300000 ]; do i=$((i+1)); done; echo done"
    )
    try:
        issued, replayed = sealed_m2m._execute_beside(
            Path(bwrap), root, source, outputs, ("/bin/sh", "-c", script), mount, cpus=cpus, timeout_seconds=60
        )
    except SealedM2MError as exc:
        if "Unknown option" in str(exc) or "namespace" in str(exc).lower():
            pytest.skip(f"this bubblewrap or host cannot run the sealed sandbox flags: {exc}")
        raise
    assert issued == replayed and issued["stdout"]["bytes"] == len("done\n")
    assert [sorted(p.name for p in path.iterdir()) for path in outputs] == [["out"], ["out"]]
    assert not any((root / mount.relative_to("/")).iterdir()), "the guest mount point itself is never written"
