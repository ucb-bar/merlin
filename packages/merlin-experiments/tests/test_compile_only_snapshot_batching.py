"""Actual custody boundaries before publishing public compiler observations."""

import json
import os
import runpy
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import compile_only as C
from merlin_experiments.phase2 import normal_resource_launch as resources

from merlin.common import bundle_snapshot as BW
from merlin.common.paths import repo_root


@dataclass(frozen=True)
class Policy:
    argv: tuple = ("/usr/bin/true",)
    env_prefix: str = ""


def _shared_original(original, tmp_path):
    from merlin.targetgen.common_transforms import CommonTransformService
    from merlin.targetgen.sandbox import bwrap
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V8, attach_common_substrate
    from merlin.targetgen.tool_profile_shared import prepare_shared_source_release

    sdk = tmp_path / "public-sdk"
    prepare_shared_source_release(sdk, python_source_root=repo_root() / "src", license_directory=repo_root(), version=8)
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V8, python_source_root=sdk, host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(sdk / row["path"].removeprefix("merlin/python/"))
    inputs = tmp_path / "selected-inputs"
    bundle["allowed"].append({"path": str(inputs), "kind": "directory", "note": "synthetic custody inputs"})
    archive, workspace = tmp_path / "shared.yaml", tmp_path / "shared" / "workspace"
    archive.write_text(yaml.safe_dump(bundle))
    workspace.mkdir(parents=True)
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo_root())
    original.service = CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo_root())
    original.padding = BW.snapshot_path(BW.bundle_snapshot_root(workspace), inputs / "payload.bin", repo_root())
    return original


@pytest.fixture
def command(tmp_path, monkeypatch, request):
    fixture = runpy.run_path(str(repo_root() / "merlin/tests/targetgen/test_common_request_snapshot_scope.py"))
    original = fixture["original"].__wrapped__(tmp_path, monkeypatch)
    if getattr(request, "param", None) == "v8":
        original = _shared_original(original, tmp_path)
    package = tmp_path / "candidate"
    package.mkdir()
    output = tmp_path / "products"
    output.mkdir()
    source = tmp_path / "public.mlir"
    source.write_text("original public source")
    destination = tmp_path / "published.json"
    executor = C._Executor("fixture", package, output, common_service=original.service)
    selected = {}

    def argv(pkg, name, incoming, outgoing):
        selected["outgoing"] = outgoing
        return [sys.executable]

    @contextmanager
    def transport(owner, **kwargs):
        owner._verify()
        yield SimpleNamespace(env_prefix=lambda prefix: prefix, remaining_timeout=lambda: 30)
        owner._verify()

    def run(argv, **kwargs):
        original.service._verify()
        selected["environment"] = kwargs["env"]
        if selected["outgoing"] is not None:
            selected["outgoing"].write_text("actual public product")
        return subprocess.CompletedProcess(argv, 0, stdout="actual compiler result", stderr="")

    monkeypatch.setattr(C.P, "_resolve_argv", argv)
    monkeypatch.setattr(C.P, "_needs_interpreter", lambda *a: False)
    monkeypatch.setattr(C.PC, "package_sandbox_policy", lambda *a, **k: Policy())
    import merlin_experiments.phase1.brokers.compiler_call_channel as channel
    monkeypatch.setattr(channel, "compiler_call_transport", transport)
    monkeypatch.setattr(resources, "bounded_normal_scratch", lambda prefix, **k: nullcontext(prefix))
    monkeypatch.setattr(C.subprocess, "run", run)
    pkg = SimpleNamespace(directory=package)

    def invoke(stage="parse"):
        with C.P.scoped_package_executor(executor):
            return executor.run_entrypoint(pkg, stage, source, destination,
                                           invocation_directory=output, timeout=30)

    return SimpleNamespace(original=original, executor=executor, invoke=invoke, output=output,
                           destination=destination, source=source, pkg=pkg, selected=selected,
                           run=run, channel=channel)


def reads(monkeypatch, member):
    digest = BW.hashlib.file_digest
    observed = []

    def measured(stream, *a, **k):
        observed.append(Path(stream.name))
        return digest(stream, *a, **k)

    monkeypatch.setattr(BW.hashlib, "file_digest", measured)
    return lambda: observed.count(member)


def test_each_original_compiler_command_keeps_fresh_entry_exit_and_publishes_afterward(command, monkeypatch):
    count = reads(monkeypatch, command.original.padding)
    for _ in range(2):
        assert command.invoke().returncode == 0
        assert command.destination.read_text() == "actual public product"
    assert count() == 4  # Two per actual invocation, never across-command reuse.
    assert BW._ACTIVE_VERIFICATION.get() is None
    records = [json.loads(p.read_bytes()) for p in command.output.rglob("invocation.json")]
    assert len(records) == 2 and all(row["status"] == "completed" for row in records)
    for row in records:
        actual = row["process_timing"]
        assert row["started_ns"] <= actual["started_ns"] <= actual["finished_ns"] <= row["finished_ns"]
        assert actual["wall_s"] >= 0


def test_inner_actual_proof_request_retains_independent_full_boundaries(command, monkeypatch):
    count = reads(monkeypatch, command.original.padding)

    @contextmanager
    def proof(owner, **kwargs):
        actual = owner.check(command.original.request, work_root=command.output, timeout_s=30)
        assert actual["status"] == "unsupported"
        yield SimpleNamespace(env_prefix=lambda prefix: prefix, remaining_timeout=lambda: 30)

    monkeypatch.setattr(command.channel, "compiler_call_transport", proof)
    command.invoke()
    assert count() == 4  # Outer compiler pre/post plus independent real check pre/post.
    assert BW._ACTIVE_VERIFICATION.get() is None


@pytest.mark.parametrize("mutation", ["payload", "archive"])
def test_final_mutation_refuses_receipt_and_staged_output_publication(command, monkeypatch, mutation):
    def changed(argv, **kwargs):
        process = command.run(argv, **kwargs)
        path = command.original.padding if mutation == "payload" else command.original.service.bundle_manifest
        before = path.stat()
        path.chmod(0o600)
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return process

    monkeypatch.setattr(C.subprocess, "run", changed)
    with pytest.raises((ValueError, RuntimeError)):
        command.invoke()
    assert not command.destination.exists()
    records = [json.loads(p.read_bytes()) for p in command.output.rglob("invocation.json")]
    assert len(records) == 1 and records[0]["status"] == "interrupted"
    assert BW._ACTIVE_VERIFICATION.get() is None


def test_observed_process_environment_remains_the_same_snapshot(command, monkeypatch):
    monkeypatch.setenv("PUBLIC_ENV_CONTROL", "before")
    observe = C.invocation_record.observe

    @contextmanager
    def changed(*a, **k):
        with observe(*a, **k) as record:
            os.environ["PUBLIC_ENV_CONTROL"] = "after"
            yield record

    monkeypatch.setattr(C.invocation_record, "observe", changed)
    command.invoke()
    assert command.selected["environment"]["PUBLIC_ENV_CONTROL"] == "before"


def test_foreign_or_unscoped_executor_cannot_enter_original_command(command):
    with pytest.raises(ValueError, match="scoped executor"):
        command.executor.run_entrypoint(command.pkg, "parse", command.source,
                                        invocation_directory=command.output)
    foreign = SimpleNamespace(build_package=lambda *a: None, run_entrypoint=lambda *a: None)
    with C.P.scoped_package_executor(foreign), pytest.raises(ValueError, match="scoped executor"):
        command.executor.run_entrypoint(command.pkg, "parse", command.source,
                                        invocation_directory=command.output)
    assert not list(command.output.rglob("invocation.json"))


def test_setup_batches_custody_only_and_closes_before_body(command, monkeypatch):
    count = reads(monkeypatch, command.original.padding)
    owner = command.original.service
    with C._snapshot_operation(owner):
        owner._verify()
        owner._verify()
    assert count() == 2 and BW._ACTIVE_VERIFICATION.get() is None
    with pytest.raises(ValueError, match="exact original profile owner"):
        with C._snapshot_operation(object()):
            pytest.fail("foreign operation owner was admitted")


def test_original_namespace_reuse_never_reuses_snapshot_checks_across_commands(command, monkeypatch):
    count = reads(monkeypatch, command.original.padding)
    with C._compile_imports(command.original.service):
        assert BW._ACTIVE_VERIFICATION.get() is None
        command.invoke()
        command.invoke()
    assert count() == 7  # Namespace setup2/exit1 and two independent commands2 each.
    assert BW._ACTIVE_VERIFICATION.get() is None


def test_namespace_final_check_refuses_caught_mutation_before_aggregate_publication(command):
    owner = command.original.service
    published = []
    with pytest.raises(RuntimeError, match="content differs"):
        with C._compile_imports(owner):
            assert BW._ACTIVE_VERIFICATION.get() is None
            path = command.original.padding
            before = path.stat()
            path.chmod(0o600)
            content = path.read_bytes()
            path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        published.append("aggregate")
    assert not published and BW._ACTIVE_VERIFICATION.get() is None


@pytest.mark.parametrize("command", ["v8"], indirect=True)
def test_shared_profile_commands_keep_independent_entry_and_final_payload_reads(command, monkeypatch):
    count = reads(monkeypatch, command.original.padding)
    for stage in ("parse", "lower_interface_to_target", "emit_target_bundle"):
        assert command.invoke(stage).returncode == 0
    assert count() == 6
    assert BW._ACTIVE_VERIFICATION.get() is None
    records = [json.loads(p.read_bytes()) for p in command.output.rglob("invocation.json")]
    assert len(records) == 3 and all(row["status"] == "completed" for row in records)


@pytest.mark.parametrize("command", ["v8"], indirect=True)
@pytest.mark.parametrize("mutation", ["payload", "archive", "marker", "member"])
def test_shared_profile_final_mutation_cannot_publish_product_or_completed_receipt(command, monkeypatch, mutation):
    def changed(argv, **kwargs):
        process = command.run(argv, **kwargs)
        owner = command.original.service
        if mutation == "member":
            path = command.original.padding.parent / "undeclared.bin"
            path.parent.chmod(0o700)
            path.write_bytes(b"undeclared")
        else:
            path = {
                "payload": command.original.padding,
                "archive": owner.bundle_manifest,
                "marker": BW.bundle_snapshot_root(owner.workspace) / BW.MARKER,
            }[mutation]
            before = path.stat()
            path.chmod(0o600)
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return process

    monkeypatch.setattr(C.subprocess, "run", changed)
    with pytest.raises((ValueError, RuntimeError)):
        command.invoke()
    assert not command.destination.exists()
    records = [json.loads(p.read_bytes()) for p in command.output.rglob("invocation.json")]
    assert len(records) == 1 and records[0]["status"] == "interrupted"
    assert BW._ACTIVE_VERIFICATION.get() is None


@pytest.mark.parametrize("version", range(1, 9))
def test_only_existing_isa_profiles_select_operation_batching(command, monkeypatch, version):
    owner, scopes = command.original.service, []
    monkeypatch.setitem(owner.profile, "common_substrate", f"merlin.common_compiler_substrate.v{version}")

    @contextmanager
    def scope(*args, **kwargs):
        scopes.append("entry")
        yield
        scopes.append("final")

    monkeypatch.setattr(BW, "verified_snapshot_scope", scope)
    with C._snapshot_operation(owner):
        pass
    assert scopes == ([] if version < 3 else ["entry", "final"])
