"""Existing Phase1 channels reach the same frozen checker on supported profiles.

Real original manifests, copied sources, request leases and fixed service methods
execute. No Lean, model compiler, numerical grader or simulator is launched.
"""

import hashlib
import threading
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import compiler_call_channel as C
from merlin_experiments.phase1.brokers import selfcheck as P1
from test_compiler_call_channel import original as original

from merlin.transforms import client


def submit(workspace, field=None):
    text = "module {}"
    payload = dict(schema=client.CHECK_SCHEMA, before_ir=text, after_ir=text, entry="absent")
    if field is not None:
        payload[field] = "candidate must not choose this"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return client._submit(
        workspace,
        payload,
        action="common-check",
        expected=dict(source_sha256=digest, target_sha256=digest),
        wait_seconds=5,
        transport_command=None,
        strict=True,
    )


def check_refusals(workspace, owner):
    answer = submit(workspace)
    assert answer["status"] == "unavailable" and answer["all_pass"] is False
    assert "no explicitly selected common formal toolchain" in answer["reason"]
    for field in ("decoder", "callback", "semantics", "proof_result"):
        answer = submit(workspace, field)
        assert answer["status"] == "unknown" and answer["all_pass"] is False
        assert "cannot select semantics" in answer["reason"]
    archive = owner.bundle_manifest
    original_bytes = archive.read_bytes()
    try:
        archive.write_bytes(original_bytes + b"\n")
        answer = submit(workspace)
        assert answer["status"] == "unknown" and answer["all_pass"] is False
        assert "authority changed" in answer["reason"]
    finally:
        archive.write_bytes(original_bytes)
    owner._verify()


@pytest.mark.parametrize(
    "original", ["merlin.common_compiler_substrate.v" + str(i) for i in (3, 4, 5, 6)], indirect=True
)
def test_actual_per_invocation_channel_preserves_schema_and_original_authority(original):
    value = original
    with C.compiler_call_transport(
        value.owner,
        workspace=value.workspace,
        package=value.package,
        input_path=value.source,
        command_name="compile",
        timeout_s=20,
    ):
        check_refusals(value.workspace, value.owner)


@pytest.mark.parametrize(
    "original", ["merlin.common_compiler_substrate.v" + str(i) for i in (3, 4, 5, 6)], indirect=True
)
def test_actual_selfcheck_channel_preserves_schema_and_original_authority(original, monkeypatch):
    value = original
    workspace = value.owner.workspace
    capsules = value.root / "capsules"
    capsules.mkdir()
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    failures = []

    def run():
        try:
            P1.main(
                [
                    "--ws",
                    str(workspace),
                    "--poll",
                    "0.01",
                    "--compiler-bundle-manifest",
                    str(value.owner.bundle_manifest),
                ],
                context=SimpleNamespace(
                    repo=value.owner.repo, descriptor=value.root / "unused.yaml", readback_policy=None
                ),
                capsules_root=capsules,
            )
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not (workspace / ".qa_channel/broker_health.json").exists():
            assert thread.is_alive() and time.monotonic() < deadline, failures
            time.sleep(0.01)
        check_refusals(workspace, value.owner)
    finally:
        (workspace / ".qa_channel/STOP").write_text("stop")
        thread.join(timeout=5)
    assert not thread.is_alive() and not failures
