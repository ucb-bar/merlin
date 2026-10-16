"""Read-only kernel refusals and pure plans; no hard-enforcement load evidence."""

import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from merlin_experiments.execution import chia_native
from merlin_experiments.execution.container_transport import PreparedContainerTransport
from merlin_experiments.phase2 import _kernel_resources as kernel
from merlin_experiments.phase2 import external_resources as resources
from merlin_experiments.phase2.contracts import StageGateError, canonical_json
from merlin_experiments.phase2.evaluation_pool import AssignedResources


def test_cpu_and_mount_parsers_have_no_issuance_authority():
    assert kernel.cpu_set("0-2,4,2") == (0, 1, 2, 4)
    with pytest.raises(StageGateError):
        kernel.cpu_set("3-1")
    row = kernel.mounts("23 1 0:2 / /owned\\040space rw,nosuid - tmpfs none rw,size=1024k\n")
    assert row == ((23, "/", "/owned space", "tmpfs", "rw,nosuid", "rw,size=1024k"),)
    with pytest.raises(StageGateError):
        kernel.mounts("not a kernel mount row")


def test_live_kernel_cannot_issue_one_byte_storage_budget(tmp_path):
    """Actual /proc/statvfs reads, without creating limits or running work."""
    with pytest.raises(StageGateError):
        resources.prepare_external_resource_boundary(
            cpu_ids=tuple(sorted(os.sched_getaffinity(0))),
            ram_bytes=1,
            disk_bytes=1,
            writable_roots=(tmp_path,),
        )
    inventory = kernel.mounts(Path("/proc/self/mountinfo").read_text())
    with pytest.raises(StageGateError):
        kernel._filesystem(inventory, (tmp_path,), 1)


def test_linked_or_reconstructed_resource_owners_refuse(tmp_path):
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(StageGateError, match="linked"):
        kernel.canonical(linked / "new")
    boundary = resources.ExternalResourceBoundary((0,), 1, 1, (tmp_path,), (), object())
    with pytest.raises(StageGateError, match="issued"):
        boundary.verify()
    with pytest.raises(StageGateError, match="exact issued"):
        with resources.executing_resource_boundary({"cpu_ids": [0]}):
            pytest.fail("serialized resource data became an execution owner")


@pytest.mark.parametrize(
    "mount",
    [
        (1, "/", "/", "ext4", "rw", "rw"),
        (2, "/", "/dev", "tmpfs", "rw", "rw"),
        (3, "/", "/dev/shm", "tmpfs", "rw", "rw"),
    ],
)
def test_namespace_plan_cannot_replace_actual_readonly_or_capped_backing(tmp_path, mount):
    # Pure negative mount parser controls, never a positive kernel-owner issue.
    capacity = os.statvfs(tmp_path)
    state = ((), tmp_path.stat().st_dev, capacity.f_blocks * capacity.f_frsize, capacity.f_frsize)
    inventory = (mount,) if mount[2] == "/" else ((1, "/", "/", "ext4", "ro", "rw"), mount)
    with pytest.raises(StageGateError):
        kernel._readonly_namespace(inventory, state)


def test_inherited_writable_regular_fd_cannot_escape_selected_backing(tmp_path):
    path = tmp_path / "inherited-output"
    capacity = os.statvfs(tmp_path)
    # A deliberately mismatched selection exercises actual /proc/fd inspection;
    # it cannot issue an enforcing boundary or establish a successful quota.
    state = ((), tmp_path.stat().st_dev + 1, capacity.f_blocks * capacity.f_frsize, capacity.f_frsize)
    with path.open("wb") as stream:
        process = Path("/proc") / str(os.getpid())
        with pytest.raises(StageGateError, match="inherited writable"):
            kernel._descriptor(process / "fd" / str(stream.fileno()), process / "fdinfo" / str(stream.fileno()), state)


def test_nonminimal_device_namespace_refuses_without_opening_devices(tmp_path):
    (tmp_path / "unselected-device").write_bytes(b"not a kernel device")
    with pytest.raises(StageGateError, match="nonminimal"):
        kernel._device_namespace(tmp_path)


def test_assigned_resource_json_has_no_live_authority():
    assigned = AssignedResources(tuple(sorted(os.sched_getaffinity(0)))[:1], 4096, 4096, 1, author_cores=0)
    assigned.verify()
    assert "external_boundary" not in asdict(assigned)
    assert canonical_json(asdict(assigned))
    with pytest.raises(StageGateError, match="enforcing envelope"):
        replace(assigned, require_hard_limits=True).verify()
    with pytest.raises(StageGateError, match="exact live-issued"):
        assigned.verify(external_boundary={"memory.max": 4096})


def test_scratch_plan_retains_only_original_writable_grants(tmp_path):
    argv = [
        "bwrap",
        "--unshare-all",
        "--ro-bind",
        "/usr",
        "/usr",
        "--bind",
        str(tmp_path),
        "/output",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--",
        "/usr/bin/true",
    ]
    planned, index, writable = resources._scratch_plan(argv)
    assert planned == argv and planned is not argv
    assert planned[index : index + 2] == ["--tmpfs", "/tmp"] and writable == (tmp_path,)
    with pytest.raises(StageGateError, match="another unbounded"):
        resources._scratch_plan(["bwrap", "--dev", "/dev", "--tmpfs", "/elsewhere", "--", "true"])
    with pytest.raises(StageGateError, match="unsupported"):
        resources._scratch_plan(["bwrap", "--share-net", "--dev", "/dev", "--tmpfs", "/tmp", "--", "true"])
    owner = tmp_path / "unused"
    with resources.bounded_tool_scratch(argv, owner=owner) as actual:
        assert actual is argv and not owner.exists()


def test_external_native_start_refuses_before_any_service_io(monkeypatch):
    """Invalid active selection only tests the fail-closed guard, never issuance."""
    monkeypatch.setattr(resources, "active_resource_boundary", lambda: object())
    with pytest.raises(StageGateError, match="unbound external native"):
        chia_native._exchange("/unselected-service", {"operation": "start"})


def test_external_container_refuses_before_owner_or_service_io(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "active_resource_boundary", lambda: object())
    unissued = object.__new__(PreparedContainerTransport)
    with pytest.raises(StageGateError, match="unbound external container"):
        unissued.execute(mounts=(), command=("true",), cwd="/", evidence_root=tmp_path)
