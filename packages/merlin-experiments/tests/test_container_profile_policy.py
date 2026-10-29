"""Selected environment/mask metadata is exact data, never runtime authority."""

import copy
import os

import pytest
from merlin_experiments.execution.container_policy import (
    _verify_created_container,
    prepare_strict_policy,
    selected_environment,
)
from merlin_experiments.phase2.contracts import StageGateError
from test_container_transport import original_policy


def test_original_answer_masks_and_explicit_environment_are_retained(tmp_path):
    policy = original_policy(tmp_path)
    workspace = tmp_path / "masks"
    workspace.mkdir()
    environment = (
        ("HOME", "/tmp"),
        ("PATH", "/usr/bin:/bin"),
        ("PYTHONPATH", "/component-inputs/compiler"),
        ("PYTHONDONTWRITEBYTECODE", "1"),
        ("PYTHONNOUSERSITE", "1"),
        ("MERLIN_COMMON_WORKSPACE", "/isolated-workspace"),
    )
    policy += (
        "--setenv",
        "MERLIN_COMMON_WORKSPACE",
        "/isolated-workspace",
        "--tmpfs",
        "/protected-directory",
        "--ro-bind",
        "/dev/null",
        "/public/protected-file",
    )
    plan = prepare_strict_policy(policy, original_environment=environment, mask_workspace=workspace)
    assert plan.environment == environment and plan.namespace_masks == ("/protected-directory",)
    mask = next(row for row in plan.mounts if row.destination == "/public/protected-file")
    assert mask.read_only and mask.source.read_bytes() == b""
    mask.source.write_text("unapproved file contents")
    with pytest.raises(StageGateError, match="closed original mount"):
        mask.verify()


@pytest.mark.parametrize("values", [(("X", "a"), ("X", "b")), (("X", "a\x00b"),), (("X", "a" * 8193),)])
def test_environment_cannot_hide_duplicates_or_unbounded_values(values):
    with pytest.raises(StageGateError):
        selected_environment(values)


def test_created_container_env_and_directory_masks_must_match_exact_plan():
    environment = (("PYTHONPATH", "/exact-public-roots"), ("MERLIN_COMMON_WORKSPACE", "/isolated-workspace"))
    masks = ("/protected-directory",)
    command, cwd, image, timeout = ("/candidate/tool",), "/candidate", "sha256:" + "1" * 64, 3
    host = {
        "NetworkMode": "none",
        "PidMode": "",
        "IpcMode": "private",
        "Privileged": False,
        "ReadonlyRootfs": True,
        "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges=true"],
        "PidsLimit": 64,
        "Memory": 256 * 1024**2,
        "NanoCpus": 1000000000,
        "AutoRemove": False,
        "CgroupnsMode": "private",
        "RestartPolicy": {"Name": "no"},
        "Tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=64m", masks[0]: "ro,nosuid,nodev,noexec,size=16m"},
    }
    row = {
        "Image": image,
        "HostConfig": host,
        "Mounts": [],
        "Config": {
            "User": f"{os.getuid()}:{os.getgid()}",
            "WorkingDir": cwd,
            "Entrypoint": ["/usr/bin/merlin-command-owner"],
            "Cmd": [str(timeout), "--", *command],
            "Env": [key + "=" + value for key, value in environment],
        },
    }
    options = dict(
        image_id=image, mounts=(), command=command, cwd=cwd, timeout_s=timeout, masks=masks, environment=environment
    )
    _verify_created_container(row, **options)
    for field in ("environment", "directory mask"):
        mutated = copy.deepcopy(row)
        if field == "environment":
            mutated["Config"]["Env"][0] = "PYTHONPATH=/foreign-answer-root"
        else:
            mutated["HostConfig"]["Tmpfs"].pop(masks[0])
        with pytest.raises(StageGateError, match="original"):
            _verify_created_container(mutated, **options)
