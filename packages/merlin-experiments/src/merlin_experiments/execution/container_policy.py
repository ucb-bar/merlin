"""Closed data grants and exact ordinary compiler tool-policy conversion.

No raw policy or mount inventory grants authoring/runtime authority. Callers own
original independent admission. Existing transport metadata is compared only to
its actual supplied create plan, never interpreted as kernel isolation proof.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from merlin_experiments.phase2.contracts import StageGateError


def _tree(path):
    path = Path(path)
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise StageGateError("container mount source is absent or linked")
    rows = []
    for member in [path] if path.is_file() else sorted(path.rglob("*")):
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise StageGateError("container mount contains linked or special files")
        if member.is_file():
            rows.append(
                (
                    str(member.relative_to(path)) if path.is_dir() else ".",
                    hashlib.sha256(member.read_bytes()).hexdigest(),
                )
            )
        else:
            rows.append((str(member.relative_to(path)), "directory"))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ContainerMount:
    source: Path
    destination: str
    read_only: bool
    membership_sha256: str

    def verify(self):
        target = PurePosixPath(self.destination)
        if (
            not self.source.is_absolute()
            or not target.is_absolute()
            or str(target) != self.destination
            or target == PurePosixPath("/")
            or ".." in target.parts
            or any(char in str(self.source) + self.destination for char in ",\n\r\x00")
            or type(self.read_only) is not bool
            or _tree(self.source) != self.membership_sha256
        ):
            raise StageGateError("container grant differs from its closed original mount")


def mount_from_source(source: Path, destination: str, *, read_only: bool) -> ContainerMount:
    """Pin explicit data only. Callers retain their ordinary grant admission."""
    if Path(source).is_symlink():
        raise StageGateError("container grant cannot resolve a linked owner")
    actual = Path(source).resolve(strict=True)
    result = ContainerMount(actual, destination, read_only, _tree(actual))
    result.verify()
    return result


def selected_environment(values):
    """Validate bounded original process data, never grant an environment policy."""
    if type(values) is not tuple or len(values) > 32:
        raise StageGateError("selected container environment requires bounded immutable pairs")
    result = {}
    for row in values:
        if type(row) is not tuple or len(row) != 2:
            raise StageGateError("selected container environment requires exact name/value pairs")
        key, value = row
        if (
            type(key) is not str
            or not key
            or len(key) > 128
            or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in key)
            or type(value) is not str
            or len(value.encode()) > 8192
            or any(c in value for c in "\x00\n\r")
            or key in result
        ):
            raise StageGateError("selected container environment contains invalid or duplicate data")
        result[key] = value
    return result


def mounts_from_strict_policy(
    argv: tuple[str, ...],
    *,
    original_environment: tuple[tuple[str, str], ...] | None = None,
) -> tuple[ContainerMount, ...]:
    mounts, _ = _convert_strict_policy(argv, original_environment=original_environment)
    return mounts


@dataclass(frozen=True)
class ContainerPolicyPlan:
    mounts: tuple[ContainerMount, ...]
    namespace_masks: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]


def prepare_strict_policy(argv, *, original_environment, mask_workspace):
    """Consume explicit original masks without granting a broader source view."""
    expected = selected_environment(original_environment)
    root = Path(mask_workspace)
    if root.is_symlink() or root.resolve() != root or not root.is_dir() or any(root.iterdir()):
        raise StageGateError("container masks require an empty canonical private owner")
    mounts, masks = _convert_strict_policy(argv, original_environment=original_environment, mask_workspace=root)
    return ContainerPolicyPlan(mounts, masks, tuple(expected.items()))


def _convert_strict_policy(argv, *, original_environment=None, mask_workspace=None):
    """Consume the exact current component tool policy language, without executing it.

    The caller must verify the original ComponentView/runtime/candidate owners;
    this conversion neither creates those grants nor admits a raw caller policy.
    Unsupported/environment/network options refuse before any service request.
    """
    switches = {"--die-with-parent", "--new-session", "--unshare-all", "--clearenv"}
    scopes = {"--proc": "/proc", "--dev": "/dev", "--tmpfs": "/tmp"}
    environment = {
        "HOME": "/tmp",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": "/component-inputs/compiler",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if original_environment is not None:
        environment = selected_environment(original_environment)
    if not argv:
        raise StageGateError("container conversion needs the verified original tool policy")
    observed, seen_env, seen_scopes, mounts, cwd, masks = set(), {}, {}, [], None, []
    index = 1
    try:
        while index < len(argv):
            flag = argv[index]
            if flag in switches:
                if flag in observed:
                    raise StageGateError("duplicate original isolation switch")
                observed.add(flag)
                index += 1
            elif flag == "--setenv":
                key, value = argv[index + 1 : index + 3]
                if key in seen_env:
                    raise StageGateError("duplicate original environment grant")
                seen_env[key] = value
                index += 3
            elif flag in scopes:
                if flag == "--tmpfs" and argv[index + 1] != scopes[flag]:
                    if mask_workspace is None:
                        raise StageGateError("original namespace mask has no explicit container owner")
                    target = PurePosixPath(argv[index + 1])
                    if (
                        not target.is_absolute()
                        or str(target) != argv[index + 1]
                        or ".." in target.parts
                        or str(target) in {"/", "/dev", "/proc", "/sys"}
                    ):
                        raise StageGateError("unsupported original namespace mask")
                    if str(target) in masks or len(masks) >= 256:
                        raise StageGateError("duplicate or excessive original namespace masks")
                    masks.append(str(target))
                    index += 2
                    continue
                if flag in seen_scopes:
                    raise StageGateError("duplicate original namespace grant")
                seen_scopes[flag] = argv[index + 1]
                index += 2
            elif flag in ("--ro-bind", "--bind"):
                source, destination = argv[index + 1 : index + 3]
                if source == "/dev/null" and flag == "--ro-bind" and mask_workspace is not None:
                    if len(mounts) >= 2048:
                        raise StageGateError("original container mask/mount roster exceeds its bound")
                    empty = mask_workspace / "empty-mask"
                    if not empty.exists():
                        empty.write_bytes(b"")
                    source = str(empty)
                mounts.append(mount_from_source(Path(source), destination, read_only=flag == "--ro-bind"))
                if len(mounts) > 2048:
                    raise StageGateError("original container mount roster exceeds its bound")
                index += 3
            elif flag == "--chdir":
                if cwd is not None:
                    raise StageGateError("duplicate original working directory")
                cwd = argv[index + 1]
                index += 2
            else:
                raise StageGateError("unsupported original tool policy option")
    except (IndexError, ValueError) as error:
        raise StageGateError("incomplete original tool policy grammar") from error
    if observed != switches or seen_env != environment or seen_scopes != scopes:
        raise StageGateError("original tool policy is missing its closed isolation/environment scopes")
    if cwd is None or cwd not in {row.destination for row in mounts}:
        raise StageGateError("original tool policy lacks its exact mounted working directory")
    return tuple(mounts), tuple(masks)


def verify_owned_container(row, *, image_id, name, token):
    """Never inspect, signal or remove a foreign delegated command."""
    identity = row.get("Id", "")
    if (
        len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
        or row.get("Name") != "/" + name
        or row.get("Image") != image_id
        or row.get("Config", {}).get("Labels", {}).get("merlin.command-owner") != token
    ):
        raise StageGateError("service did not return this invocation's exact created container")
    return identity


def _verify_created_container(row, *, image_id, mounts, command, cwd, timeout_s, masks, environment=None):
    config, host = row["Config"], row["HostConfig"]
    expected = {
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
    }
    if (
        row["Image"] != image_id
        or any(host.get(key) != value for key, value in expected.items())
        or any(host.get(key) for key in ("CapAdd", "Binds", "Devices", "DeviceRequests", "VolumesFrom"))
        or host.get("RestartPolicy", {}).get("Name") != "no"
        or config.get("User") != f"{os.getuid()}:{os.getgid()}"
        or config.get("WorkingDir") != cwd
        or config.get("Entrypoint") != ["/usr/bin/merlin-command-owner"]
        or config.get("Cmd") != [str(timeout_s), "--", *command]
        or host.get("Tmpfs")
        != {
            "/tmp": "rw,nosuid,nodev,noexec,size=64m",
            **{path: "ro,nosuid,nodev,noexec,size=16m" for path in masks},
        }
        or {(item["Source"], item["Destination"], item["RW"]) for item in row["Mounts"]}
        != {(str(item.source), item.destination, not item.read_only) for item in mounts}
    ):
        raise StageGateError("owned container differs from its exact original isolation/mount/command plan")
    if environment is not None:
        actual = config.get("Env")
        if type(actual) is not list or any(type(value) is not str or "=" not in value for value in actual):
            raise StageGateError("owned container omitted its original selected environment")
        observed = selected_environment(tuple(tuple(value.split("=", 1)) for value in actual))
        if observed != selected_environment(environment):
            raise StageGateError("owned container changed its original selected environment")
